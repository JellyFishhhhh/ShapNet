import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch

from shapnet.io import load_checkpoint
from trimorph.demo import generate
from trimorph.runner import run


class RunnerTests(unittest.TestCase):
    def test_multiround_step_resume_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=json.loads(generate(Path(tmp)/'demo').read_text())
            c['output_dir']=str(Path(tmp)/'uninterrupted')
            with patch('trimorph.runner.export_curves'):
                a=run(c)
                c['output_dir']=str(Path(tmp)/'resumed')
                self.assertEqual(run(c,max_updates=2)['status'],'bounded')
                # Repeated pause inside optimizer steps, then a round boundary pause.
                self.assertEqual(run(c,resume=True,max_updates=3)['status'],'bounded')
                self.assertEqual(run(c,resume=True,max_rounds=1)['status'],'bounded')
                b=run(c,resume=True)
            self.assertEqual(a,b)
            first=load_checkpoint(Path(tmp)/'uninterrupted/state.pt')
            second=load_checkpoint(Path(tmp)/'resumed/state.pt')
            self.assertEqual(first['audit'],second['audit'])
            self.assertEqual(len(first['history']),3)
            self.assertEqual(first['history'][0]['queried'],6)
            for i,(left,right) in enumerate(zip(first['history'],second['history'])):
                self.assertEqual(left['test'],right['test'])
                self.assertEqual(left['next_query'],right['next_query'])
                self.assertTrue(set(left['morphology_fit_patients'])<=set(left['queried_patients']))
                self.assertGreater(left['training_synthetic_bags'],0)
                cp1=load_checkpoint(Path(tmp)/f'uninterrupted/round_{i:03d}/model.pt')
                cp2=load_checkpoint(Path(tmp)/f'resumed/round_{i:03d}/model.pt')
                for k in cp1['model']:
                    torch.testing.assert_close(cp1['model'][k],cp2['model'][k],rtol=0,atol=0)
            with patch('trimorph.runner.export_curves'):
                self.assertEqual(run(c,resume=True),b)
            with self.assertRaises(FileExistsError): run(c)
            c['acquisition']['alpha']=2
            with self.assertRaisesRegex(ValueError,'identity changed'): run(c,resume=True)

    def test_unrevealed_labels_cannot_change_first_round(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); c=json.loads(generate(root/'demo').read_text())
            c['budget']['ensure_initial_classes']=False
            c['output_dir']=str(root/'a')
            with patch('trimorph.runner.export_curves'): run(c,max_rounds=1)
            state=load_checkpoint(root/'a/state.pt')
            splits=json.loads((root/'a/splits.json').read_text())
            # Preserve splits; permute only labels unavailable before first acquisition.
            initially_queried=set(state['history'][0]['queried_patients'])
            hidden=set(splits['train'])-initially_queried
            manifest=Path(c['manifest'])
            with manifest.open() as f: rows=list(csv.DictReader(f))
            for row in rows:
                if row['patient_id'] in hidden or row['patient_id'] in splits['test']:
                    row['label']=str(1-int(row['label']))
                if row['patient_id'] in splits['test']:
                    feature=manifest.parent/row['feature_path']
                    torch.save(torch.load(feature,weights_only=True)+1000,feature)
            with manifest.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
            c['splits_file']=str(root/'a/splits.json'); c['output_dir']=str(root/'b')
            with patch('trimorph.runner.export_curves'): run(c,max_rounds=1)
            left=json.loads((root/'a/round_000/acquisition.json').read_text())
            right=json.loads((root/'b/round_000/acquisition.json').read_text())
            self.assertEqual(left,right)
            a=load_checkpoint(root/'a/round_000/model.pt')['model']
            b=load_checkpoint(root/'b/round_000/model.pt')['model']
            for k in a: torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)

    def test_controls_random_entropy_disagreement_no_synthesis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); c=json.loads(generate(root/'demo').read_text())
            c['synthesis']['enabled']=False
            for strategy in ('random','entropy','disagreement','coverage'):
                c['acquisition']['strategy']=strategy; c['output_dir']=str(root/strategy)
                with patch('trimorph.runner.export_curves'): run(c,max_rounds=1)
                state=load_checkpoint(root/strategy/'state.pt')
                self.assertEqual(state['history'][0]['synthetic_bags'],0)
                selected=state['history'][0]['next_query']
                self.assertEqual(len(selected),3)
                self.assertTrue(set(selected)<=set(state['splits']['train']))
