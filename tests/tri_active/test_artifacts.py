import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from shapnet.io import write_json,save_checkpoint
from trimorph.demo import generate
from trimorph.evaluation import export_curves
from trimorph.experiments import generate as generate_suite, summarize, VARIANTS
from trimorph.prepare import prepare


class ArtifactTests(unittest.TestCase):
    def test_suite_paired_seeds_and_required_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); config=generate(root/'demo')
            jobs=generate_suite(config,root/'suite')
            self.assertEqual(len(jobs),42)
            self.assertEqual(set(j['variant'] for j in jobs),set(VARIANTS))
            self.assertEqual({j['seed'] for j in jobs},{0,1,2})
            configurations=[json.loads(Path(j['config']).read_text()) for j in jobs]
            self.assertEqual(len({c['split_seed'] for c in configurations}),1)
            self.assertEqual(len({c['output_dir'] for c in configurations}),42)
            self.assertFalse(any(Path(c['output_dir']).exists() for c in configurations))
            self.assertEqual(len(summarize(root/'suite')['missing']),42)

    def test_curve_area_and_common_interval_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            jobs=[]
            for seed,x in ((0,[2,4,6]),(1,[3,5,6])):
                output=root/f'seed{seed}'; output.mkdir()
                rounds=[]
                for i,budget in enumerate(x):
                    value=budget/10
                    metric=dict(auc=value,macro_f1=value)
                    rounds.append(dict(round=i,queried=budget,test=dict(slide=metric,patient=metric),
                                       synthetic_bags=1,seconds=.1,peak_cuda_bytes=0))
                with patch.dict('sys.modules',{'matplotlib':None}): export_curves(output,rounds,10,2)
                area=json.loads((output/'curve_area.json').read_text())
                self.assertAlmostEqual(area['metrics']['slide_auc']['normalized_area'],(x[0]+x[-1])/20)
                save_checkpoint(output/'state.pt',dict(stage='complete'))
                jobs.append(dict(variant='full',seed=seed,output=str(output)))
            write_json(root/'jobs.json',jobs)
            summary=summarize(root)
            self.assertEqual(summary['common_range'],[.3,.6])
            self.assertAlmostEqual(summary['variants']['full']['metrics']['slide_auc']['mean_area'],.45)
            self.assertAlmostEqual(summary['variants']['full']['metrics']['slide_auc']['std_area'],0.)

    def test_prepare_requires_complete_real_patient_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); generate(root/'demo'); features=root/'features'; (features/'pt').mkdir(parents=True)
            (features/'label.csv').write_text('s000_0,0\ns001_0,1\n')
            import shutil
            for sid in ('s000_0','s001_0'):
                shutil.copy(root/'demo'/f'{sid}.pt',features/'pt'/f'{sid}.pt')
            mapping=root/'mapping.csv'; mapping.write_text('slide_id,patient_id\ns000_0,explicit_A\n')
            with self.assertRaisesRegex(ValueError,'Complete exact mapping'):
                prepare(features,mapping,root/'manifest.csv','pt')
            mapping.write_text('slide_id,patient_id\ns000_0,explicit_A\ns001_0,explicit_B\n')
            self.assertEqual(prepare(features,mapping,root/'manifest.csv','pt'),dict(slides=2,patients=2))
