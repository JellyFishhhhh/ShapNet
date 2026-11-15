import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch

from shapnet.io import load_checkpoint
from trimorph.data import FeatureStore, Slide, read_cohort, make_splits
from trimorph.demo import generate
from trimorph.evaluation import validation_threshold, metrics, evaluate as real_evaluate
from trimorph.runner import run


class TrainingProtocolTests(unittest.TestCase):
    def test_validation_threshold_and_auc_invariance(self):
        y=[0,0,1,1]; p=[.1,.2,.3,.4]
        threshold=validation_threshold(y,p)
        self.assertAlmostEqual(threshold,.3)
        self.assertEqual(metrics(y,p,threshold)['macro_f1'],1.)
        self.assertEqual(metrics(y,p,.5)['auc'],metrics(y,p,threshold)['auc'])
        self.assertEqual(validation_threshold([0,1],[.2,.8]),.5)
        with self.assertRaises(ValueError): validation_threshold([0,0],[.1,.2])

    def test_pt_memory_mapping_preserves_all_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'features.pt'; x=torch.randn(17,4); torch.save(x,path)
            slides=[Slide('s','p',str(path))]
            torch.testing.assert_close(FeatureStore(slides,4,True).load('s'),
                                       FeatureStore(slides,4,False).load('s'),rtol=0,atol=0)

    def test_early_stop_resume_and_test_label_firewall(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); c=json.loads(generate(root/'demo').read_text())
            c['feature_memory_map']=True
            c['output_dir']=str(root/'continuous')
            c['synthesis']['enabled']=False
            c['budget'].update(initial=1.,maximum=1.)
            c['training'].update(epochs=8,patience=2,threshold_mode='validation_macro_f1')
            _,labels=read_cohort(c['manifest'])
            val_patients=set(make_splits(labels)['val'])
            def constant_val_auc(model,morph,store,patients,labels,threshold=.5):
                result,pred=real_evaluate(model,morph,store,patients,labels,threshold)
                if set(patients)==val_patients: result['slide']['auc']=.5
                return result,pred
            with patch('trimorph.runner.evaluate',side_effect=constant_val_auc),patch('trimorph.runner.export_curves'):
                run(c)
                c['output_dir']=str(root/'resumed')
                run(c,max_updates=2)
                run(c,resume=True)
            a=load_checkpoint(root/'continuous/state.pt'); b=load_checkpoint(root/'resumed/state.pt')
            self.assertEqual(len(a['revealed']),len(a['splits']['train']))
            self.assertEqual(a['history'][0]['test'],b['history'][0]['test'])
            self.assertEqual(a['training_history'],b['training_history'])
            self.assertTrue(a['stopped_early']); self.assertEqual(a['best_epoch'],0)
            self.assertEqual(len(a['training_history']),3)
            for k in a['best']: torch.testing.assert_close(a['best'][k],b['best'][k],rtol=0,atol=0)
            # Flip only test labels, preserving every training/validation identity.
            manifest=Path(c['manifest'])
            with manifest.open() as f: rows=list(csv.DictReader(f))
            for row in rows:
                if row['patient_id'] in a['splits']['test']: row['label']=str(1-int(row['label']))
            with manifest.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
            c['splits_file']=str(root/'continuous/splits.json');c['output_dir']=str(root/'changed_test')
            with patch('trimorph.runner.evaluate',side_effect=constant_val_auc),patch('trimorph.runner.export_curves'):
                run(c)
            d=load_checkpoint(root/'changed_test/state.pt')
            self.assertEqual(a['best_epoch'],d['best_epoch'])
            self.assertEqual(a['history'][0]['selected_threshold'],d['history'][0]['selected_threshold'])
            for k in a['best']: torch.testing.assert_close(a['best'][k],d['best'][k],rtol=0,atol=0)
