import csv
from dataclasses import fields
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from trimorph.data import FeatureStore, Oracle, Slide, budget_schedule, make_splits, read_cohort, validate_splits
from trimorph.demo import generate
from trimorph.model import TriMorphMIL, supervised_loss
from trimorph.morphology import Morphology
from trimorph.selection import acquire, greedy_coverage, information_value, similarity
from trimorph.synthesis import synthesize, validate_bank


class MethodTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_normalized_entropy_js(self):
        p=torch.tensor([[[1.,0.],[1.,0.],[1.,0.]],[[1.,0.],[0.,1.],[.5,.5]],[[.5,.5]]*3])
        h,js,q=information_value(p)
        np.testing.assert_allclose(h,[0,1,1],atol=1e-12)
        np.testing.assert_allclose(js,[0,2/3,0],atol=1e-12)
        self.assertGreater(q[1],q[2])

    def test_exact_facility_gains_frozen_u(self):
        s=np.array([[1,.4,.1],[.4,1,.7],[.1,.7,1]])
        q=np.array([.2,1,.8]); base=np.array([.3,.1,.2])
        selected,gains=greedy_coverage(['a','b','c'],s,q,base,3)
        chosen=[]; last=0.
        for patient,gain in zip(selected,gains):
            candidates=[i for i in range(3) if i not in chosen]
            def objective(indices):
                return np.dot(q,np.maximum(base,s[:,indices].max(1))-base)
            expected=max(candidates,key=lambda i:objective(chosen+[i]))
            self.assertEqual(patient,['a','b','c'][expected]); chosen.append(expected)
            value=objective(chosen); self.assertAlmostEqual(gain,value-last); last=value
        self.assertGreaterEqual(gains[0],gains[1]); self.assertGreaterEqual(gains[1],gains[2])
        np.testing.assert_array_equal(base,[.3,.1,.2])

    def test_synthetic_discount_and_kernel(self):
        def entry(a):
            return dict(representation=[a,1.],composition=[.5,.5],view_probs=[[.5,.5]]*3)
        u={'a':entry(0),'b':entry(1)}; labeled={'l':entry(3)}; synthetic=[entry(0)]
        _,off=acquire(u,labeled,synthetic,1,synthetic_coverage=False,sigma=.1)
        _,zero=acquire(u,labeled,synthetic,1,eta=0,sigma=.1)
        _,on=acquire(u,labeled,synthetic,1,eta=.7,sigma=.1)
        np.testing.assert_allclose(off['base_coverage'],zero['base_coverage'])
        self.assertGreaterEqual(on['base_coverage'][0],.7)
        self.assertEqual(on['frozen_batch'],True)
        with self.assertRaises(ValueError): acquire(u,labeled,synthetic,1,eta=1)
        k=similarity([[1,2],[2,1]],[[1,0],[0,1]])
        np.testing.assert_allclose(k.diagonal(),1)
        self.assertTrue(((k>=0)&(k<=1)).all())

    def test_oracle_and_identity_barriers(self):
        self.assertNotIn('label',[f.name for f in fields(Slide)])
        oracle=Oracle({'a':0,'b':1,'v':0,'t':1},{'train':['a','b'],'val':['v'],'test':['t']})
        with self.assertRaises(ValueError): oracle.query(['v'],0)
        with self.assertRaises(ValueError): oracle.query(['a','a'],0)
        self.assertEqual(oracle.query(['a'],0),{'a':0})
        with self.assertRaises(ValueError): oracle.query(['a'],1)
        with self.assertRaises(ValueError): oracle.evaluation_labels('train')
        with self.assertRaises(ValueError): validate_splits(dict(train=['a','b'],val=['a'],test=['t']),{'a':0,'b':1,'t':1})
        self.assertEqual(budget_schedule(40),[6,10,14,18,20])
        self.assertEqual(budget_schedule(40,3,4,14),[3,7,11,14])

    def test_conflicting_patient_labels_are_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=generate(tmp)
            c=json.loads(path.read_text()); manifest=Path(c['manifest'])
            with manifest.open() as f: rows=list(csv.DictReader(f))
            rows[1]['patient_id']=rows[0]['patient_id']
            with manifest.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
            with self.assertRaisesRegex(ValueError,'Conflicting'): read_cohort(manifest)

    def test_views_losses_ablations_and_importance_interface(self):
        torch.manual_seed(1)
        morph=Morphology(torch.zeros(4),torch.ones(4),torch.eye(4)[:3],['q'],'queried_only')
        x=torch.randn(17,4)
        for module in ({},{'use_morphology':False},{'use_view_ids':False},{'use_fusion':False},{'three_views':False}):
            model=TriMorphMIL(4,8,4,3,diverse_patches=1,balanced_patches=5,**module)
            out=model(x,morph)
            self.assertEqual(out['view_probs'].shape,(3,2))
            self.assertEqual(len(out['indices'][0]),len(x))
            loss=supervised_loss(out,1); loss.backward()
            self.assertIsNotNone(model.classifier.weight.grad)
            self.assertTrue(torch.isfinite(loss))
        model=TriMorphMIL(4,8,4,3,diverse_patches=0,discriminative_ratio=.1)
        out=model(x,morph,importance=lambda m,x,d:torch.arange(len(x),device=x.device).float())
        self.assertEqual(out['indices'][1].tolist(),[15,16])
        # The only direct CE targets are full and fused outputs; subset logits get no own CE.
        out=model(x,morph); out['view_logits'].retain_grad(); supervised_loss(out,0).backward()
        self.assertEqual(float(out['view_logits'].grad[1:].abs().sum()),0.)

    def test_morphology_and_synthesis_use_only_queried_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            config=generate(tmp)
            slides,labels=read_cohort(Path(tmp)/'manifest.csv')
            store=FeatureStore(slides,8)
            splits=make_splits(labels)
            queried={p:labels[p] for p in splits['train'][:12]}
            morph=Morphology.fit(store,queried,splits['train'],k=3,per_slide=10,max_fit=80)
            self.assertEqual(set(morph.fit_patients),set(queried))
            self.assertTrue(set(store.access_log)<=set(queried))
            model=TriMorphMIL(8,12,6,3)
            for mode in ('matched','ordinary'):
                bank=synthesize(store,queried,model,morph,0,max_patches=12,mode=mode)
                self.assertGreater(len(bank),0)
                validate_bank(bank,queried,store)
                for bag in bank:
                    self.assertLessEqual(bag.actual_replacements,bag.requested_replacements)
                    for i in bag.protected_positions:
                        origin=bag.origins[i]
                        original=store.load(origin['slide_id'])[origin['patch_index']]
                        torch.testing.assert_close(bag.features[i],original)
                self.assertTrue(set(store.access_log)<=set(queried))
                with self.assertRaises(ValueError): validate_bank(bank,{},store)
            none=synthesize(store,{next(iter(queried)):0},model,morph,0)
            self.assertEqual(none,[])
            pool=Morphology.fit(store,queried,splits['train'],k=3,protocol='train_pool_unlabeled')
            self.assertEqual(set(pool.fit_patients),set(splits['train']))

    def test_h5_loader_and_invalid_tensor(self):
        import h5py
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'a.h5'
            with h5py.File(path,'w') as f: f['features']=np.ones((3,4),np.float32)
            store=FeatureStore([Slide('s','p',str(path))],4)
            self.assertEqual(store.load('s').shape,(3,4))
            with h5py.File(path,'w') as f: f['features']=np.array([[np.nan]*4],np.float32)
            with self.assertRaisesRegex(ValueError,'Nonfinite'): store.load('s')
