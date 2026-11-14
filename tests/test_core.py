import copy
import unittest
import torch
from shapnet.models import ImportanceStudent, build_mil
from shapnet.streaming import backward_bag, score_bag, select_indices, full_context, block_ranges
from shapnet.objectives import objective, sample_pairs, normalize_targets
from shapnet.metrics import macro_ranking, ranking_metrics, binary_metrics

torch.set_num_threads(1)
class CoreTests(unittest.TestCase):
    def test_replay_matches_reference_gradients(self):
        for kind in ['patch_mlp','context_mlp','transformer']:
            for loss_kind in ['mse','pairwise']:
                torch.manual_seed(7)
                a=ImportanceStudent(8,kind,16,4,1); b=copy.deepcopy(a)
                x=torch.randn(13,8); y=torch.rand(13); pairs=sample_pairs(13,100,5)
                context=full_context(x,'cpu')
                pred=torch.cat([a(x[s:e],context) for s,e in block_ranges(13,5)])
                objective(pred,y,loss_kind,pairs).backward()
                backward_bag(b,x,y,5,'cpu',loss_kind,pairs)
                for p,q in zip(a.parameters(),b.parameters()):
                    torch.testing.assert_close(p.grad,q.grad,atol=2e-6,rtol=2e-5)
    def test_patch_chunk_invariance(self):
        model=ImportanceStudent(8,'patch_mlp',16)
        x=torch.randn(17,8)
        torch.testing.assert_close(score_bag(model,x,5,'cpu'),score_bag(model,x,17,'cpu'))
    def test_full_sampling_preserves_order(self):
        self.assertEqual(select_indices(torch.tensor([.3,.8,.1]),1.).tolist(),[0,1,2])
        self.assertEqual(select_indices(torch.tensor([.3,.8,.1]),.5,min_patches=1).tolist(),[1])
    def test_constant_target_has_no_fake_ranks(self):
        self.assertEqual(len(torch.unique(normalize_targets(torch.ones(5),'quantile'))),1)
        self.assertIsNone(ranking_metrics(torch.ones(5),torch.ones(5))['spearman'])
    def test_fixed_threshold(self):
        m=binary_metrics([0,1],[.7,.8],.5)
        self.assertEqual(m['accuracy'],.5)
    def test_mil_forward_and_gradient(self):
        for name in ['attmil','meanmil','maxmil','dsmil','transmil','rrtmil']:
            with self.subTest(model=name):
                model=build_mil(dict(name=name,input_dim=8,dropout=0.))
                x=torch.randn(1,20,8,requires_grad=True)
                model.eval(); logits=model(x)
                self.assertEqual(tuple(logits.shape),(1,2))
                g,=torch.autograd.grad(logits[0,1],x)
                self.assertTrue(torch.isfinite(g).all())
                model.train(); loss=model.training_loss(x.detach(),torch.tensor([1])); loss.backward()

if __name__=='__main__': unittest.main()
