import unittest,torch
from shapnet.optim import make_optimizer,make_scheduler
class OptimTests(unittest.TestCase):
 def test_explicit_decay_and_cosine(self):
  p=torch.nn.Parameter(torch.ones(1));opt=make_optimizer([p],'adam',2e-4,1e-5)
  self.assertIsInstance(opt,torch.optim.Adam)
  self.assertEqual(opt.param_groups[0]['weight_decay'],1e-5)
  sched=make_scheduler(opt,'cosine',100)
  for _ in range(100): opt.step();sched.step()
  self.assertAlmostEqual(opt.param_groups[0]['lr'],0.)
 def test_student_schedule_floor(self):
  p=torch.nn.Parameter(torch.ones(1));opt=make_optimizer([p],'adamw',1e-3,1e-4)
  sched=make_scheduler(opt,'cosine',50,.01)
  for _ in range(50): opt.step();sched.step()
  self.assertAlmostEqual(opt.param_groups[0]['lr'],1e-5)
