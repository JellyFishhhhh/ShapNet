import csv,json,tempfile,unittest
from pathlib import Path
import torch
from unittest.mock import patch
from shapnet.io import save_checkpoint,load_checkpoint
from shapnet.__main__ import run
from shapnet.data import read_manifest,make_folds

class PipelineTests(unittest.TestCase):
 def test_end_to_end_and_reload(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp); manifest=root/'manifest.csv'
   with manifest.open('w',newline='') as f:
    writer=csv.writer(f); writer.writerow(['slide_id','patient_id','label','feature_path'])
    for i in range(24):
     x=torch.randn(10+i%4,8)+i%2
     path=root/f'{i}.pt'; torch.save(x,path)
     writer.writerow([f's{i}',f'p{i}',i%2,str(path)])
   rows=read_manifest(manifest)
   self.assertEqual(make_folds(rows,3,.25,7),make_folds(rows,3,.25,7))
   config={'manifest':str(manifest),'output_dir':str(root/'run'),'folds':3,'val_ratio':.25,
    'run_folds':[0],'seed':7,'block_size':5,'mil_epochs':2,'student_epochs':1,
    'mil':{'name':'attmil','input_dim':8,'dropout':0.},
    'student':{'input_dim':8,'kind':'transformer','hidden_dim':16,'n_heads':4,'n_layers':1},
    'student_loss':'pairwise','pair_count':64}
   path=root/'config.json'; path.write_text(json.dumps(config))
   def interrupt_after_checkpoint(checkpoint,value):
    save_checkpoint(checkpoint,value)
    if Path(checkpoint).name=='teacher_last.pt': raise RuntimeError('simulated interruption')
   with patch('shapnet.__main__.save_checkpoint',side_effect=interrupt_after_checkpoint):
    with self.assertRaisesRegex(RuntimeError,'simulated interruption'): run(path,['teacher'])
   run(path,['teacher','targets','student','evaluate'],resume=True)
   reference=dict(config,output_dir=str(root/'reference'))
   reference_path=root/'reference.json';reference_path.write_text(json.dumps(reference))
   run(reference_path,['teacher'])
   actual=load_checkpoint(root/'run/fold_0/teacher_last.pt')['model']
   expected=load_checkpoint(root/'reference/fold_0/teacher_last.pt')['model']
   for key in actual: torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0)
   result=root/'run/fold_0/metrics.json'; before=result.read_text()
   run(path,['evaluate']); self.assertEqual(before,result.read_text())
