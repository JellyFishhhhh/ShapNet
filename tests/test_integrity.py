import hashlib,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from shapnet.data import load_features
class IntegrityTests(unittest.TestCase):
 def test_transient_read_and_persistent_change(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'x.pt';torch.save(torch.ones(3,4),p);raw=p.read_bytes()
   row={'slide_id':'synthetic','feature_path':str(p),'feature_sha256':hashlib.sha256(raw).hexdigest()}
   with patch.object(Path,'read_bytes',side_effect=[b'bad',raw]):
    with self.assertWarns(UserWarning): self.assertEqual(tuple(load_features(row).shape),(3,4))
   with patch.object(Path,'read_bytes',return_value=b'bad'):
    with self.assertRaisesRegex(ValueError,'after 3 reads'): load_features(row)
 def test_nonfinite_never_accepted(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'x.pt';torch.save(torch.tensor([[float('nan')]]),p)
   with self.assertRaisesRegex(ValueError,'after 3 reads'):load_features({'slide_id':'synthetic','feature_path':str(p)})
