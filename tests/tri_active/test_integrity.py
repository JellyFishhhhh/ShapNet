import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import torch
from shapnet.io import digest
from trimorph.data import Slide
from trimorph.integrity import feature_digests


class IntegrityTests(unittest.TestCase):
    def test_cache_reuse_and_change_with_restored_mtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); feature=root/'x.pt'; cache=root/'cache.json'
            torch.save(torch.ones(3,4),feature)
            slide=Slide('s','p',str(feature))
            # Ensure the first cached file predates the conservative timestamp window.
            time.sleep(1.05)
            original=feature_digests([slide],cache)
            with patch('trimorph.integrity.digest',side_effect=AssertionError('Must reuse unchanged hash')):
                self.assertEqual(feature_digests([slide],cache),original)
            stat=feature.stat()
            torch.save(torch.zeros(3,4),feature)
            os.utime(feature,ns=(stat.st_atime_ns,stat.st_mtime_ns))
            changed=feature_digests([slide],cache)
            self.assertNotEqual(original,changed)
            self.assertEqual(changed['s'],digest(feature))

    def test_recent_same_tick_write_cannot_reuse_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); feature=root/'x.pt'; cache=root/'cache.json'
            torch.save(torch.ones(3,4),feature)
            slide=Slide('s','p',str(feature))
            before=feature_digests([slide],cache)
            stat=feature.stat()
            torch.save(torch.zeros(3,4),feature)
            os.utime(feature,ns=(stat.st_atime_ns,stat.st_mtime_ns))
            after=feature_digests([slide],cache)
            self.assertNotEqual(before,after)
