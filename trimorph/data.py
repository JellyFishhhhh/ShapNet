"""Setup-only oracle and label-free slide records; identities are never inferred."""
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class Slide:
    slide_id: str
    patient_id: str
    feature_path: str


class Oracle:
    """Only query() releases training labels. Evaluation labels have separate access."""
    def __init__(self, labels, splits, revealed=None, audit=None):
        self.__labels = dict(labels)
        self.splits = {k: tuple(sorted(v)) for k, v in splits.items()}
        self.revealed = dict(revealed or {})
        self.audit = list(audit or [])
        if not set(self.revealed) <= set(self.splits['train']):
            raise ValueError('Revealed labels outside training pool')
        if any(self.__labels[p] != y for p, y in self.revealed.items()):
            raise ValueError('Revealed label mismatch')

    def query(self, patients, round_index, reason='acquisition'):
        patients = list(patients)
        if len(set(patients)) != len(patients):
            raise ValueError('Duplicate query')
        if not set(patients) <= set(self.splits['train']) - set(self.revealed):
            raise ValueError('Queries must be unqueried real training patients')
        for p in patients:
            self.revealed[p] = self.__labels[p]
            self.audit.append(dict(patient_id=p, round=round_index, reason=reason))
        return {p: self.revealed[p] for p in patients}

    def evaluation_labels(self, split):
        if split not in ('val', 'test'):
            raise ValueError('Only held-out evaluation labels are accessible here')
        return {p: self.__labels[p] for p in self.splits[split]}


def read_cohort(path):
    path = Path(path).resolve()
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        if not {'slide_id', 'patient_id', 'label', 'feature_path'} <= set(reader.fieldnames or []):
            raise ValueError('Manifest requires explicit slide_id,patient_id,label,feature_path; patient IDs cannot be guessed')
        rows = list(reader)
    slides, labels, seen, paths = [], {}, set(), set()
    for r in rows:
        sid, pid = r['slide_id'].strip(), r['patient_id'].strip()
        y = int(r['label'])
        feature = (path.parent / r['feature_path']).resolve()
        if not sid or not pid or sid in seen or feature in paths or y not in (0, 1):
            raise ValueError('Invalid/duplicate identity, reused feature path, or nonbinary label')
        if pid in labels and labels[pid] != y:
            raise ValueError(f'Conflicting slide labels for patient {pid}; manual resolution required')
        if not feature.is_file():
            raise FileNotFoundError(feature)
        labels[pid] = y
        slides.append(Slide(sid, pid, str(feature)))
        seen.add(sid)
        paths.add(feature)
    if not slides:
        raise ValueError('Empty cohort')
    return sorted(slides, key=lambda x: x.slide_id), labels


def make_splits(labels, seed=2026, val_fraction=.2, test_fraction=.2):
    """Stratification is setup-only, never an acquisition input."""
    if not 0 < val_fraction < 1 or not 0 < test_fraction < 1 - val_fraction:
        raise ValueError('Invalid validation/test fractions')
    ids = sorted(labels)
    trainval, test = train_test_split(ids, test_size=test_fraction, random_state=seed,
                                     stratify=[labels[p] for p in ids])
    train, val = train_test_split(trainval, test_size=val_fraction / (1-test_fraction),
                                 random_state=seed+1, stratify=[labels[p] for p in trainval])
    return validate_splits(dict(train=train, val=val, test=test), labels)


def validate_splits(splits, labels):
    if set(splits) != {'train', 'val', 'test'}:
        raise ValueError('Splits require train, val, test patient lists')
    seen = set()
    for name, ids in splits.items():
        if not ids or len(ids) != len(set(ids)) or seen.intersection(ids):
            raise ValueError('Empty/duplicate/overlapping patient splits')
        if not set(ids) <= set(labels):
            raise ValueError('Unknown split patient')
        if {labels[p] for p in ids} != {0, 1}:
            raise ValueError(f'Both classes required in {name}')
        seen.update(ids)
    if seen != set(labels):
        raise ValueError('Splits must partition the entire cohort')
    return {k: sorted(v) for k, v in splits.items()}


class FeatureStore:
    """No labels, no unbounded feature cache. H5 is read only, PT uses weights_only."""
    def __init__(self, slides, input_dim, memory_map=False):
        self.slides = {s.slide_id: s for s in slides}
        self.input_dim = input_dim
        self.memory_map = memory_map
        self.access_log = []

    def load(self, slide_id):
        s = self.slides[slide_id]
        self.access_log.append(s.patient_id)
        if Path(s.feature_path).suffix.lower() in ('.h5', '.hdf5'):
            import h5py
            with h5py.File(s.feature_path, 'r') as f:
                x = torch.from_numpy(f['features'][:])
        else:
            x = torch.load(s.feature_path, map_location='cpu', weights_only=True, mmap=self.memory_map)
        if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[0] == 0 or x.shape[1] != self.input_dim:
            raise ValueError(f'Expected nonempty [N,{self.input_dim}] features: {slide_id}')
        x = x.float()
        if not torch.isfinite(x).all():
            raise ValueError(f'Nonfinite features: {slide_id}')
        return x

    def for_patients(self, patients):
        patients = set(patients)
        return [s for s in self.slides.values() if s.patient_id in patients]

    def identity(self):
        return [asdict(s) for s in self.slides.values()]


def budget_schedule(n, initial=.15, step=.1, maximum=.5):
    def count(v):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0:
            raise ValueError('Budget must be positive integer patients or fraction (0,1]')
        if isinstance(v, float) and v > 1:
            raise ValueError('Use integers for absolute patient counts')
        return int(np.ceil(v*n)) if isinstance(v, float) else v
    start, delta, end = map(count, (initial, step, maximum))
    if not 1 <= start <= end <= n:
        raise ValueError('Require 1 <= initial <= maximum <= training patients')
    return sorted(set(list(range(start, end+1, delta)) + [end]))
