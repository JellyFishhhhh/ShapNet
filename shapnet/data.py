"""Explicit patient identities and persisted patient-disjoint folds."""
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

from .io import object_digest


def read_manifest(path):
    path = Path(path).resolve()
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"slide_id", "patient_id", "label", "feature_path"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest requires columns: {sorted(required)}")
        rows = list(reader)
    if not rows:
        raise ValueError("Empty manifest")
    seen, patient_labels = set(), {}
    out = []
    for row in rows:
        slide, patient = row["slide_id"].strip(), row["patient_id"].strip()
        if not slide or not patient or slide in seen:
            raise ValueError("Empty identity or duplicate slide_id")
        label = int(row["label"])
        if label not in (0, 1):
            raise ValueError("This experiment pipeline supports binary labels 0/1 only")
        if patient in patient_labels and patient_labels[patient] != label:
            raise ValueError("Conflicting labels for one patient")
        feature = Path(row["feature_path"])
        if not feature.is_absolute():
            feature = path.parent / feature
        if not feature.is_file():
            raise FileNotFoundError(feature)
        seen.add(slide)
        patient_labels[patient] = label
        out.append(dict(slide_id=slide, patient_id=patient, label=label,
                        feature_path=str(feature.resolve())))
    return sorted(out, key=lambda r: r["slide_id"])


def load_features(row, input_dim=None):
    x = torch.load(row["feature_path"], map_location="cpu", weights_only=True)
    if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[0] == 0:
        raise ValueError(f"Expected nonempty [N,D] feature tensor: {row['slide_id']}")
    if input_dim is not None and x.shape[1] != input_dim:
        raise ValueError(f"Feature dimension mismatch: {row['slide_id']}")
    x = x.float()
    if not torch.isfinite(x).all():
        raise ValueError(f"Nonfinite features: {row['slide_id']}")
    return x


def manifest_identity(rows):
    return object_digest(rows)


def make_folds(rows, n_splits=3, val_ratio=0.2, seed=2021):
    if n_splits < 2 or not 0 < val_ratio < 1:
        raise ValueError("n_splits >= 2 and 0 < val_ratio < 1 are required")
    labels = {r["patient_id"]: r["label"] for r in rows}
    patients = np.array(sorted(labels))
    y = np.array([labels[p] for p in patients])
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for k, (trainval, test) in enumerate(splitter.split(patients, y)):
        train, val = train_test_split(trainval, test_size=val_ratio,
                                      stratify=y[trainval], random_state=seed + k)
        groups = {"train": set(patients[train]), "val": set(patients[val]),
                  "test": set(patients[test])}
        fold = {s: [r["slide_id"] for r in rows if r["patient_id"] in ps]
                for s, ps in groups.items()}
        validate_fold(rows, fold)
        folds.append(fold)
    return {"schema_version": 1, "manifest_sha256": manifest_identity(rows),
            "seed": seed, "n_splits": n_splits, "val_ratio": val_ratio, "folds": folds}


def validate_fold(rows, fold):
    by_id = {r["slide_id"]: r for r in rows}
    all_slides, all_patients = set(), set()
    for name in ("train", "val", "test"):
        ids = fold[name]
        if not ids or len(ids) != len(set(ids)) or any(s not in by_id for s in ids):
            raise ValueError(f"Invalid {name} split")
        patients = {by_id[s]["patient_id"] for s in ids}
        if all_slides.intersection(ids) or all_patients.intersection(patients):
            raise ValueError("Slide/patient leakage between splits")
        if {by_id[s]["label"] for s in ids} != {0, 1}:
            raise ValueError(f"Both classes required in {name}")
        all_slides.update(ids)
        all_patients.update(patients)
    if all_slides != set(by_id):
        raise ValueError("Split must cover every slide exactly once")


def select_rows(rows, ids):
    by_id = {r["slide_id"]: r for r in rows}
    return [by_id[s] for s in ids]
