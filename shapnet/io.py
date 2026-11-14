"""Artifact identity, atomic writes, RNG control, and safe tensor reads."""
import hashlib
import json
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import torch


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(tmp, path)


def save_checkpoint(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def environment():
    root = Path(__file__).resolve().parents[1]
    def git(*args):
        p = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else None
    return {
        "torch": str(torch.__version__), "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda,
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
    }
