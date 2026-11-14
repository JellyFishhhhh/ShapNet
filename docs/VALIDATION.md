# Validation performed

2026-11-14, existing RRT conda environment, Python 3.12 / PyTorch 2.5.1+cu124, CPU. No dependency installation.

Command: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v`

7 tests passed (29.624 s). Includes six MIL forward/backward subcases and six numerical replay-gradient comparisons (three students × two objectives), a full synthetic patient-split pipeline, checkpoint reload with identical predictions, sample-order identity, constant-target handling, and fixed-threshold metrics.

See `real_feature_smoke.json` for a bounded CPU test against one existing lung-cancer feature tensor. No patient IDs, real features, manifests, trained weights or private paths are published.

CUDA was unavailable. No full lung-cancer training, GPU memory benchmark, convergence test or predictive-performance validation has been performed. Teacher stages still use full bags. This repository establishes tested CPU execution and explicit scientific contracts, not a guarantee that arbitrary full datasets fit GPU memory.
