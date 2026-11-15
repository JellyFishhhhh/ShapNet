# ShapNet

Feature-based binary WSI classification and patch-importance distillation, refactored from RRT-MIL. Original files and datasets are not modified.

## Run

Use Python 3.12 with PyTorch, NumPy, SciPy, scikit-learn; RRTMIL/TransMIL additionally require timm/einops. The existing server `RRT` conda environment supplies these. No installation was performed during development.

From this repository directory:

```bash
python -m unittest discover -s tests -v
python -m shapnet --config configs/lung_cancer.json
# Reuse completed artifacts; never silently restart training:
python -m shapnet --config configs/lung_cancer.json --stages evaluate
```

Prepare `data/manifest.csv` with a header:

```csv
slide_id,patient_id,label,feature_path
slide001,patient001,0,/absolute/path/slide001.pt
slide002,patient002,1,/absolute/path/slide002.pt
```

Each `.pt` is a finite float tensor `[N,D]`. Provide real patient IDs explicitly, not guessed from slide names. Both classes and enough patients for stratified train/validation/test splits are required. Paths in configuration are relative to the current working directory; feature paths are relative to the manifest. Use a new output directory for each changed configuration. Edit `device` to `cpu` on CPU hosts.

Stages: `teacher,targets,student,evaluate`, always executed in that order. Later stages require earlier artifacts. Existing training artifacts are never overwritten. This supports stage reuse, not optimizer/epoch resume. Teacher training uses one full WSI per update and can still OOM for very large bags; this release does not alter teacher context by silently subsampling it.

## Scientific contract

- Patient-disjoint persisted folds; a separate validation set is mandatory. Student validation uses that held-out patient set.
- Importance is **positive-class logit input-gradient L2 norm**, not exact Shapley values. Failures stop the run; samples are never silently discarded.
- Students: `patch_mlp`, `context_mlp`, `transformer`. LayerNorm replaces BatchNorm. Dropout is zero to permit exact deterministic recomputation.
- Full-bag mean is explicit context. Transformer attention is confined to deterministic contiguous blocks: **not equivalent to full-bag self-attention**. Block size is an experimental hyperparameter and identical during train/validation/inference.
- Two-pass backprop retains scalar predictions on CPU, then recomputes one block at a time. Global sampled pairwise comparisons cross block boundaries. Tests compare gradients with a reference retained-graph calculation.
- Losses: MSE, Huber, pairwise logistic (including the alias `margin_pairwise_logistic`). Unsupported historical ranking losses are rejected. The low-level BCE helper is available, but CLI binary-target training is not supported.
- Each bag has one optimizer update; all patches contribute to regression, sampled global pairs to ranking. No hidden forward subsampling.
- Ranking metrics are computed per WSI then macro-averaged; undefined correlations are reported as null with valid-bag counts.
- Top-K preserves original relative order, including 100% retention. Removing patches still changes positional layouts for RRTMIL/TransMIL.
- Each inference mode calibrates its threshold on validation data only. Test labels are never used to choose thresholds.
- Teacher and feature SHA256 checks prevent stale target reuse. Config, environment, splits, histories and checkpoints are saved locally.

MIL models: AttMIL, RRTMIL, TransMIL, DSMIL, MeanMIL, MaxMIL. Selected upstream implementations are vendored with source hashes. CLAM, survival, legacy student weights, SHAP Kernel/Deep explainers, spatial visualization and efficiency benchmarking are not included in this first refactor. New students require retraining. `MILClassifier.load_legacy` supports strict upstream MIL state loading, but old checkpoints' provenance and split identities must be checked externally.

## Validation limits

CPU synthetic tests cover numerical gradient equivalence, all six MIL forward/backward paths, deterministic sampling and a complete train→targets→student→test→reload run. Passing tests does not establish GPU memory limits, convergence, classification performance, or clinical validity. Real-data manifests and weights are intentionally excluded from Git.

## Aligned hyperparameter run

`configs/lung_cancer_rrtmil_aligned.json` selects Fold 0 with RRTMIL region_num=16, Adam lr=2e-4 / weight_decay=1e-5 / cosine; student architecture and pairwise objective are unchanged. Student AdamW uses lr=1e-3 / weight_decay=1e-4 / cosine to 1e-5, noise augmentation, clip=5 and patience=10. Validation fraction is 0.1. The student remains context_mlp (128), with 4096 sampled global pairs, not the historical Transformer. Patient isolation and validation-only thresholds remain in force. A local full PT/H5 validation is required before this run.

## Runtime integrity and teacher restart

Optional manifest column `feature_sha256` pins every feature file to prevalidated bytes. Every read verifies and deserializes the same byte buffer; three bounded retries allow a transient bad read to recover, emitting a warning. Persistent corruption is never ignored or filled. Teacher `teacher_last.pt` now stores model, optimizer, scheduler, epoch, history and torch CPU/CUDA RNG state after every epoch. Use `--resume` with a teacher-stage run to continue from that checkpoint. Historical runs without this artifact cannot be resumed exactly. Student epoch resume is not implemented.
