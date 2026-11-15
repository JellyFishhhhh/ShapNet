# TriMorph implementation validation — 2026-11-14

Base: `0036500` on ShapNet main. Work uses an independent clone and branch
`codex/trimorph-patient-active-learning`; no existing source checkout was edited.

Environment: existing RRT conda environment, Python 3.12, torch 2.5.1+cu124,
NumPy 2.1.3, scikit-learn 1.6.0, h5py 3.16.0. Validation ran on CPU, one
thread for the TriMorph runner. No packages were installed. No GPUs were occupied.

## Automated verification

`python -m unittest discover -s tests -v`: **21 tests passed** (7 existing,
14 added), 36.219 seconds in the recorded run. Coverage includes:

- Normalized entropy/JS, exact supplied facility objective and marginal gains,
  discount boundaries, nonnegative kernel and frozen batch accounting.
- Patient-query oracle, duplicate/held-out query rejection, explicit identity,
  conflicting labels, PT/H5 checks, fraction and absolute patient budgets.
- All network module ablations, shared head gradients, zero direct subset CE,
  and the replaceable importance interface.
- Queried-only morphology fitting and source access, explicit train-pool protocol,
  matched/ordinary synthesis, protected patch equality, cross-patient same-class
  provenance, and absence of eligible donors.
- Three-round training and acquisition, repeated optimizer-step pauses and a
  round-boundary pause; resumed queries, metrics and model parameters are exactly
  equal to an uninterrupted CPU run (zero-tolerance tensor comparisons).
- Unrevealed training labels are flipped, and held-out test labels/features are
  changed while preserving patient splits: first-round task-model parameters and
  the full acquisition record remain exactly identical.
- Random, entropy, information-only and coverage controls without synthesis;
  all 42 paired experiment configs, complete explicit mapping validation,
  trapezoidal area and seed summary on a common observed budget interval.
- Completed-run reload and changed-config resume rejection.

The first all-tests invocation found a test-package naming collision; the tests
were moved to `tests/tri_active` and the complete suite above subsequently passed.
The existing vendor timm deprecation warning does not affect these checks.

## CLI smoke validation

A generated cohort contains 40 explicit synthetic patients and 60 slides, each
with 8-dimensional features. Fixed splits contain 24 training-pool patients, 8
validation patients and 8 test patients. This uses deliberately tiny epochs and
is a software validation, not an estimate of medical performance.

- First invocation stops after 2 optimizer updates.
- `--resume` completes 3 rounds, with 6 → 9 → 12 queried training patients.
- Including validation, labeled-patient counts are 14 → 17 → 20.
- Coverage pseudo-bags are 6 → 9 → 12; final update count is 111.
- CSV, PNG, per-run area JSON, initialization, query audit, best round models,
  morphology dictionaries, separate training/coverage banks, provenance and
  invocation wall-time records were emitted.
- Active invocation wall time was approximately 13 seconds including checkpoint
  and plot I/O. This is host-specific and not a real-WSI runtime estimate.

## Bounded real-feature inspection

`/home/users/lsy/WSI/dataset/lung_cancer_features` was accessible:
390 label rows, 390 PT files, 390 H5 files. The first two sorted H5 files have
shapes `[64477,1024]` and `[38794,1024]`. For each, 512 patch rows were checked;
all sampled values were finite and matched the corresponding memory-mapped PT
rows exactly. No task model, normalization or morphology dictionary was fitted
on these real features.

No explicit slide-to-patient mapping was found in that directory. **390 slides
must not be reported as 390 patients.** No real patient split, within-patient
label audit, acquisition run or real performance result was produced.

## Scope and limits

The implementation and bounded validation are complete. Clinical efficacy,
real-patient AUC, multi-seed significance, synthesis realism, GPU determinism and
large-bag GPU memory limits are unverified. Real experiments require the true
patient mapping, review of cohort labels, and scheduled compute. Complete 3-seed
GPU experiments are provided as launchable configuration generation, not run.

Synthetic coverage is a proxy with a configurable kernel and discount. Depending
on those parameters and the data, real coverage may dominate all synthetic
references; a benefit over the real-only control is not assumed or established.
Cold MIL initialization defaults to a fixed random state on supplied pretrained
patch features; optional external MIL weights require explicit provenance.

## Follow-up: device-enabled execution and reproducibility

The default command sandbox hid NVIDIA character devices; device-enabled execution
confirmed working RTX 3090 GPUs, driver 555.42.06 and the existing CUDA 12.4 RRT
runtime. CUDA availability in the sandbox must not be interpreted as a host-driver
diagnosis. No driver or package installation was needed.

The final suite passes **24 tests**. Added tests cover safe SHA256 reuse including
rapid writes with coalesced timestamps, and repeatable dictionaries under a
parallel outer runtime. K-means accumulation is now single-threaded. CUDA setup
selects the configured device and supplies a default deterministic cuBLAS workspace
before initialization; reset/peak memory accounting explicitly uses that device.

A separate synthetic-only CUDA test ran 3 rounds and 111 updates on `cuda:1`,
starting without an exported `CUBLAS_WORKSPACE_CONFIG`. A continuous run and a run
paused at update 2 then resumed produced exactly equal model parameters, query
audits and metrics. All round peaks on the selected device were positive. These
checks establish execution and replay behavior, not convergence or clinical value.
Private cohort artifacts and results are kept outside version control.

## Full-label protocol checks

27 tests pass after adding optional PT memory mapping, validation-AUC early
stopping and validation-only classification threshold selection. Tests verify
mapped tensors retain all values, AUC is invariant to decision threshold, and
stopping counters plus epoch-loss accumulation survive optimizer-step resume.
Flipping held-out test labels leaves the trained model, chosen epoch and
validation-derived threshold unchanged.

A synthetic-only `cuda:1` check revealed all 24 training labels, paused at update
2 and resumed through validation early stopping. The selected threshold came
from validation; fixed-0.5 and selected-threshold AUCs matched, dictionary fitting
used training patients only, and the configured-device peak memory was positive.
No private test feature was used for this runtime check.
