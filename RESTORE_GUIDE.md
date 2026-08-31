# SnnProject compact restore and reproducibility guide

This guide restores the 2026-08-30 project snapshot on another Linux
workstation. The compact backup keeps the two original raw-data archives,
actual source code, configuration, tests, selected checkpoints, locked result
evidence, strict HCES preprocessing inputs, the exact V8R4 continuation state,
environment locks, and provenance. It omits the extracted duplicate dataset,
virtual environment, general deterministic caches, and redundant intermediate
experiments.

Read `AGENTS.md` before modifying the restored repository. For current work,
`artifacts/COMMERCIAL_SNN_GOAL_V4_CONTINUATION_2026-08-31.md`,
`artifacts/COMMERCIAL_SNN_GOAL_V3_2026-08-31.md`, and
`artifacts/SNN_PROJECT_DEVELOPMENT_PROGRESS_2026-08-31.md` have authority over
Goal v2, older execution plans/progress reports, preserved release manifests,
and `artifacts/commercial_goal_report.json`. Those older objects are historical
evidence and cannot authorize a current run.

## 1. Backup set

Download every file named in `SnnProject_RESTORE_SHA256SUMS_2026-08-30.txt` into
one directory. The essential set is:

```text
SnnProject_repro_core_2026-08-30.tar.zst
SnnProject_v8r4_state_2026-08-30.tar.zst
HAI_EXPERIMENT-20260827T035530Z-1-001.zip
HAI_EXPERIMENT-20260827T035530Z-1-002.zip
SnnProject_RESTORE_SHA256SUMS_2026-08-30.txt
SnnProject_RESTORE_INDEX_2026-08-30.md
```

The two HAI archives are non-overlapping parts of one `HAI_EXPERIMENT/` tree.
Both are required. Do not concatenate them and do not keep only the numerically
larger part.

Recommended free disk space:

- download plus source restore only: at least 8 GB
- raw extraction plus canonical/SVD/harmonic cache rebuild: at least 30 GB
- retraining with new checkpoints and temporary packs: at least 45 GB

## 2. Verify before extraction

Run from the directory containing the downloaded files:

```bash
sha256sum -c SnnProject_RESTORE_SHA256SUMS_2026-08-30.txt
tar --zstd -tf SnnProject_repro_core_2026-08-30.tar.zst >/dev/null
tar --zstd -tf SnnProject_v8r4_state_2026-08-30.tar.zst >/dev/null
unzip -t HAI_EXPERIMENT-20260827T035530Z-1-001.zip
unzip -t HAI_EXPERIMENT-20260827T035530Z-1-002.zip
```

Stop if any checksum, CRC, or archive test fails. Redownload the failed object;
do not try to repair model or physiological data bytes manually.

## 3. Extract the project and raw data

```bash
tar --zstd -xf SnnProject_repro_core_2026-08-30.tar.zst
tar --zstd -xf SnnProject_v8r4_state_2026-08-30.tar.zst
unzip -n HAI_EXPERIMENT-20260827T035530Z-1-001.zip -d SnnProject
unzip -n HAI_EXPERIMENT-20260827T035530Z-1-002.zip -d SnnProject
cd SnnProject
```

Expected raw layout after both parts are extracted:

```text
HAI_EXPERIMENT/
├── S01_CMS/
├── ...
└── S30_SJE/
```

There must be 30 immediate session directories. One session (`S24_KHJ`) has
empty radar streams and is intentionally excluded by the audited parser.

Quick structural check:

```bash
find HAI_EXPERIMENT -mindepth 1 -maxdepth 1 -type d | wc -l
```

Expected output: `30`.

## 4. Recreate the Python environment

The recorded environment used Python 3.12.13, PyTorch 2.13.0+cu130, CUDA build
13.0, cuDNN 9.20, and an RTX 4070. Python must satisfy `>=3.12,<3.13`.

The recommended portable installation uses `uv`:

```bash
bash restore/bootstrap_env.sh
```

This creates `.venv`, installs `.[dev]`, adds the observed optional XGBoost
dependency, and imports the core libraries.

For a best-effort exact Linux/CUDA distribution replay:

```bash
bash restore/bootstrap_env.sh exact-linux-cu130
```

The exact file is an audit snapshot. CUDA/PyTorch wheel sources and driver
compatibility are host-specific. If the exact install is unavailable, install
a compatible official PyTorch build, retain every deviation in a new
environment receipt, and rerun all tests and latency measurements. Never reuse
the historical CUDA latency claim for different hardware.

Environment evidence:

- `restore/environment_snapshot.json`
- `restore/requirements-top-level.txt`
- `restore/requirements-linux-cu130.txt`
- `pyproject.toml`

Check the restored device:

```bash
.venv/bin/python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")'
```

## 5. Validate source and selected artifacts

```bash
bash restore/verify_restore.sh
```

The script checks repository shape, the 30-session directory count, Python
compilation, and the full pytest suite when `.venv` exists.

Historical fixed evidence was 694 tests collected, 690 passed, and 4 skipped.
The backup-time audit is stricter: two consecutive full-suite runs ended with
688 passed, 4 skipped, and 2 failed, although both failed tests passed in an
isolated rerun. Read `restore/backup_validation_2026-08-30.json` for the exact
node IDs and observed precision failure. This is an unresolved test-order,
thread, or floating-point-state issue, not a clean test pass. Diagnose it
before issuing a new scientific authorization; do not merely loosen the
tolerances. The four skips were real-bubblewrap tests blocked by the original
sandbox's unprivileged-namespace policy. A capable workstation must execute
them before claiming execution closure.

Inspect the scientific status before running anything expensive:

```bash
sed -n '1,220p' artifacts/SNN_PROJECT_TECHNICAL_STATUS_REPORT_2026-08-30.md
```

## 6. Audit raw data and rebuild derived caches

The commands in this section reconstruct the historical cache lineage for
inspection and legacy-result reproduction. They do not convert the current
acquisition-v3 diagnostic state (`synchronization authorized: 0/29`) into new
scientific authority. Use new versioned output roots for experiments and obey
the current strict acquisition guards.

### 6.1 Dataset audit and canonical RF cache

```bash
.venv/bin/python scripts/audit_dataset.py HAI_EXPERIMENT \
  --format json --output artifacts/dataset_audit.json

.venv/bin/python scripts/build_features.py \
  --config configs/default.yaml --force
```

Expected historical canonical cache contract:

- 29 usable sessions
- 9,576 candidate windows
- 2,327 reference-valid windows
- `maps.npy` shape `[N,3,73,182]`, float16
- `aux.npy` shape `[N,1205]`, float32
- physical-identity split count 18

Compare the new dataset audit with the preserved report. A mismatch is a data
or parser-version event and must be investigated before training.

### 6.2 SVD cache

```bash
.venv/bin/python scripts/build_svd_features.py \
  --dataset-root HAI_EXPERIMENT \
  --canonical-cache artifacts/cache/rf32s \
  --output-dir artifacts/cache/svd_components_all_v1 \
  --all-windows --components 12 --nfft 4096 --n-iter 2 --force
```

SVD extraction must remain label-free. BIOPAC/reference values may be copied as
metadata for later evaluation but must not enter source separation.

### 6.3 Harmonic candidate caches

The harmonic cache is proposer-, fold-, seed-, and iteration-specific. Do not
create one global cache and silently substitute it for nested caches. Read:

- `configs/harmonic_set_v2.yaml`
- `artifacts/campaigns/harmonic_candidate_set_snn_v2/ADAPTIVE_CAMPAIGN_CONTRACT.json`
- `scripts/build_harmonic_set_cache.py --help`

An explicit low-level build has the following form, but its proposer and fold
assignments must be the exact authorized nested artifacts:

```bash
.venv/bin/python scripts/build_harmonic_set_cache.py \
  --rf-cache artifacts/cache/rf32s \
  --svd-cache artifacts/cache/svd_components_all_v1 \
  --proposer PATH_TO_AUTHORIZED_PROPOSER \
  --fold-assignments artifacts/runs/final_alias_gate_s12_deterministic/fold_assignments.json \
  --output-dir PATH_TO_NEW_VERSIONED_CACHE \
  --merge-radius-bpm 0.5 \
  --proposal-selection posterior-nms \
  --posterior-nms-suppression-bpm 1.25 \
  --base-proposals expected-map \
  --svd-components 12 \
  --proposer-features
```

Never replace `PATH_TO_AUTHORIZED_PROPOSER` with a model fitted on the outer
identity being predicted.

### 6.4 Rebuild the current V3 diagnostic lineage

Use fresh, absent output names. These commands reconstruct byte/timing evidence
and label-free RF/SVD inputs; they do not authorize training.

```bash
.venv/bin/python scripts/reconstruct_acquisition.py \
  --schema-version v3 \
  --output-dir artifacts/acquisition/RESTORED_V3_DIAGNOSTIC \
  --skip-range-tracks --skip-review-plots

.venv/bin/python scripts/build_features.py \
  --config configs/default.yaml --dataset-root HAI_EXPERIMENT \
  --cache-dir artifacts/cache/RESTORED_RF_V3_DIAGNOSTIC \
  --acquisition-manifest \
    artifacts/acquisition/RESTORED_V3_DIAGNOSTIC/manifest.json \
  --acquisition-mode diagnostic

.venv/bin/python scripts/build_svd_features.py \
  --dataset-root HAI_EXPERIMENT \
  --canonical-cache artifacts/cache/RESTORED_RF_V3_DIAGNOSTIC \
  --output-dir artifacts/cache/RESTORED_SVD_V3_DIAGNOSTIC \
  --all-windows --components 12 --nfft 4096 --n-iter 2 --workers 4
```

Expected current diagnostic shape: 30 source sessions, 29 usable sessions, 18
physical identities, 9,575 RF/SVD rows, 18 mapping-bearing sessions, 11
radar-only unmapped sessions, and zero reference-valid rows. Any mismatch is a
new source/config/parser event; preserve the failed attempt and investigate it.

## 7. Reproduce the established structured-SNN leader

This is a historical legacy reproduction, not a new scientific campaign. It
cannot establish new performance while current acquisition authority is 0/29.
The complete commands are maintained in `README.md`. Every attempt must use a
unique versioned root; never write to the preserved leader paths. The core
sequence is:

```bash
REPRO_ROOT=artifacts/reproductions/legacy_leader_20260831_attempt_001
mkdir -p artifacts/reproductions
mkdir "$REPRO_ROOT"  # fail if this attempt root already exists

.venv/bin/python scripts/build_features.py \
  --config configs/default.yaml --cache-dir "$REPRO_ROOT/cache/rf32s" --force

.venv/bin/python scripts/train.py \
  --config configs/default.yaml --model both --fold all \
  --preset default --simulation-steps 12 --device cuda --amp \
  --aux-fusion structured --cache-dir "$REPRO_ROOT/cache/rf32s" \
  --output-dir "$REPRO_ROOT/structured_aux"

.venv/bin/python scripts/train.py \
  --config configs/default.yaml --model snn --fold all \
  --preset default --simulation-steps 12 --device cuda --amp \
  --deterministic --aux-fusion structured --exact-aux-alignment \
  --cache-dir "$REPRO_ROOT/cache/rf32s" \
  --teacher-checkpoint "$REPRO_ROOT/structured_aux/fold_{fold}/teacher_best.pt" \
  --output-dir "$REPRO_ROOT/structured_exact"

.venv/bin/python scripts/ensemble.py \
  --run-a "$REPRO_ROOT/structured_aux" \
  --run-b "$REPRO_ROOT/structured_exact" \
  --cache-dir "$REPRO_ROOT/cache/rf32s" \
  --output-dir "$REPRO_ROOT/ensemble" \
  --device cuda --workers 4
```

The compact core already contains the selected historical checkpoints and
metrics at the preserved paths. `--resume` is permitted only within the same
new `$REPRO_ROOT` and only with its bound cache/checkpoints. Never resume into a
preserved path or another attempt's root.

Expected leader metrics are MAE 1.291, macro MAE 1.220, RMSE 2.410, within ±2
80.79%, error >5 6.23%, and high-RR MAE 4.216. These fail all six commercial
gates.

## 8. HCES and DHFER continuation boundary

The combined restore set preserves:

- locked HCES full OOF and proposer checkpoints
- HCES radar-mask, release, uncertainty, and streaming evidence
- DHFER H0 discovery-v7 best/last checkpoint and validation evidence
- campaign contracts, receipts, diagnostics, and V8R4A ledger history
- 18 strict HCES preprocessing stacks, so the present harmonic caches can be
  rebuilt without retraining all 90 nested proposer models

The separate V8R4 state archive, rather than the core archive, preserves the
DHFER run outputs and the entire
`artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/` tree, including the
six large physically outer-removed discovery packs. It is required to inspect
or continue from the exact current state; it is not a substitute for a valid
pretrain authorization.

HCES locked OOF is historical evidence. Do not reopen the test predictions to
select a new threshold or policy.

DHFER/V8R4 is not at a valid production-training launch point. The CONTEXT1
test receipt, source snapshot, and pretrain authorization are absent. This
source generation is terminal fail-closed: it has no independently governed
external test issuer/runner trust root or signature verifier. A real-bubblewrap
pass alone, local/self-hashed or self-signed JSON, monkeypatching constants, or
hand-writing the trio cannot issue authority. Continuation requires a new
governed source generation that implements and independently audits that trust
anchor and verifier.

V8R5 axis-preserving risk routing source, config, format-v2 cache validator,
and synthetic correctness tests are implemented. It remains an unmeasured,
training-unauthorized proposal; these implementation artifacts are not an
accuracy result or a launch authorization.

## 9. Selected artifact map

Important restored paths include:

```text
artifacts/runs/final_structured_aux_s12/
artifacts/runs/final_structured_exact_s12_deterministic/
artifacts/runs/ensemble_structured_exact/
artifacts/runs/harmonic_candidate_set_snn_v2/hcs_locked_oof/
artifacts/runs/harmonic_candidate_set_snn_v2/hcs_locked_radar_masks/
artifacts/runs/harmonic_candidate_set_snn_v2/hcs_streaming_deployment_v4/
artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/discovery_v7/
artifacts/campaigns/
artifacts/robustness/
artifacts/benchmarks/
artifacts/report/
```

The 15 GB general `artifacts/cache` tree is not in the compact backup. Its
absence is intentional. The 1.3 GB V8R4 split-input packs are present in the
separate V8R4 state archive and are verified by
`restore/V8R4_STATE_SHA256SUMS.txt`.

For exact V8R4 continuation, restore at
`/home/hwiseong/Documents/SnnProject`. Existing receipts bind the canonical
absolute root, Python executable/hash, source snapshot, modes, and ledger
bytes. At a different root or with a different interpreter, retain the files
only as historical evidence. The current validator cannot issue a replacement
context; that requires the new externally governed trust-root/verifier
generation described above. Never rewrite an old receipt to make it pass.

## 10. Troubleshooting

### `.venv/bin/python` exists but pip does not

The source environment was managed with `uv` and did not expose a pip module.
Use `uv pip ... --python .venv/bin/python`; do not treat `python -m pip` as a
required invariant.

### CUDA is unavailable

Confirm the NVIDIA driver with `nvidia-smi`, then verify that the installed
PyTorch build matches the supported driver/toolkit combination. CPU tests may
still run, but historical CUDA latency and numerical parity must be remeasured.

### Bubblewrap tests skip or fail

Check `bwrap --version`, unprivileged user namespaces, mount permissions, and
the exact target-sandbox tests. Do not convert a security failure into a skip
or remove the sandbox guard.

### Cache hash or schema differs

Stop. Confirm raw archive hashes, Python/Numpy/Scipy versions, config, source
snapshot, feature-name order, dtype, shape, and candidate proposer binding.
Create a new cache version only after recording the deviation.

### Metrics differ

Check physical-identity assignments, seed, CUDA determinism warnings, cache
hashes, exact checkpoint/config bindings, overlapping/non-overlapping window
selection, and whether the comparison is full OOF or only discovery validation.

## 11. Restore acceptance checklist

A restoration is usable when all of the following are true:

- every package-level SHA-256 passes
- both raw ZIP CRC tests pass
- 30 raw session directories exist
- dataset audit reports 29 usable sessions and 18 identities
- Python 3.12 imports the project and core dependencies
- full tests pass after the two backup-time order/precision-sensitive failures
  are diagnosed, with any bubblewrap skip explicitly explained
- selected checkpoint paths and preserved report files exist
- the V8R4 state checksum list passes after its separate archive is overlaid
- no preserved historical artifact was overwritten
- any environment/hardware deviation has its own receipt
- the restored agent acknowledges that current performance is not commercial

Only after this checklist should cache regeneration or historical reproduction
begin. New scientific training additionally requires the current acquisition,
nested-cache, split, and externally governed execution authorizations; restore
success alone grants none of them.
