# AGENTS.md — SnnProject working contract

This file is the first instruction source for any AI or human agent working in
this repository. It describes the scientific claim boundary, restoration
procedure, data-safety rules, and the current execution boundary as of
2026-08-31. Read `RESTORE_GUIDE.md` next.

## 1. Project status: do not overclaim

This repository estimates respiratory rate from three XeThru UWB radars using
hybrid/spiking neural networks. It is a retrospective research system, not a
commercial or medical product.

The best completed full-coverage identity-disjoint OOF result currently is:

- MAE 1.291 bpm
- identity-macro MAE 1.220 bpm
- RMSE 2.410 bpm
- within ±2 bpm 80.79%
- error greater than 5 bpm 6.23%
- 25–35 bpm MAE 4.216 bpm

All six declared commercial accuracy gates fail. HCES v2 also fails all fixed
seed gates. DHFER v3r1 has only a retrospective discovery-validation result,
not a completed full OOF result. Candidate-oracle values are diagnostics that
use the target and must never be described as deployable performance.

Commercial claims remain prohibited until all fixed seeds pass the internal
gates and an independent prospective cohort, independent reference, target
device, fault campaign, calibration, shadow/canary, and rollback validation
also pass.

Current authority order:

- `artifacts/COMMERCIAL_SNN_GOAL_V3_2026-08-31.md`
- `artifacts/SNN_PROJECT_DEVELOPMENT_PROGRESS_2026-08-31.md`
- `artifacts/SNN_PROJECT_TECHNICAL_STATUS_REPORT_2026-08-30.md`
- `README.md`
- `REPORT.md`

Goal v3 and the 2026-08-31 progress report govern the current source
generation. Goal v2, earlier execution plans/progress reports, existing release
manifests, and `artifacts/commercial_goal_report.json` are immutable historical
evidence only. They do not authorize current training, evaluation, or release.
If an older document conflicts with Goal v3 or the current progress report,
follow the two current documents and preserve the conflict for audit.

## 2. Non-negotiable scientific invariants

1. Split by physical identity, never by session or overlapping window.
2. Treat all adaptive work on the 18-person cohort as retrospective.
3. Never fit a scaler, proposer, router, threshold, calibration, ensemble
   weight, or checkpoint decision using the outer-test target.
4. Reference/BIOPAC values are allowed only in label construction, supervised
   training on the permitted split, or the final sealed join/evaluation stage.
5. `radar_observable` is target-dependent and is forbidden as an inference
   feature.
6. Missing radar/branch/candidate cells must remain structurally masked and
   exact zero after scaling. Do not infer availability from a numeric zero.
7. Preserve all three fixed seeds independently. Never average away a failed
   seed or choose a different architecture/release mode per seed or fold.
8. Never present an oracle, post-hoc threshold, selective subset, or
   retrospective best run as production performance.
9. Do not fabricate, hand-edit, or bypass provenance/authorization receipts.
10. Do not delete failed runs or GPU-usage evidence. Failure is part of the
    scientific record.

## 3. Data boundary

The raw dataset is private physiological data. Keep it out of public
repositories, public Drive folders, logs, prompts, issue trackers, and model
artifacts intended for broad sharing.

Expected restored layout:

```text
SnnProject/
├── HAI_EXPERIMENT/                 # private, 30 session folders
├── configs/
├── src/snn_rr/
├── scripts/
├── tests/
├── artifacts/
└── restore/
```

Known data facts that affect code:

- 30 source session folders; 29 usable three-radar+BIOPAC sessions
- 18 physical identities; repeated sessions must remain in the same fold
- `S24_KHJ` has empty radar streams and is excluded
- `S17_RJS` maps to physical identity PJS
- `S07_KDM` has a timestamp counter reset handled deterministically
- `S22_KJH` has one radar-2 outlier repaired from past-only samples
- no common radar/BIOPAC hardware trigger; retain this limitation in reports

Do not rename or rewrite raw files. Parsers must treat raw input as read-only.

Current acquisition-v2 evidence:

- frozen authority: 30 sessions, 29 usable, 18 usable physical identities
- full causal-bound reconstruction: complete but diagnostic
- synchronization authorized: 0/29
- measured-timing eligible: 19/29
- stage-metric eligible: 0/29
- strict-cache/scientific eligible: 0/29
- diagnostic RF cache: 5,826 windows from 18 mapping-bearing sessions,
  `reference_valid=0` for every row

The authoritative paths and content hashes are recorded in
`artifacts/SNN_PROJECT_DEVELOPMENT_PROGRESS_2026-08-31.md`. Do not train from
that diagnostic cache or convert its proposed mappings into approvals.

## 4. First actions after restoration

From the download directory, verify and extract the mode-preserving core
archive, then enter the project root:

```bash
sha256sum -c SnnProject_RESTORE_SHA256SUMS_2026-08-30.txt
tar --zstd -xf SnnProject_repro_core_2026-08-30.tar.zst
# Overlay the exact V8R4 continuation state, including sealed split packs.
tar --zstd -xf SnnProject_v8r4_state_2026-08-30.tar.zst
cd SnnProject

# Extract both non-overlapping raw archive parts into the project root.
unzip -n ../HAI_EXPERIMENT-20260827T035530Z-1-001.zip -d .
unzip -n ../HAI_EXPERIMENT-20260827T035530Z-1-002.zip -d .

# Build the Python environment.
bash restore/bootstrap_env.sh

# Validate repository shape, data count, compilation, and tests.
bash restore/verify_restore.sh
```

For an exact Linux/CUDA audit attempt, use:

```bash
bash restore/bootstrap_env.sh exact-linux-cu130
```

The exact lock is platform-specific and wheel availability may differ. A
successful import/test run is required even if dependency installation exits
successfully.

## 5. Rebuildable caches

The compact restore backup intentionally excludes `artifacts/cache` because it
is approximately 15 GB and deterministic/rebuildable from the raw archives.
The 18 strict HCES preprocessing stacks needed to rebuild the present harmonic
caches without retraining 90 nested proposers are preserved in the core
archive. The V8R4 split packs are preserved separately in the V8R4 state
archive and must not be substituted for the general cache tree.

Canonical cache:

```bash
.venv/bin/python scripts/audit_dataset.py HAI_EXPERIMENT \
  --format json --output artifacts/dataset_audit.json
.venv/bin/python scripts/build_features.py \
  --config configs/default.yaml --force
```

SVD cache used by the harmonic pipeline:

```bash
.venv/bin/python scripts/build_svd_features.py \
  --dataset-root HAI_EXPERIMENT \
  --canonical-cache artifacts/cache/rf32s \
  --output-dir artifacts/cache/svd_components_all_v1 \
  --all-windows --components 12 --nfft 4096 --n-iter 2 --force
```

Before relying on these commands, compare `--help` and the frozen campaign
contracts. Cache paths and hashes are part of downstream provenance. Never
silently reuse a cache whose schema/config/source hash differs.

## 6. Baseline and current model families

### Structured TriRadarRRSNN

The reproducible leader path is documented in `README.md`. It uses a shared
2-D radar encoder, range attention, radar reliability, frequency-preserving
PLIF/LIF blocks, 12 internal simulation steps, and a validation-locked
two-component ensemble.

### HCES v2

HCES represents up to 12 RR candidates as graph nodes with harmonic
relationships. Locked full OOF exists for seeds 20260828/29/30 and is preserved
in the restore bundle. The locked policy fell back to no action; it did not
meet the commercial gates. Do not rerun outer-test selection to search for a
better policy.

### DHFER v3r1 / V8R4

DHFER uses the frozen 571-wide candidate layout, seven directed harmonic
relations, PLIF graph blocks, a causal PLIF→ALIF factor router, and hard
anchor/candidate experts. The latest implementation identifies a coordinate-
evidence pooling defect and a hard-routing risk-gradient mismatch.

Current execution boundary:

- ROOTBIND1 preserved a real fail-closed pretrain-context failure.
- CONTEXT1 source fixes and tests exist.
- `IMPLEMENTATION_TEST_RECEIPT_V8R4A_CONTEXT1.json` is absent.
- `V3R1_SOURCE_SNAPSHOT_V8R4A_CONTEXT1.json` is absent.
- `PRETRAIN_AUTHORIZATION_V8R4A_CONTEXT1.json` is absent.
- V8R4 efficiency, 18-unit discovery, and 18-unit promotion are not complete.
- V8R5 is a proposal, not measured evidence.

Do not manually create those three CONTEXT1 artifacts or start production
training around their fail-closed guards. The current CONTEXT1 validator is a
terminal fail-closed generation: no independently governed external test
issuer/runner trust root or signature verifier is implemented. Local JSON,
self-hash, self-signature, monkeypatching constants, or a real-bubblewrap pass
cannot issue the trio. Continuation requires a new governed source generation
with an external trust anchor and verifier; it is not a command available in
this snapshot.

### AxisRiskRouterSNN V8R5

V8R5 is a separately versioned, unmeasured successor proposal using the frozen
571-wide layout. It is not ancestry-independent: it explicitly reuses the
governed feature-layout contract and `EpisodeSpikingCell`, with both source
dependencies hash-bound in its receipt. Its 228,838-parameter source joins
every evidence value and per-feature availability bit with
radar/ratio/branch/candidate coordinates before pooling, uses bidirectional
axial attention, seven directed harmonic relations, two PLIF graph blocks, a
causal PLIF→ALIF state, disjoint value/route/risk heads, explicit classical-RR
availability, and soft expected-risk training with inference-only hard
routing. A concrete format-v2 cache validator is implemented but always
returns `training_authorized: false`. Synthetic correctness tests do not
authorize training or prove accuracy. The config remains
`training_authorized: false` and `commercial_claim_allowed: false`.

The checkpoint contract stores canonical tensor-only source/config/layout/
dependency and runtime-structure receipts, rejects partial/assigning/nonfinite
loads before mutation, and fail-closes nonfinite learned experts or temporal
state to a finite classical fallback or exact-zero unavailable output. These
guards still do not prove the bytes actually compiled before import; protected
execution requires the externally governed isolated-source launcher described
in Goal v3.

The current source also closes diagnostic-cache training entry points,
revalidates imported V3R1 train/predict entries independently, preserves
timing masks and cache provenance in custom prediction, and prevents invalid
resampling intervals from becoming synchronization motion markers. These are
scientific-integrity improvements, not new performance evidence.

## 7. Tests and verification

Minimum checks after source changes:

```bash
.venv/bin/python -m py_compile scripts/validate_hfr_v3r1_authorization.py
.venv/bin/python -m pytest -q
```

The backup-time snapshot remains recorded in
`restore/backup_validation_2026-08-30.json`: 688 passed, 4 skipped, and 2
order/precision-sensitive failures. The current source removes import-time
Torch thread mutations and uses fixed-order accumulation for the affected SVD
projection. An earlier intermediate 2026-08-31 generation collected 1,545
tests, passed 1,541 twice, skipped four real-bubblewrap tests in the managed
namespace, and then passed those four tests in a capable host context. The
final 2026-08-31 source generation collects 1,766 tests and passed two
consecutive full-suite runs with 1,762 passed and the same four managed-host
bubblewrap skips; those four tests then passed 4/4 in a capable host context.
The active V8R4 fixed collection is exactly 739 node IDs with semantic SHA-256
`b9b192c084d3f6d69094657bceb9047e368c12cac4e4420c2f75ef3c7fc39df4`.
Do not rewrite the backup snapshot. A passing unit suite still does not
authorize a new scientific run.

For changes touching split, target firewall, sealed packs, authorization,
ledger, or sandbox code, run the corresponding focused tests first and then
the full suite. A passing unit suite does not authorize a new scientific run.

## 8. Safe working rules for agents

- Inspect before editing; prefer `rg`/`rg --files` for discovery.
- Preserve unrelated user changes and immutable historical artifacts.
- Never run destructive cleanup against `HAI_EXPERIMENT`, `artifacts`, or the
  project root.
- Write new adaptive experiments to a new versioned contract/cache/output
  root. Do not overwrite locked evidence.
- Record commands, source/config/data hashes, seeds, split identities, device,
  and termination state for every material run.
- Reset temporal state at physical session boundaries and never use future
  windows in inference.
- Treat `artifacts/cache` and sealed packs as derived data, not source truth.
- If the exact environment cannot be recreated, document the deviation before
  producing new metrics.
- If a required authorization or prospective dataset is missing, stop at that
  boundary and preserve a resume-ready state; do not weaken the guard.

## 9. What the compact backup does and does not preserve

It preserves:

- source, configs, tests, documentation, environment locks, and restore tools
- both original raw-data ZIP parts
- current leader checkpoints and OOF/ensemble evidence
- locked HCES OOF, radar-mask, uncertainty/release, and streaming evidence
- all 18 strict HCES preprocessing stacks used for harmonic-cache rebuilding
- the measured DHFER H0 discovery artifact and governance/provenance history
- the complete current V8R4 state, including six physically outer-removed
  split packs, ledgers, lifecycle evidence, and sealed inputs, in a separate
  mode-preserving archive

It excludes:

- `.venv` (recreated from locks)
- extracted `HAI_EXPERIMENT` (recreated from the two raw ZIP parts)
- the 15 GB derived cache tree
- redundant development/smoke/failed-model tensor payloads

The backup supports source recovery, result inspection, immediate historical
inference with selected checkpoints, legacy-result reproduction after cache
regeneration, and forensic recovery of the exact V8R4 continuation state.
That restoration/retraining capability is not current scientific authorization;
new attempts must use a new versioned root and obey the current acquisition and
training guards. Existing V8R4 receipts
bind absolute paths, interpreter hashes, source bytes, and file modes. Restore
to `/home/hwiseong/Documents/SnnProject` when exact continuation is required;
at any other path, retain the old state as historical evidence. The present
terminal fail-closed validator cannot issue a replacement context; a new
externally governed trust-root/verifier generation is required. Never edit an
old receipt. The backup does not make CUDA execution bitwise deterministic
across different hardware.
