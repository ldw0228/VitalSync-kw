#!/usr/bin/env bash
set -euo pipefail

SNN_VERIFY_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SNN_VERIFY_ROOT="$(cd -- "${SNN_VERIFY_SCRIPT_DIR}/.." && pwd)"
cd "${SNN_VERIFY_ROOT}"

if [[ -f restore/CORE_SHA256SUMS.txt ]]; then
  sha256sum -c --quiet restore/CORE_SHA256SUMS.txt
  echo "core content SHA-256 verification: OK"
fi

SNN_V8R4_SENTINEL="artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/v8r4_split_inputs/discovery_shard_outer_3/units/outer_3_seed_20260828/discovery_cache/node_features.npy"
if [[ -f "${SNN_V8R4_SENTINEL}" ]]; then
  if [[ ! -f restore/V8R4_STATE_SHA256SUMS.txt ]]; then
    echo "V8R4 state is present but its checksum list is missing" >&2
    exit 2
  fi
  sha256sum -c --quiet restore/V8R4_STATE_SHA256SUMS.txt
  echo "V8R4 state SHA-256 verification: OK"
else
  echo "V8R4 state archive is not overlaid; exact continuation is unavailable" >&2
fi

for SNN_REQUIRED_PATH in \
  AGENTS.md \
  RESTORE_GUIDE.md \
  README.md \
  REPORT.md \
  pyproject.toml \
  configs/default.yaml \
  configs/harmonic_set_v2.yaml \
  configs/harmonic_factor_router_v3.yaml \
  src/snn_rr \
  scripts \
  tests; do
  if [[ ! -e "${SNN_REQUIRED_PATH}" ]]; then
    echo "missing required project path: ${SNN_REQUIRED_PATH}" >&2
    exit 3
  fi
done

if [[ -d HAI_EXPERIMENT ]]; then
  SNN_SESSION_COUNT="$(find HAI_EXPERIMENT -mindepth 1 -maxdepth 1 -type d | wc -l)"
  if [[ "${SNN_SESSION_COUNT}" -ne 30 ]]; then
    echo "expected 30 raw session directories, found ${SNN_SESSION_COUNT}" >&2
    exit 4
  fi
  echo "raw dataset directory count: OK (30)"
else
  echo "raw dataset is not extracted; follow RESTORE_GUIDE.md" >&2
fi

if [[ -x .venv/bin/python ]]; then
  .venv/bin/python -m py_compile scripts/validate_hfr_v3r1_authorization.py
  .venv/bin/python -m pytest -q
else
  echo "Python environment is not installed; run restore/bootstrap_env.sh" >&2
fi

echo "restore structure verification: OK"
