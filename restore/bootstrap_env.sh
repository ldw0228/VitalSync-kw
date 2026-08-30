#!/usr/bin/env bash
set -euo pipefail

SNN_RESTORE_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SNN_RESTORE_ROOT="$(cd -- "${SNN_RESTORE_SCRIPT_DIR}/.." && pwd)"
SNN_UV_CACHE_DIR="${TMPDIR:-/tmp}/snnproject-uv-cache"

cd "${SNN_RESTORE_ROOT}"
mkdir -p "${SNN_UV_CACHE_DIR}"
export UV_CACHE_DIR="${SNN_UV_CACHE_DIR}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install uv from its official distribution, then rerun." >&2
  exit 2
fi

if [[ "${1:-portable}" == "exact-linux-cu130" ]]; then
  uv venv --python 3.12 .venv
  uv pip install --python .venv/bin/python \
    -r restore/requirements-linux-cu130.txt
  uv pip install --python .venv/bin/python --no-deps -e .
else
  uv venv --python 3.12 .venv
  uv pip install --python .venv/bin/python -e '.[dev]'
  uv pip install --python .venv/bin/python xgboost==3.4.1
fi

.venv/bin/python -c 'import numpy, pandas, scipy, sklearn, snntorch, torch, snn_rr; print("environment imports: OK"); print("torch", torch.__version__, "cuda", torch.cuda.is_available())'
