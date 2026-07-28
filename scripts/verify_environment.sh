#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-${ROOT}/configs/flexiv_inspire.yaml}"
PYTHON="${ROOT}/.venv/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing ${PYTHON}; run the documented uv venv/bootstrap step first." >&2
  exit 2
fi

exec "${PYTHON}" -m flexiv_inspire_isaac.verify --config "${CONFIG}"

