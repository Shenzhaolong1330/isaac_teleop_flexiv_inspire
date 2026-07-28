#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-${ROOT}/configs/flexiv_inspire.yaml}"
shift || true

export PYTHONPATH="${ROOT}/src:/home/hb/flexiv_inspire_ws/src/dual_arm_teleop${PYTHONPATH:+:${PYTHONPATH}}"

# This module refuses to connect unless the new YAML has command_enabled: true,
# --enable-hardware-command is supplied, and the exact area-clear token is given.
exec conda run --no-capture-output -n flexiv_teleop \
  python -m flexiv_inspire_isaac.gateway_safe_runtime \
  --config "${CONFIG}" "$@"

