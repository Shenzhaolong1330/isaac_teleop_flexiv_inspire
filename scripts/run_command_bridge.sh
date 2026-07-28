#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-${ROOT}/configs/flexiv_inspire.yaml}"

source /opt/ros/jazzy/setup.bash
if [[ -f /home/hb/flexiv_inspire_ws/install/setup.bash ]]; then
  source /home/hb/flexiv_inspire_ws/install/setup.bash
fi

exec "${ROOT}/.venv/bin/python" \
  -m flexiv_inspire_isaac.ros2_runtime \
  --config "${CONFIG}" \
  --enable-command

