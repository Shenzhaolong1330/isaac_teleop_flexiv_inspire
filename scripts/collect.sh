#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 0 ]]; then
  echo "collect.sh takes no arguments; edit config/recording.yaml" >&2
  exit 2
fi

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/scripts/env/activate_ros.sh"
if [[ -f "$PROJECT_ROOT/ros2_ws/install/setup.bash" ]]; then
  source "$PROJECT_ROOT/ros2_ws/install/setup.bash"
fi
exec "$PROJECT_ROOT/envs/ros-py312/bin/python" -m flexiv_inspire_isaac.cli collect
