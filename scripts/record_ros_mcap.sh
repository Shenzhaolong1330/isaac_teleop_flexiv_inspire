#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_ID="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT="${ROOT}/sessions/${SESSION_ID}/ros2"

source /opt/ros/jazzy/setup.bash
mkdir -p "$(dirname -- "${OUTPUT}")"
if [[ -e "${OUTPUT}" ]]; then
  echo "Refusing to overwrite existing rosbag output: ${OUTPUT}" >&2
  exit 2
fi

exec ros2 bag record \
  --storage mcap \
  --output "${OUTPUT}" \
  --topics \
  /xr_teleop/ee_poses \
  /xr_teleop/controller_data \
  /xr_teleop/hand \
  /xr_teleop/finger_joints \
  /tf \
  /tf_static \
  /isaac_flexiv/status \
  /isaac_flexiv/gateway_status \
  /isaac_flexiv/diagnostics \
  /isaac_flexiv/requested/left_delta \
  /isaac_flexiv/requested/right_delta \
  /isaac_flexiv/applied/left_delta \
  /isaac_flexiv/applied/right_delta \
  /flexiv/left/joint_states \
  /flexiv/right/joint_states \
  /flexiv/left/tcp_pose \
  /flexiv/right/tcp_pose \
  /inspire/left/actuator_states \
  /inspire/right/actuator_states

