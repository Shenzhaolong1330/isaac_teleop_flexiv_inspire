#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
SESSION_ID="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT="${ROOT}/sessions/${SESSION_ID}/ros2"

source /opt/ros/jazzy/setup.bash
mkdir -p "$(dirname -- "${OUTPUT}")"
if [[ -e "${OUTPUT}" ]]; then
  printf 'Refusing to overwrite existing rosbag output: %s\n' "${OUTPUT}" >&2
  exit 2
fi

printf '%s\n' \
  'INFO: ROS-only diagnostic capture (native DDS CDR in rosbag2 MCAP).' \
  'INFO: For an episode manifest and validated F/T-zero provenance, use isaac-flexiv-episode.' >&2

exec ros2 bag record \
  --storage mcap \
  --output "${OUTPUT}" \
  --topics \
  /xr_teleop/ee_poses \
  /xr_teleop/controller_data \
  /xr_teleop/hand \
  /tf \
  /tf_static \
  /command_sources/teleop/command \
  /command_sources/teleop/heartbeat \
  /command_sources/policy/command \
  /command_sources/policy/heartbeat \
  /command_sources/replay/command \
  /command_sources/replay/heartbeat \
  /control/requested_command \
  /control/safe_command \
  /control/sent_command \
  /control/command_trace \
  /control/state \
  /control/stop \
  /robot/left_arm/state \
  /robot/left_arm/joint_states \
  /robot/left_arm/tcp_pose \
  /robot/left_arm/tcp_twist \
  /robot/left_arm/raw_ft \
  /robot/left_arm/tcp_wrench \
  /robot/right_arm/state \
  /robot/right_arm/joint_states \
  /robot/right_arm/tcp_pose \
  /robot/right_arm/tcp_twist \
  /robot/right_arm/raw_ft \
  /robot/right_arm/tcp_wrench \
  /robot/left_hand/state \
  /robot/left_hand/joint_states \
  /robot/left_hand/dynamic_joint_states \
  /robot/left_hand/tactile_raw \
  /robot/right_hand/state \
  /robot/right_hand/joint_states \
  /robot/right_hand/dynamic_joint_states \
  /robot/right_hand/tactile_raw \
  /camera/head/color/image_raw/compressed \
  /camera/head/color/frame \
  /camera/head/color/acquisition \
  /camera/left_wrist/color/image_raw/compressed \
  /camera/left_wrist/color/frame \
  /camera/left_wrist/color/acquisition \
  /camera/right_wrist/color/image_raw/compressed \
  /camera/right_wrist/color/frame \
  /camera/right_wrist/color/acquisition \
  /maintenance/zero_ft_sensors/_action/goal \
  /maintenance/zero_ft_sensors/_action/result \
  /maintenance/zero_ft_sensors/_action/feedback \
  /maintenance/zero_ft_sensors/_action/status \
  /maintenance/zero_ft_sensors/_action/cancel
