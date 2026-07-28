#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  printf 'Usage: source scripts/env/activate_isaac.sh\n' >&2
  exit 2
fi

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
teleop_activate_env isaac-py312 no
teleop_select_cuda_12_8

# Only the system Jazzy installation is allowed; no external workspace overlay.
teleop_source_ros_jazzy
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=1
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export CYCLONEDDS_URI="file://${TELEOP_ROOT}/scripts/env/cyclonedds-local.xml"
