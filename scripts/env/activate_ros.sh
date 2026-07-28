#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  printf 'Usage: source scripts/env/activate_ros.sh\n' >&2
  exit 2
fi

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
teleop_activate_env ros-py312 no

teleop_source_ros_jazzy
if [[ -f "${TELEOP_ROOT}/install/setup.bash" ]]; then
  restore_nounset=no
  if [[ "$-" == *u* ]]; then
    restore_nounset=yes
    set +u
  fi
  # shellcheck disable=SC1091
  source "${TELEOP_ROOT}/install/setup.bash"
  if [[ "${restore_nounset}" == "yes" ]]; then
    set -u
  fi
  unset restore_nounset
fi
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=1
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export CYCLONEDDS_URI="file://${TELEOP_ROOT}/scripts/env/cyclonedds-local.xml"
