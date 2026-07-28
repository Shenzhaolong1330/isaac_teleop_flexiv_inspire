#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  printf 'Usage: source scripts/env/activate_rdk.sh\n' >&2
  exit 2
fi

# The RDK daemon intentionally runs without ROS, Isaac or LeRobot paths.
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
teleop_activate_env rdk-py310 yes
