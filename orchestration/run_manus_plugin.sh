#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLUGIN="$PROJECT_ROOT/third_party/IsaacTeleop/install/manus-isaac/plugins/manus/manus_hand_plugin"
ERGONOMICS_UDP="127.0.0.1:15053"

while (( $# > 0 )); do
  case "$1" in
    --ergonomics-udp)
      if (( $# < 2 )); then
        echo "--ergonomics-udp requires HOST:PORT" >&2
        exit 2
      fi
      ERGONOMICS_UDP="$2"
      shift 2
      ;;
    *)
      echo "unknown MANUS plugin argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ ! "$ERGONOMICS_UDP" =~ ^(127\.0\.0\.1|localhost):[0-9]{1,5}$ ]]; then
  echo "MANUS Ergonomics destination must be loopback HOST:PORT" >&2
  exit 2
fi

if [[ ! -x "$PLUGIN" ]]; then
  echo "MANUS plugin is missing or not executable: $PLUGIN" >&2
  exit 2
fi

# Core Integrated can occasionally leave the MetaglovePro Sensor Dongle
# enumerated by Linux while failing to discover it (Prime1 remains visible but
# both gloves disappear).  A device-scoped USB reset restored 318DCDA6 during
# the site diagnosis.  Record owns this fresh MANUS process, so reset only the
# exact Metaglove vendor/product before Core opens it; never touch Prime1,
# Quest, cameras, or unrelated USB devices.
if command -v usbreset >/dev/null 2>&1 && lsusb -d 3325:0049 >/dev/null 2>&1; then
  if usbreset 3325:0049; then
    echo "MANUS Sensor Dongle 3325:0049 reset before Core startup"
  else
    echo "warning: MANUS Sensor Dongle reset failed; Core discovery will decide readiness" >&2
  fi
fi

# shellcheck disable=SC1091
source "$PROJECT_ROOT/scripts/env/activate_isaac.sh"
# This station uses MANUS only for finger articulation. Quest Touch controllers
# own wrist position/orientation through xr_raw_ros_source. Skipping the plugin's
# OpenXR session makes Ergonomics independent of CloudXR video/runtime startup.
export ISAAC_TELEOP_MANUS_WRIST_SOURCE=controllers
export ISAAC_TELEOP_MANUS_OPENXR_ENABLED=0
export ISAAC_TELEOP_MANUS_ERGONOMICS_UDP="$ERGONOMICS_UDP"
exec "$PLUGIN"
