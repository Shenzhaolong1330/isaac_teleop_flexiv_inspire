#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CXR_RUN="${NV_CXR_RUNTIME_DIR:-$HOME/.cloudxr/run}"
ENV_FILE="$CXR_RUN/cloudxr.env"
PID_FILE="$CXR_RUN/cloudxr.pid"
PLUGIN="$PROJECT_ROOT/third_party/IsaacTeleop/install/manus-isaac/plugins/manus/manus_hand_plugin"

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

deadline=$((SECONDS + 60))
while (( SECONDS < deadline )); do
  if [[ -f "$ENV_FILE" && -f "$PID_FILE" ]]; then
    runtime_pid="$(tr -dc '0-9' < "$PID_FILE")"
    if [[ -n "$runtime_pid" ]] && kill -0 "$runtime_pid" 2>/dev/null; then
      break
    fi
  fi
  sleep 0.2
done

if [[ ! -f "$ENV_FILE" || ! -f "$PID_FILE" ]]; then
  echo "CloudXR runtime did not become ready within 60 seconds." >&2
  exit 2
fi
runtime_pid="$(tr -dc '0-9' < "$PID_FILE")"
if [[ -z "$runtime_pid" ]] || ! kill -0 "$runtime_pid" 2>/dev/null; then
  echo "CloudXR runtime process is not alive after 60 seconds." >&2
  exit 2
fi

# shellcheck disable=SC1091
source "$PROJECT_ROOT/scripts/env/activate_isaac.sh"
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a
# This station uses MANUS only for finger articulation. Quest Touch controllers
# own wrist position/orientation, so optical Quest hand tracking must never
# steal the MANUS wrist root when it appears briefly and then disappears after
# the operator picks up the controllers.
export ISAAC_TELEOP_MANUS_WRIST_SOURCE=controllers
exec "$PLUGIN"
