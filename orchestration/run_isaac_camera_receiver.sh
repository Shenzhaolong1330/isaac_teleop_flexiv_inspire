#!/usr/bin/env bash
set -euo pipefail
CONFIG_PATH="${1:?usage: run_isaac_camera_receiver.sh CONFIG [xr|monitor]}"
DISPLAY_MODE="${2:-xr}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CAMERA_ROOT="$PROJECT_ROOT/third_party/IsaacTeleop/examples/camera_streamer"
IMAGE_NAME="isaac-teleop-camera:latest"
CXR_ROOT="${CXR_HOST_VOLUME_PATH:-$HOME/.cloudxr}"
XR_JSON="${XR_RUNTIME_JSON:-$CXR_ROOT/openxr_cloudxr.json}"
CXR_RUN="${NV_CXR_RUNTIME_DIR:-$CXR_ROOT/run}"
if [[ ! -f "$CONFIG_PATH" ]]; then echo "receiver config not found: $CONFIG_PATH" >&2; exit 2; fi
if [[ ! -x "$CAMERA_ROOT/camera_streamer.sh" ]]; then echo "IsaacTeleop camera streamer missing" >&2; exit 2; fi
if [[ ! -f "$XR_JSON" || ! -f "$CXR_RUN/cloudxr.env" ]]; then
  echo "CloudXR environment is not initialized. Run orchestration/run_cloudxr_runtime.sh in a separate local terminal first." >&2
  exit 2
fi
if ! pgrep -f 'isaacteleop\.cloudxr' >/dev/null 2>&1; then
  echo "CloudXR runtime is not running. Start orchestration/run_cloudxr_runtime.sh in a separate local terminal." >&2
  exit 2
fi
if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
  echo "Building the IsaacTeleop camera receiver image (one-time operation)..." >&2
  "$CAMERA_ROOT/camera_streamer.sh" build
fi
NAME="flexiv-inspire-xr-receiver"
cleanup() { docker stop "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM
docker rm -f "$NAME" >/dev/null 2>&1 || true
ARGS=(--rm --name "$NAME" --runtime nvidia --privileged --network host --ulimit stack=33554432
      -e "XR_RUNTIME_JSON=$XR_JSON" -e "NV_CXR_RUNTIME_DIR=$CXR_RUN"
      -v /dev:/dev -v /run/udev:/run/udev:rw
      -v "$CXR_ROOT:$CXR_ROOT:ro" -v "$CAMERA_ROOT:/camera_streamer:ro"
      -v "$CONFIG_PATH:/runtime/receiver.yaml:ro")
if [[ -n "${DISPLAY:-}" ]]; then ARGS+=(-e "DISPLAY=$DISPLAY" -v /tmp/.X11-unix:/tmp/.X11-unix); fi
docker run "${ARGS[@]}" "$IMAGE_NAME" python3 /camera_streamer/teleop_camera_app.py   --config /runtime/receiver.yaml --source rtp --mode "$DISPLAY_MODE"
