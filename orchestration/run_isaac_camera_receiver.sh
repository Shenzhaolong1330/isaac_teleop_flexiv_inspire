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
deadline=$((SECONDS + 60))
while (( SECONDS < deadline )); do
  if [[ -f "$XR_JSON" && -f "$CXR_RUN/cloudxr.env" && -f "$CXR_RUN/cloudxr.pid" ]]; then
    runtime_pid="$(tr -dc '0-9' < "$CXR_RUN/cloudxr.pid")"
    if [[ -n "$runtime_pid" ]] && kill -0 "$runtime_pid" 2>/dev/null; then
      break
    fi
  fi
  sleep 0.2
done
if [[ ! -f "$XR_JSON" || ! -f "$CXR_RUN/cloudxr.env" || ! -f "$CXR_RUN/cloudxr.pid" ]]; then
  echo "CloudXR runtime did not become ready within 60 seconds." >&2
  exit 2
fi
runtime_pid="$(tr -dc '0-9' < "$CXR_RUN/cloudxr.pid")"
if [[ -z "$runtime_pid" ]] || ! kill -0 "$runtime_pid" 2>/dev/null; then
  echo "CloudXR runtime process is not alive after 60 seconds." >&2
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
