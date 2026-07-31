#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CAMERA_ROOT="$PROJECT_ROOT/third_party/IsaacTeleop/examples/camera_streamer"
if [[ ! -x "$CAMERA_ROOT/camera_streamer.sh" ]]; then
  echo "IsaacTeleop camera receiver source is missing: $CAMERA_ROOT" >&2
  exit 2
fi
if ! docker info >/dev/null 2>&1; then
  echo "Docker is unavailable to this login. Log out/in after adding hb to the docker group, then retry." >&2
  exit 2
fi
echo "Building the official IsaacTeleop camera-receiver image. This is a one-time multi-GB download."
"$CAMERA_ROOT/camera_streamer.sh" build
