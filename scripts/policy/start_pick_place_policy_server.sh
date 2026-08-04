#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/hb/isaac_teleop_flexiv_inspire
source "$ROOT/scripts/env/activate_ros.sh"
source "$ROOT/ros2_ws/install/setup.bash"

exec flexiv-inspire-policy-server \
  --bind 127.0.0.1 \
  --port 50051 \
  --server-cert "$ROOT/certs/server.crt" \
  --server-key "$ROOT/certs/private/server.key" \
  --arm-rate-hz 200 \
  --hand-rate-hz 15 \
  --camera-rate-hz 15 \
  --action-rate-hz 30
