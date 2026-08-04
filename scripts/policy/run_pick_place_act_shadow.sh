#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/hb/isaac_teleop_flexiv_inspire
DUAL_ARM_ROOT=/home/hb/flexiv_inspire_ws/src/dual_arm_teleop
LEROBOT_ROOT=/home/hb/flexiv_inspire_ws/src/Le-nero
RECORD=/home/hb/miniconda3/envs/flexiv_teleop/bin/robot-record
CONFIG="$ROOT/config/policies/pick_place_act_shadow.yaml"
CHECKPOINT="$ROOT/artifacts/policies/pick_place_demo_act_v1/checkpoints/last/pretrained_model"

for required in config.json model.safetensors; do
  if [[ ! -f "$CHECKPOINT/$required" ]]; then
    echo "checkpoint 不完整，缺少：$CHECKPOINT/$required" >&2
    echo "先运行：$ROOT/scripts/policy/train_pick_place_act.sh" >&2
    exit 2
  fi
done

source "$ROOT/scripts/env/activate_ros.sh"
source "$ROOT/ros2_ws/install/setup.bash"

for topic in /robot/left_arm/state /robot/right_arm/state; do
  if ! ros2 topic list | grep -Fxq "$topic"; then
    echo "缺少真机观测 topic：$topic" >&2
    echo "先在另一个终端启动本仓库的 robot record，使 RDK、双手和相机保持运行。" >&2
    exit 2
  fi
done

export PYTHONPATH="$LEROBOT_ROOT/src:$DUAL_ARM_ROOT:$ROOT/libs/policy_contracts/src:$ROOT/libs/policy_client/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$RECORD" --config "$CONFIG"
