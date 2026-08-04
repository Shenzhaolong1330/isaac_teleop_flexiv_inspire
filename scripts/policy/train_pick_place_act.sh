#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/hb/isaac_teleop_flexiv_inspire
DUAL_ARM_ROOT=/home/hb/flexiv_inspire_ws/src/dual_arm_teleop
LEROBOT_ROOT=/home/hb/flexiv_inspire_ws/src/Le-nero
TRAIN=/home/hb/miniconda3/envs/flexiv_teleop/bin/robot-train
CONFIG="$ROOT/config/policies/pick_place_act_train.yaml"
DATASET="$ROOT/sessions/pick_place_demo/lerobot_merged/dual_arm_lerobot_v1"
OUTPUT="$ROOT/artifacts/policies/pick_place_demo_act_v1"

if [[ ! -f "$DATASET/meta/info.json" ]]; then
  echo "训练数据不存在：$DATASET" >&2
  echo "先运行 robot convert --conversion-config config/conversion_dual_arm.yaml" >&2
  exit 2
fi
export PYTHONPATH="$LEROBOT_ROOT/src:$DUAL_ARM_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ -e "$OUTPUT" ]]; then
  LAST_STATE="$OUTPUT/checkpoints/last/training_state/training_step.json"
  if [[ ! -f "$LAST_STATE" ]]; then
    echo "训练目录已存在，但没有可恢复的 last checkpoint：$OUTPUT" >&2
    echo "请先人工检查该目录；脚本不会覆盖或删除它。" >&2
    exit 2
  fi
  RESUME_CONFIG="$(mktemp /tmp/pick-place-act-resume.XXXXXX.yaml)"
  trap 'rm -f -- "$RESUME_CONFIG"' EXIT
  sed 's/^  resume: false$/  resume: true/' "$CONFIG" >"$RESUME_CONFIG"
  echo "发现 last checkpoint，继续训练：$LAST_STATE"
  "$TRAIN" --config "$RESUME_CONFIG"
  exit $?
fi

exec "$TRAIN" --config "$CONFIG"
