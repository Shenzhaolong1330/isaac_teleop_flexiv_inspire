#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM="$PROJECT_ROOT/third_party/IsaacTeleop"
PATCH="$PROJECT_ROOT/patches/isaac_teleop/manus_controller_wrist.patch"

if [[ ! -d "$UPSTREAM/.git" ]]; then
  echo "IsaacTeleop checkout is missing: $UPSTREAM" >&2
  exit 2
fi

if git -C "$UPSTREAM" apply --reverse --check "$PATCH" >/dev/null 2>&1; then
  echo "IsaacTeleop MANUS controller-wrist patch is already applied"
elif git -C "$UPSTREAM" apply --check "$PATCH"; then
  git -C "$UPSTREAM" apply "$PATCH"
  echo "Applied IsaacTeleop MANUS controller-wrist patch"
else
  echo "IsaacTeleop MANUS source does not match the pinned patch" >&2
  exit 2
fi
