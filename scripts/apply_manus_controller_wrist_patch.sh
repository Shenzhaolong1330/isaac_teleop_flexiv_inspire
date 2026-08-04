#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM="$PROJECT_ROOT/third_party/IsaacTeleop"
PATCH="$PROJECT_ROOT/patches/isaac_teleop/manus_controller_wrist.patch"
ERGONOMICS_PATCH="$PROJECT_ROOT/patches/isaac_teleop/manus_ergonomics_udp.patch"

if [[ ! -d "$UPSTREAM/.git" ]]; then
  echo "IsaacTeleop checkout is missing: $UPSTREAM" >&2
  exit 2
fi

apply_patch_once() {
  local patch_path="$1"
  local label="$2"
  if git -C "$UPSTREAM" apply --reverse --check "$patch_path" >/dev/null 2>&1; then
    echo "IsaacTeleop MANUS $label patch is already applied"
  elif git -C "$UPSTREAM" apply --check "$patch_path"; then
    git -C "$UPSTREAM" apply "$patch_path"
    echo "Applied IsaacTeleop MANUS $label patch"
  else
    echo "IsaacTeleop MANUS source does not match the pinned $label patch" >&2
    exit 2
  fi
}

# The Ergonomics patch is intentionally based on the controller-wrist patch
# and clang-formats the shared hunks. Check the final patch first because that
# formatting means the legacy patch is no longer exactly reverse-applicable.
if git -C "$UPSTREAM" apply --reverse --check "$ERGONOMICS_PATCH" >/dev/null 2>&1; then
  echo "IsaacTeleop MANUS controller-wrist patch is already applied"
  echo "IsaacTeleop MANUS Ergonomics-UDP patch is already applied"
  exit 0
fi

apply_patch_once "$PATCH" "controller-wrist"
apply_patch_once "$ERGONOMICS_PATCH" "Ergonomics-UDP"
