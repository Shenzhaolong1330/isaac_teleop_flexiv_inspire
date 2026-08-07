#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE="$ROOT/third_party/oculus_reader"
PYTHON="$ROOT/envs/ros-py312/bin/python"
UV="${UV_BIN:-/home/hb/.local/bin/uv}"
REVISION=9689484d319c4798e54d59509b192436647b7427
APK="$SOURCE/oculus_reader/APK/teleop-debug.apk"

command -v git-lfs >/dev/null 2>&1 || {
  printf 'ERROR: git-lfs is required for the OculusReader Quest APK.\n' >&2
  exit 1
}
[[ -x "$PYTHON" ]] || {
  printf 'ERROR: create ros-py312 before installing OculusReader.\n' >&2
  exit 1
}
[[ -x "$UV" ]] || {
  printf 'ERROR: uv not found at %s\n' "$UV" >&2
  exit 1
}

if [[ ! -d "$SOURCE/.git" ]]; then
  git clone https://github.com/jborbik/oculus_reader.git "$SOURCE"
fi

if [[ -n "$(git -C "$SOURCE" status --short)" ]]; then
  printf 'ERROR: managed OculusReader checkout has local changes: %s\n' "$SOURCE" >&2
  exit 1
fi

git -C "$SOURCE" fetch origin "$REVISION"
git -C "$SOURCE" checkout --detach "$REVISION"
git -C "$SOURCE" lfs install --local
git -C "$SOURCE" lfs pull --include=oculus_reader/APK/teleop-debug.apk

apk_size="$(stat -c %s "$APK" 2>/dev/null || printf 0)"
if (( apk_size < 1000000 )); then
  printf 'ERROR: OculusReader APK is missing or still a Git LFS pointer: %s\n' "$APK" >&2
  exit 1
fi

"$UV" pip install --python "$PYTHON" --no-deps --editable "$SOURCE"
printf 'OculusReader %s installed; APK size=%s bytes\n' "$REVISION" "$apk_size"
