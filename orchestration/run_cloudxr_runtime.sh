#!/usr/bin/env bash
# Start the upstream CloudXR runtime without accepting its license on behalf of
# the operator. Keep this process running for the whole Quest session.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_ROOT="$PROJECT_ROOT/third_party/IsaacTeleop"

if [[ ! -d "$UPSTREAM_ROOT" ]]; then
  echo "IsaacTeleop checkout is missing: $UPSTREAM_ROOT" >&2
  exit 2
fi

# shellcheck disable=SC1091
source "$PROJECT_ROOT/scripts/env/activate_isaac.sh"
export GIT_ROOT="$UPSTREAM_ROOT"
# shellcheck disable=SC1091
source "$UPSTREAM_ROOT/scripts/setup_cloudxr_env.sh"

if ! python -c 'import isaacteleop.cloudxr' >/dev/null 2>&1; then
  echo "IsaacTeleop CloudXR extra is not installed in envs/isaac-py312." >&2
  exit 2
fi

echo "CloudXR will show the license prompt unless this local operator explicitly supplies --accept-eula."
exec python -m isaacteleop.cloudxr "$@"
