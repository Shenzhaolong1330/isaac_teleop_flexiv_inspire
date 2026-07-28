#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/jazzy/setup.bash

exec "${ROOT}/.venv/bin/python" \
  -m flexiv_inspire_isaac.synthetic_source "$@"

