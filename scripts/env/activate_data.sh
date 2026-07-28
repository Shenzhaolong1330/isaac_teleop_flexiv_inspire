#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  printf 'Usage: source scripts/env/activate_data.sh\n' >&2
  exit 2
fi

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
teleop_activate_env data-py312 yes
teleop_select_cuda_12_8
