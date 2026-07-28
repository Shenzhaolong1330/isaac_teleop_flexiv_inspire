#!/usr/bin/env bash
set -euo pipefail

SYSTEM_NVCC="$(command -v nvcc)"
SYSTEM_VERSION="$("${SYSTEM_NVCC}" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p' | tail -n 1)"
ISAAC_NVCC="/usr/local/cuda-12.8/bin/nvcc"
[[ -x "${ISAAC_NVCC}" ]] || {
  printf 'ERROR: CUDA 12.8 nvcc is missing at %s\n' "${ISAAC_NVCC}" >&2
  exit 1
}
ISAAC_VERSION="$("${ISAAC_NVCC}" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p' | tail -n 1)"
DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)"

[[ "${SYSTEM_NVCC}" == "/usr/bin/nvcc" ]] || {
  printf 'ERROR: default nvcc changed: %s\n' "${SYSTEM_NVCC}" >&2
  exit 1
}
[[ "${SYSTEM_VERSION}" == 12.0* ]] || {
  printf 'ERROR: expected system CUDA 12.0, got %s\n' "${SYSTEM_VERSION}" >&2
  exit 1
}
[[ "${ISAAC_VERSION}" == 12.8* ]] || {
  printf 'ERROR: expected isolated CUDA 12.8, got %s\n' "${ISAAC_VERSION}" >&2
  exit 1
}
[[ "${DRIVER_VERSION}" == 580.173.02* ]] || {
  printf 'ERROR: NVIDIA driver changed: %s\n' "${DRIVER_VERSION}" >&2
  exit 1
}
[[ ! -e /usr/local/cuda ]] || {
  printf 'ERROR: generic /usr/local/cuda must remain absent\n' >&2
  exit 1
}

printf 'PASS: system nvcc=%s, Isaac nvcc=%s, driver=%s\n' \
  "${SYSTEM_VERSION}" "${ISAAC_VERSION}" "${DRIVER_VERSION}"
