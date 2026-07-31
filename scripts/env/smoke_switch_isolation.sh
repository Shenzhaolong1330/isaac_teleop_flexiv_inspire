#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

for ignored_tree in envs third_party vendor; do
  [[ -f "${ROOT}/${ignored_tree}/COLCON_IGNORE" ]] || {
    printf 'ERROR: %s/COLCON_IGNORE is missing\n' "${ignored_tree}" >&2
    exit 1
  }
done

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

assert_eq() {
  local name="$1"
  local expected="$2"
  local actual="${!name-}"
  [[ "${actual}" == "${expected}" ]] ||
    fail "${name}: expected '${expected}', got '${actual}'"
}

assert_unset() {
  local name="$1"
  [[ -z "${!name+x}" ]] || fail "${name} leaked into the active environment"
}

seed_conda_leak() {
  export CONDA_PREFIX=/home/hb/miniconda3/envs/legacy
  export CONDA_DEFAULT_ENV=legacy
  export CONDA_PROMPT_MODIFIER='(legacy) '
  export CONDA_EXE=/home/hb/miniconda3/bin/conda
  export CONDA_PYTHON_EXE=/home/hb/miniconda3/bin/python
  export CONDA_SHLVL=1
  export _CONDA_ROOT=/home/hb/miniconda3
  export _CONDA_EXE=/home/hb/miniconda3/bin/conda
  export _CE_CONDA=legacy
  export _CE_M=legacy
  export _CE_TEST_LEAK=legacy
}

assert_no_conda() {
  local name
  for name in \
    CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_EXE \
    CONDA_PYTHON_EXE CONDA_SHLVL _CONDA_ROOT _CONDA_EXE \
    _CE_CONDA _CE_M _CE_TEST_LEAK; do
    assert_unset "${name}"
  done
  [[ "${PATH}" != *"/home/hb/miniconda3"* ]] ||
    fail "legacy Conda path leaked into PATH"
  [[ "${LD_LIBRARY_PATH-}" != *"/home/hb/miniconda3"* ]] ||
    fail "legacy Conda path leaked into LD_LIBRARY_PATH"
}

assert_python() {
  local environment_name="$1"
  local expected="${ROOT}/envs/${environment_name}/bin/python"
  [[ "$(readlink -f "$(command -v python)")" == "$(readlink -f "${expected}")" ]] ||
    fail "python does not belong to ${environment_name}"
  assert_eq PYTHONNOUSERSITE 1
  assert_no_conda
}

seed_conda_leak
# shellcheck disable=SC1091
source "${ROOT}/scripts/env/activate_isaac.sh"
assert_python isaac-py312
assert_eq CUDA_HOME /usr/local/cuda-12.8
assert_eq CUDACXX /usr/local/cuda-12.8/bin/nvcc
assert_eq RMW_IMPLEMENTATION rmw_cyclonedds_cpp
assert_eq ROS_DOMAIN_ID 42
assert_eq ROS_LOCALHOST_ONLY 1
assert_eq ROS_AUTOMATIC_DISCOVERY_RANGE LOCALHOST

seed_conda_leak
# shellcheck disable=SC1091
source "${ROOT}/scripts/env/activate_rdk.sh"
assert_python rdk-py310
assert_unset CUDA_HOME
assert_unset CUDA_PATH
assert_unset CUDACXX
assert_unset ROS_DISTRO
assert_unset RMW_IMPLEMENTATION
assert_unset ROS_DOMAIN_ID
[[ "$(command -v nvcc)" == /usr/bin/nvcc ]] ||
  fail "RDK inherited the isolated CUDA path"

seed_conda_leak
# shellcheck disable=SC1091
source "${ROOT}/scripts/env/activate_ros.sh"
assert_python ros-py312
assert_unset CUDA_HOME
assert_unset CUDA_PATH
assert_unset CUDACXX
assert_eq RMW_IMPLEMENTATION rmw_cyclonedds_cpp
assert_eq ROS_DOMAIN_ID 42
assert_eq ROS_LOCALHOST_ONLY 1
assert_eq ROS_AUTOMATIC_DISCOVERY_RANGE LOCALHOST

seed_conda_leak
# shellcheck disable=SC1091
source "${ROOT}/scripts/env/activate_data.sh"
assert_python data-py312
assert_eq CUDA_HOME /usr/local/cuda-12.8
assert_eq CUDACXX /usr/local/cuda-12.8/bin/nvcc
assert_unset ROS_DISTRO
assert_unset RMW_IMPLEMENTATION
assert_unset ROS_DOMAIN_ID

/usr/bin/nvcc --version | grep -q 'release 12\.0'
printf 'PASS: isaac -> rdk -> ros -> data activation is isolated and repeatable.\n'
