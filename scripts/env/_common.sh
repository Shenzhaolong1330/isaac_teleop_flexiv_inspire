#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  printf 'ERROR: source this helper from an environment activation script.\n' >&2
  exit 2
fi

_TELEOP_ROOT_CANDIDATE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -n "${TELEOP_ROOT:-}" ]]; then
  if [[ "${TELEOP_ROOT}" != "${_TELEOP_ROOT_CANDIDATE}" ]]; then
    printf 'ERROR: TELEOP_ROOT already points to a different project: %s\n' "${TELEOP_ROOT}" >&2
    return 1
  fi
else
  TELEOP_ROOT="${_TELEOP_ROOT_CANDIDATE}"
fi
readonly TELEOP_ROOT
unset _TELEOP_ROOT_CANDIDATE

teleop_activate_env() {
  local environment_name="$1"
  local remove_ros="${2:-no}"
  local environment_path="${TELEOP_ROOT}/envs/${environment_name}"
  local conda_variable

  if [[ ! -x "${environment_path}/bin/python" ]]; then
    printf 'ERROR: environment is missing: %s\n' "${environment_path}" >&2
    return 1
  fi

  export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
  unset PYTHONPATH LD_LIBRARY_PATH CMAKE_PREFIX_PATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH
  unset VIRTUAL_ENV
  for conda_variable in ${!CONDA_@} ${!_CONDA@} ${!_CE_@}; do
    unset "${conda_variable}"
  done
  unset CUDA_HOME CUDA_PATH CUDACXX CUDA_ROOT CUDA_TOOLKIT_ROOT_DIR NVCC_PREPEND_FLAGS
  export PYTHONNOUSERSITE=1
  if [[ "${remove_ros}" == "yes" ]]; then
    unset ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION ROS_LOCALHOST_ONLY ROS_AUTOMATIC_DISCOVERY_RANGE ROS_STATIC_PEERS
    unset ROS_DOMAIN_ID RMW_IMPLEMENTATION CYCLONEDDS_URI
  fi

  # shellcheck disable=SC1090
  source "${environment_path}/bin/activate"
  export ISAAC_TELEOP_ROOT="${TELEOP_ROOT}"
  hash -r
}

teleop_select_cuda_12_8() {
  export CUDA_HOME=/usr/local/cuda-12.8
  export CUDA_PATH="${CUDA_HOME}"
  export CUDACXX="${CUDA_HOME}/bin/nvcc"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

  if [[ ! -x "${CUDACXX}" ]]; then
    printf 'ERROR: CUDA 12.8 is missing at %s\n' "${CUDACXX}" >&2
    return 1
  fi
}

teleop_source_ros_jazzy() {
  local restore_nounset=no

  if [[ "$-" == *u* ]]; then
    restore_nounset=yes
    set +u
  fi

  # ROS setup scripts reference optional variables that may be unset.
  # shellcheck disable=SC1091
  source /opt/ros/jazzy/setup.bash

  if [[ "${restore_nounset}" == "yes" ]]; then
    set -u
  fi
}
