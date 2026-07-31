#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UV="${UV_BIN:-/home/hb/.local/bin/uv}"
PY312=/usr/bin/python3.12
PY310_VERSION=3.10.19

[[ -x "${UV}" ]] || {
  printf 'ERROR: uv not found at %s\n' "${UV}" >&2
  exit 1
}
[[ -x "${PY312}" ]] || {
  printf 'ERROR: Python 3.12 not found at %s\n' "${PY312}" >&2
  exit 1
}
[[ -x /usr/local/cuda-12.8/bin/nvcc ]] || {
  printf 'ERROR: install CUDA Toolkit 12.8 before creating environments.\n' >&2
  exit 1
}

mkdir -p "${ROOT}/envs" "${ROOT}/third_party" "${ROOT}/vendor" "${ROOT}/requirements"
# These large non-workspace trees contain third-party setup.py/CMake projects.
# Keep accidental bare `colcon build` from recursively treating them as ROS
# packages; the supported build still uses exact --base-paths.
touch   "${ROOT}/envs/COLCON_IGNORE"   "${ROOT}/third_party/COLCON_IGNORE"   "${ROOT}/vendor/COLCON_IGNORE"
"${UV}" python install "${PY310_VERSION}"
PY310="$("${UV}" python find "${PY310_VERSION}")"

create_venv() {
  local name="$1"
  local python="$2"
  if [[ ! -x "${ROOT}/envs/${name}/bin/python" ]]; then
    "${UV}" venv --python "${python}" "${ROOT}/envs/${name}"
  fi
}

lock_and_sync() {
  local name="$1"
  local input="${ROOT}/requirements/${name}.in"
  local lock="${ROOT}/requirements/${name}.lock"
  local python="${ROOT}/envs/${name}/bin/python"

  if [[ ! -f "${lock}" ]] || [[ "${RELOCK:-0}" == "1" ]]; then
    "${UV}" pip compile \
      --python "${python}" \
      --index-strategy unsafe-best-match \
      --generate-hashes \
      --output-file "${lock}" \
      "${input}"
  fi
  "${UV}" pip sync \
    --python "${python}" \
    --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    "${lock}"
}

configure_ros_vendor_bindings() {
  local ros_python="${ROOT}/envs/ros-py312/bin/python"
  local isaac_python="${ROOT}/envs/isaac-py312/bin/python"
  local ros_site_packages
  local isaac_site_packages
  local vendor_path=/usr/local/lib/python3.12/dist-packages
  local system_path=/usr/lib/python3/dist-packages

  [[ -d "${vendor_path}" ]] || {
    printf 'ERROR: local ROS vendor bindings are missing: %s\n' "${vendor_path}" >&2
    return 1
  }
  [[ -d "${system_path}" ]] || {
    printf 'ERROR: system ROS Python dependencies are missing: %s\n' "${system_path}" >&2
    return 1
  }
  ros_site_packages="$("${ros_python}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  isaac_site_packages="$("${isaac_python}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  printf '%s\n' "${vendor_path}" > "${ros_site_packages}/local-ros-vendor-bindings.pth"
  printf '%s\n' "${system_path}" > "${ros_site_packages}/system-ros-python.pth"
  printf '%s\n' "${ROOT}/apps/flexiv_daemon/src" > "${ros_site_packages}/local-rdk-ipc.pth"
  printf '%s\n' "${system_path}" > "${isaac_site_packages}/system-ros-python.pth"
}

install_local_projects() {
  local environment_name="$1"
  shift
  local python="${ROOT}/envs/${environment_name}/bin/python"
  local project

  for project in "$@"; do
    if [[ ! -f "${ROOT}/${project}/pyproject.toml" ]]; then
      printf 'ERROR: local project is missing pyproject.toml: %s\n' "${project}" >&2
      return 1
    fi
    "${UV}" pip install --python "${python}" --no-deps --editable "${ROOT}/${project}"
  done
}

create_venv isaac-py312 "${PY312}"
create_venv ros-py312 "${PY312}"
create_venv rdk-py310 "${PY310}"
create_venv data-py312 "${PY312}"

lock_and_sync rdk-py310
lock_and_sync ros-py312
lock_and_sync isaac-py312
lock_and_sync data-py312
configure_ros_vendor_bindings

install_local_projects rdk-py310 libs/control_core apps/flexiv_daemon
install_local_projects ros-py312 libs/control_core .
install_local_projects isaac-py312 libs/control_core .
install_local_projects data-py312 libs/control_core .

printf 'Created and synchronized four isolated environments under %s/envs\n' "${ROOT}"
