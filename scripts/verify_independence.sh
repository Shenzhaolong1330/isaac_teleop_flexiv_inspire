#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
FORBIDDEN_ROOT="/home/hb/flexiv_inspire_ws"
FORBIDDEN_PATTERN='(/home/hb/flexiv_inspire_ws|/home/hb/(mini|ana)conda[^/]*|FlexivDualArm|flexiv_teleop)'
EXPECTED_PROJECT_VERSION="$(
  awk '
    /^\[project\]$/ { in_project=1; next }
    /^\[/ { in_project=0 }
    in_project && /^version[[:space:]]*=/ {
      value=$0
      sub(/^[^=]*=[[:space:]]*"/, "", value)
      sub(/".*$/, "", value)
      print value
      exit
    }
  ' "${ROOT}/pyproject.toml"
)"
FAILED=0

check_tree_text() {
  local path="$1"
  [[ -e "${path}" ]] || return 0
  if grep -RInE \
      --exclude-dir=.git \
      --exclude-dir=__pycache__ \
      --exclude='*.pyc' \
      --exclude='verify_independence.sh' \
      --exclude='smoke_switch_isolation.sh' \
      "${FORBIDDEN_PATTERN}" \
      "${path}"; then
    printf 'ERROR: forbidden legacy runtime reference under %s\n' "${path}" >&2
    FAILED=1
  fi
}

check_text_file() {
  local file="$1"
  if grep -InE "${FORBIDDEN_PATTERN}" "${file}"; then
    printf 'ERROR: environment metadata contains a forbidden legacy path: %s\n' \
      "${file}" >&2
    FAILED=1
  fi
}

for runtime_path in \
  "${ROOT}/configs" \
  "${ROOT}/scripts" \
  "${ROOT}/src" \
  "${ROOT}/core" \
  "${ROOT}/interfaces" \
  "${ROOT}/rdk_daemon" \
  "${ROOT}/dftp" \
  "${ROOT}/teleop" \
  "${ROOT}/policy_api" \
  "${ROOT}/data_pipeline" \
  "${ROOT}/cameras" \
  "${ROOT}/ros_ws"; do
  check_tree_text "${runtime_path}"
done

while IFS= read -r -d '' link; do
  resolved="$(readlink -f -- "${link}")"
  if [[ "${resolved}" == "${FORBIDDEN_ROOT}"* ]]; then
    printf 'ERROR: symlink %s resolves into legacy workspace: %s\n' \
      "${link}" "${resolved}" >&2
    FAILED=1
  fi
done < <(find "${ROOT}" \
  \( -path "${ROOT}/.git" -o -path "${ROOT}/upstream" -o -path "${ROOT}/envs" \) \
  -prune -o -type l -print0)

for env_path in "${ROOT}"/envs/*; do
  [[ -d "${env_path}" ]] || continue

  environment_hit="$(
    grep -RIlE --exclude='*.pyc' "${FORBIDDEN_PATTERN}" \
      "${env_path}/pyvenv.cfg" "${env_path}/bin" 2>/dev/null |
      sed -n '1p' || true
  )"
  if [[ -n "${environment_hit}" ]]; then
    printf 'ERROR: environment %s contains a legacy-workspace or Conda path\n' \
      "${env_path}" >&2
    printf '  %s\n' "${environment_hit}" >&2
    FAILED=1
  fi

  # Scan only editable path files and textual distribution metadata. Avoid
  # recursively reading native wheels and shared libraries in site-packages.
  while IFS= read -r -d '' metadata_file; do
    check_text_file "${metadata_file}"
  done < <(
    find "${env_path}" -type f \
      \( \
        -path '*/site-packages/*.pth' -o \
        -path '*/site-packages/*.dist-info/METADATA' -o \
        -path '*/site-packages/*.dist-info/direct_url.json' -o \
        -path '*/site-packages/*.dist-info/entry_points.txt' \
      \) -print0 2>/dev/null
  )

  mapfile -d '' project_metadata < <(
    find "${env_path}" -type f \
      -path '*/site-packages/flexiv_inspire_isaac-*.dist-info/METADATA' \
      -print0 2>/dev/null
  )
  if (( ${#project_metadata[@]} > 1 )); then
    printf 'ERROR: environment %s contains multiple flexiv-inspire-isaac installs\n' \
      "${env_path}" >&2
    FAILED=1
  fi
  for metadata_file in "${project_metadata[@]}"; do
    if ! grep -qxF "Version: ${EXPECTED_PROJECT_VERSION}" "${metadata_file}"; then
      printf 'ERROR: stale flexiv-inspire-isaac version in %s; expected %s\n' \
        "${metadata_file}" "${EXPECTED_PROJECT_VERSION}" >&2
      FAILED=1
    fi
  done

  while IFS= read -r -d '' editable_file; do
    editable_name="$(basename -- "${editable_file}")"
    if [[ "${editable_name}" == *flexiv_inspire_isaac* &&
          "${editable_name}" != *"${EXPECTED_PROJECT_VERSION}"* ]]; then
      printf 'ERROR: stale editable install marker: %s\n' "${editable_file}" >&2
      FAILED=1
    fi
  done < <(
    find "${env_path}" -type f \
      -path '*/site-packages/*flexiv_inspire_isaac*.pth' -print0 2>/dev/null
  )
done

if (( FAILED != 0 )); then
  exit 1
fi

printf 'PASS: runtime source, launch files, environment metadata, and symlinks are independent.\n'
