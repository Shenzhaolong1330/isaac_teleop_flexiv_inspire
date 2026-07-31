#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
ROS_PYTHON="${ROOT}/envs/ros-py312/bin/python"
POLICY_OUT="${ROOT}/apps/policy_server/src/flexiv_inspire_isaac/policy_api/generated"

if [[ ! -x "${ROS_PYTHON}" ]]; then
  printf 'ERROR: missing ROS environment: %s\n' "${ROS_PYTHON}" >&2
  exit 2
fi

bash "${ROOT}/apps/flexiv_daemon/scripts/generate_proto.sh"
mkdir -p "${POLICY_OUT}"
"${ROS_PYTHON}" -m grpc_tools.protoc \
  --proto_path="${ROOT}/libs/rpc_interfaces/proto" \
  --python_out="${POLICY_OUT}" \
  --grpc_python_out="${POLICY_OUT}" \
  "${ROOT}/libs/rpc_interfaces/proto/policy_service_v1.proto"

# grpc_tools generates a top-level import; keep the generated module package-local.
sed -i \
  's/^import policy_service_v1_pb2 as policy__service__v1__pb2$/from . import policy_service_v1_pb2 as policy__service__v1__pb2/' \
  "${POLICY_OUT}/policy_service_v1_pb2_grpc.py"
chmod 0644 \
  "${ROOT}/apps/flexiv_daemon/src/flexiv_rdk_daemon/generated/rdk_ipc_pb2.py" \
  "${POLICY_OUT}/policy_service_v1_pb2.py" \
  "${POLICY_OUT}/policy_service_v1_pb2_grpc.py"

printf 'Regenerated canonical RDK and PolicyService Python protobuf stubs.\n'
