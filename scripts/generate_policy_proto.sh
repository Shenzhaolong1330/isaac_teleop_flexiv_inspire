#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PROJECT_ROOT}/envs/ros-py312/bin/python"
PROTO_DIR="${PROJECT_ROOT}/interfaces/proto"
OUTPUT_DIR="${PROJECT_ROOT}/src/flexiv_inspire_isaac/policy_api/generated"
"${PYTHON_BIN}" -m grpc_tools.protoc \
  -I"${PROTO_DIR}" \
  --python_out="${OUTPUT_DIR}" \
  --grpc_python_out="${OUTPUT_DIR}" \
  "${PROTO_DIR}/policy_service_v1.proto"
sed -i 's/^import policy_service_v1_pb2 as /from . import policy_service_v1_pb2 as /' \
  "${OUTPUT_DIR}/policy_service_v1_pb2_grpc.py"
PYTHONPATH="${PROJECT_ROOT}/core/src:${PROJECT_ROOT}/src" \
  "${PYTHON_BIN}" -m py_compile "${OUTPUT_DIR}"/*.py
