#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RDK_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${RDK_ROOT}/../.." && pwd)"
PROTO="${WORKSPACE_ROOT}/libs/rpc_interfaces/proto/rdk_ipc.proto"
OUT="${RDK_ROOT}/src/flexiv_rdk_daemon/generated"

test -f "${PROTO}"
mkdir -p "${OUT}"
protoc --proto_path="${WORKSPACE_ROOT}/libs/rpc_interfaces/proto" \
  --python_out="${OUT}" \
  "${PROTO}"
chmod 0644 "${OUT}/rdk_ipc_pb2.py"
