#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
CERT_DIR="${PROJECT_ROOT}/certs"
PRIVATE_DIR="${CERT_DIR}/private"
CERT_FILE="${CERT_DIR}/server.crt"
KEY_FILE="${PRIVATE_DIR}/server.key"

if [[ -e "${CERT_FILE}" || -e "${KEY_FILE}" ]]; then
  printf 'Refusing to overwrite an existing policy certificate/key.\n' >&2
  printf 'Certificate: %s\nKey: %s\n' "${CERT_FILE}" "${KEY_FILE}" >&2
  exit 2
fi

mkdir -p "${PRIVATE_DIR}"
umask 077
openssl req \
  -x509 \
  -newkey rsa:3072 \
  -sha256 \
  -nodes \
  -days 365 \
  -subj '/CN=flexiv-policy-loopback' \
  -addext 'subjectAltName=DNS:localhost,IP:127.0.0.1,IP:::1' \
  -keyout "${KEY_FILE}" \
  -out "${CERT_FILE}"
chmod 0600 "${KEY_FILE}"
chmod 0644 "${CERT_FILE}"
printf 'Created loopback-only PolicyService certificate:\n  %s\n  %s\n' \
  "${CERT_FILE}" "${KEY_FILE}"
