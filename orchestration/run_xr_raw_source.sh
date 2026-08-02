#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EULA_MARKER="$HOME/.cloudxr/run/eula_accepted"
TRANSPORT="lan"
WIFI_CONNECTION=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --transport)
      TRANSPORT="${2:?--transport requires lan or usb_tcp}"
      shift 2
      ;;
    --wifi-connection)
      WIFI_CONNECTION="${2:?--wifi-connection requires a NetworkManager profile}"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

case "$TRANSPORT" in
  lan)
    CLOUDXR_USB_LOCAL=false
    if [[ -n "$WIFI_CONNECTION" ]]; then
      if ! command -v nmcli >/dev/null 2>&1; then
        echo "LAN XR transport requires NetworkManager (nmcli)." >&2
        exit 2
      fi
      if ! nmcli -t -f NAME connection show --active \
          | grep -Fqx -- "$WIFI_CONNECTION"; then
        echo "正在自动连接 XR Wi-Fi: $WIFI_CONNECTION" >&2
        if ! nmcli --wait 20 connection up "$WIFI_CONNECTION"; then
          echo "无法连接 XR Wi-Fi '$WIFI_CONNECTION'。请确认该网络已保存。" >&2
          exit 2
        fi
      fi
      echo "XR 使用局域网低延迟直连: $WIFI_CONNECTION" >&2
    fi
    ;;
  usb_tcp)
    CLOUDXR_USB_LOCAL=true
    echo "XR 使用 USB/TCP 兼容模式；该模式延迟较高。" >&2
    ;;
  *)
    echo "Unsupported XR transport '$TRANSPORT' (expected lan or usb_tcp)." >&2
    exit 2
    ;;
esac

if [[ ! -f "$EULA_MARKER" ]]; then
  echo "CloudXR EULA has not been accepted by the local operator." >&2
  echo "Run flexiv-inspire-xr-raw-source once in a local terminal to review it." >&2
  exit 2
fi

# shellcheck disable=SC1091
source "$PROJECT_ROOT/scripts/env/activate_isaac.sh"
if [[ -f "$PROJECT_ROOT/ros2_ws/install/setup.bash" ]]; then
  # shellcheck disable=SC1091
  set +u
  source "$PROJECT_ROOT/ros2_ws/install/setup.bash"
  set -u
fi

exec "$PROJECT_ROOT/envs/isaac-py312/bin/flexiv-inspire-xr-raw-source" \
  --ros-args \
  -p cloudxr_accept_eula:=false \
  -p cloudxr_setup_oob:=true \
  -p cloudxr_usb_local:="$CLOUDXR_USB_LOCAL"
