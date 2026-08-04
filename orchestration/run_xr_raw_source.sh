#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EULA_MARKER="$HOME/.cloudxr/run/eula_accepted"
TRANSPORT="lan"
WIFI_CONNECTION=""
CLIENT_PER_EYE_WIDTH="1792"
CLIENT_PER_EYE_HEIGHT="1536"
CLIENT_FRAME_RATE="72"
CLIENT_MAX_BITRATE_MBPS="80"
CLIENT_CODEC="h264"
CLIENT_ENABLE_TEX_SUB_IMAGE_2D="true"

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
    --client-per-eye-width)
      CLIENT_PER_EYE_WIDTH="${2:?--client-per-eye-width requires pixels}"
      shift 2
      ;;
    --client-per-eye-height)
      CLIENT_PER_EYE_HEIGHT="${2:?--client-per-eye-height requires pixels}"
      shift 2
      ;;
    --client-frame-rate)
      CLIENT_FRAME_RATE="${2:?--client-frame-rate requires FPS}"
      shift 2
      ;;
    --client-max-bitrate-mbps)
      CLIENT_MAX_BITRATE_MBPS="${2:?--client-max-bitrate-mbps requires Mbps}"
      shift 2
      ;;
    --client-codec)
      CLIENT_CODEC="${2:?--client-codec requires h264, h265, or av1}"
      shift 2
      ;;
    --client-enable-tex-sub-image-2d)
      CLIENT_ENABLE_TEX_SUB_IMAGE_2D="${2:?--client-enable-tex-sub-image-2d requires true or false}"
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
    if ! command -v adb >/dev/null 2>&1; then
      echo "USB XR requires adb, but adb is not installed." >&2
      exit 2
    fi
    if ! command -v turnserver >/dev/null 2>&1; then
      echo "USB XR requires coturn (turnserver), but it is not installed." >&2
      exit 2
    fi
    if [[ "$(adb get-state 2>/dev/null || true)" != "device" ]]; then
      echo "Quest USB 数据连接未就绪：请解锁头显并允许 USB 调试，然后重试。" >&2
      exit 2
    fi
    # A crashed CloudXR/WSS run can leave these device-side reverse listeners
    # behind. Clear only the ports owned by this launcher before recreating
    # them; otherwise adb reports "cannot bind listener: Address already in use".
    for cloudxr_port in 8080 48322 49100; do
      adb reverse --remove "tcp:${cloudxr_port}" >/dev/null 2>&1 || true
    done
    echo "XR 使用 USB 本地链路：信令、网页和 WebRTC 媒体均经数据线。" >&2
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
  -p cloudxr_usb_local:="$CLOUDXR_USB_LOCAL" \
  -p cloudxr_client_per_eye_width:="$CLIENT_PER_EYE_WIDTH" \
  -p cloudxr_client_per_eye_height:="$CLIENT_PER_EYE_HEIGHT" \
  -p cloudxr_client_frame_rate:="$CLIENT_FRAME_RATE" \
  -p cloudxr_client_max_bitrate_mbps:="$CLIENT_MAX_BITRATE_MBPS" \
  -p cloudxr_client_codec:="$CLIENT_CODEC" \
  -p cloudxr_client_enable_tex_sub_image_2d:="$CLIENT_ENABLE_TEX_SUB_IMAGE_2D"
