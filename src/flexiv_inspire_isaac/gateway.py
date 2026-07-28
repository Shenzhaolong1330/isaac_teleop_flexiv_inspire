"""Python 3.10 hardware gateway for the existing Flexiv/LeRobot environment."""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import stat
import time
import uuid
from pathlib import Path
from typing import Any

from .config import BridgeConfig, load_config
from .existing_stack import ExistingStackAdapter
from .protocol import decode_packet, encode_packet, validate_action

LOGGER = logging.getLogger("isaac_flexiv_gateway")
CONFIRMATION_TEXT = "FLEXIV-AREA-CLEAR"


def _safe_remove_socket(path_text: str) -> None:
    path = Path(path_text)
    resolved_parent = path.parent.resolve()
    if resolved_parent != Path("/tmp"):
        raise ValueError(f"IPC socket must be directly under /tmp: {path}")
    if not path.exists() and not path.is_symlink():
        return
    mode = path.lstat().st_mode
    if not stat.S_ISSOCK(mode):
        raise RuntimeError(f"refusing to remove non-socket IPC path: {path}")
    path.unlink()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    if hasattr(value, "item"):
        return value.item()
    return str(value)


class HardwareGateway:
    def __init__(self, config: BridgeConfig):
        self.config = config
        self.adapter = ExistingStackAdapter(
            config.existing_stack,
            max_translation_step_m=config.safety.max_translation_step_m,
            max_rotation_step_rad=config.safety.max_rotation_step_rad,
        )
        self.socket: socket.socket | None = None
        self.pedal_reader: Any | None = None
        self.gateway_instance_id = uuid.uuid4().hex
        self.active = False
        self.active_session_id = ""
        self.active_epoch = -1
        self.last_sequence = -1
        self.last_action_receive_ns = 0
        self.fault_reason = ""
        self._next_observation_ns = 0

    def setup(self) -> None:
        deadman = self.config.deadman
        if deadman.gateway_pedal_required:
            if not deadman.gateway_pedal_device:
                raise RuntimeError(
                    "gateway pedal is required but gateway_pedal_device is empty"
                )
            self.pedal_reader = self.adapter.make_pedal_reader(
                deadman.gateway_pedal_device,
                deadman.gateway_pedal_keys,
                deadman.gateway_pedal_grab,
            )
            self.pedal_reader.start()
            if not bool(getattr(self.pedal_reader, "available", False)):
                raise RuntimeError(
                    f"required pedal is unavailable: {deadman.gateway_pedal_device}"
                )

        self.adapter.connect()
        _safe_remove_socket(self.config.ipc.gateway_socket)
        server = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        server.bind(self.config.ipc.gateway_socket)
        server.settimeout(0.02)
        self.socket = server
        LOGGER.warning(
            "Hardware gateway armed for explicit packets; startup motion was %s",
            "disabled" if self.config.existing_stack.disable_startup_motion else "kept",
        )

    def _pedal_pressed(self) -> bool:
        if not self.config.deadman.gateway_pedal_required:
            return True
        if self.pedal_reader is None:
            return False
        return bool(self.pedal_reader.is_pressed())

    def _send(self, packet: dict[str, Any]) -> None:
        payload = encode_packet(_json_safe(packet))
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as client:
                client.sendto(payload, self.config.ipc.ros_socket)
        except (FileNotFoundError, ConnectionRefusedError):
            LOGGER.debug("ROS telemetry socket is not available")

    def _ack(
        self,
        incoming: dict[str, Any],
        ack_kind: str,
        ok: bool,
        reason: str = "",
    ) -> None:
        self._send(
            {
                "kind": "ack",
                "ack_kind": ack_kind,
                "ok": bool(ok),
                "reason": str(reason),
                "session_id": str(incoming.get("session_id", "")),
                "sequence": int(incoming.get("sequence", -1)),
                "epoch": int(incoming.get("epoch", -1)),
                "proposal_id": incoming.get("proposal_id"),
                "gateway_instance_id": self.gateway_instance_id,
                "gateway_monotonic_ns": time.monotonic_ns(),
            }
        )

    def _publish_status(self, reason: str = "") -> None:
        self._send(
            {
                "kind": "gateway_status",
                "active": self.active,
                "fault_reason": self.fault_reason,
                "reason": reason,
                "session_id": self.active_session_id,
                "epoch": self.active_epoch,
                "pedal_pressed": self._pedal_pressed(),
                "gateway_instance_id": self.gateway_instance_id,
                "gateway_monotonic_ns": time.monotonic_ns(),
            }
        )

    def _hold(self, reason: str) -> None:
        try:
            self.adapter.hold_current_pose()
        except Exception as exc:
            self.fault_reason = f"hold_current_pose failed: {exc}"
            self.adapter.emergency_stop()
            self.active = False
            self._publish_status(self.fault_reason)
            raise
        self.active = False
        self.last_action_receive_ns = 0
        self._publish_status(reason)

    def _fail_active(self, reason: str) -> None:
        self.fault_reason = str(reason)
        try:
            self._hold(reason)
        except Exception:
            pass
        self.adapter.emergency_stop()
        self.active = False
        self._publish_status(reason)

    def _validate_common(self, packet: dict[str, Any]) -> tuple[str, int, int]:
        session_id = str(packet.get("session_id", "")).strip()
        if not session_id:
            raise ValueError("packet session_id is empty")
        sequence = int(packet.get("sequence", -1))
        epoch = int(packet.get("epoch", -1))
        if sequence < 0 or epoch < 0:
            raise ValueError("packet sequence and epoch must be non-negative")
        sent_ns = int(packet.get("sent_monotonic_ns", 0))
        age_s = (time.monotonic_ns() - sent_ns) * 1e-9
        if sent_ns <= 0 or age_s < 0 or age_s > self.config.safety.gateway_watchdog_s:
            raise ValueError(f"packet age is invalid or stale: {age_s:.4f}s")
        return session_id, sequence, epoch

    def handle_packet(self, packet: dict[str, Any]) -> None:
        kind = str(packet.get("kind", ""))
        if kind == "hold":
            try:
                self._hold(str(packet.get("reason", "remote hold")))
                self._ack(packet, "hold", True)
            except Exception as exc:
                self._ack(packet, "hold", False, str(exc))
            return
        if self.fault_reason:
            self._ack(packet, kind or "unknown", False, self.fault_reason)
            return

        try:
            session_id, sequence, epoch = self._validate_common(packet)
            if not bool(packet.get("deadman", False)):
                raise ValueError("packet deadman is false")
            if not self._pedal_pressed():
                raise ValueError("required local foot pedal is not pressed")
            if kind == "arm":
                self.adapter.hold_current_pose()
                self.active_session_id = session_id
                self.active_epoch = epoch
                self.last_sequence = sequence
                self.last_action_receive_ns = time.monotonic_ns()
                self.active = True
                self._ack(packet, "arm", True)
                self._publish_status("armed at fresh measured robot pose")
                return
            if kind != "action":
                raise ValueError(f"unsupported packet kind: {kind!r}")
            if not self.active:
                raise ValueError("gateway is not armed")
            if session_id != self.active_session_id or epoch != self.active_epoch:
                raise ValueError("session or epoch differs from active arm")
            if sequence <= self.last_sequence:
                raise ValueError("packet sequence is not strictly increasing")
            action = validate_action(
                packet.get("action"),
                max_translation_step_m=self.config.safety.max_translation_step_m,
                max_rotation_step_rad=self.config.safety.max_rotation_step_rad,
            )
            self.adapter.send_action(action)
            self.last_sequence = sequence
            self.last_action_receive_ns = time.monotonic_ns()
            self._ack(packet, "action", True)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            LOGGER.error("Rejected %s packet: %s", kind or "unknown", reason)
            if self.active:
                self._fail_active(reason)
            self._ack(packet, kind or "unknown", False, reason)

    def _watchdog(self) -> None:
        if not self.active:
            return
        if not self._pedal_pressed():
            self._fail_active("required local foot pedal was released")
            return
        age_s = (time.monotonic_ns() - self.last_action_receive_ns) * 1e-9
        if age_s > self.config.safety.gateway_watchdog_s:
            self._fail_active(
                f"gateway packet watchdog expired after {age_s:.4f}s"
            )

    def _publish_observation(self) -> None:
        now_ns = time.monotonic_ns()
        if now_ns < self._next_observation_ns:
            return
        self._next_observation_ns = now_ns + 33_333_333
        try:
            observation = self.adapter.fresh_observation()
        except Exception as exc:
            if self.active:
                self._fail_active(f"fresh robot observation failed: {exc}")
            return
        self._send(
            {
                "kind": "observation",
                "session_id": self.active_session_id,
                "epoch": self.active_epoch,
                "observation": observation,
                "gateway_instance_id": self.gateway_instance_id,
            }
        )

    def run(self) -> None:
        if self.socket is None:
            raise RuntimeError("gateway setup was not called")
        self._publish_status("gateway started")
        while True:
            try:
                data = self.socket.recv(self.config.ipc.max_packet_bytes)
            except socket.timeout:
                data = None
            if data:
                try:
                    packet = decode_packet(
                        data, max_packet_bytes=self.config.ipc.max_packet_bytes
                    )
                    self.handle_packet(packet)
                except Exception as exc:
                    LOGGER.error("Invalid IPC packet: %s", exc)
                    if self.active:
                        self._fail_active(f"invalid IPC packet: {exc}")
            self._watchdog()
            self._publish_observation()

    def close(self) -> None:
        try:
            if self.active:
                self._hold("gateway shutdown")
        except Exception:
            self.adapter.emergency_stop()
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        try:
            _safe_remove_socket(self.config.ipc.gateway_socket)
        except Exception:
            pass
        if self.pedal_reader is not None:
            self.pedal_reader.stop()
            self.pedal_reader = None
        try:
            self.adapter.disconnect()
        except Exception:
            self.adapter.emergency_stop()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--enable-hardware-command",
        action="store_true",
        help="required in addition to command_enabled: true in the new YAML",
    )
    parser.add_argument(
        "--confirm-robot-area-clear",
        default="",
        metavar=CONFIRMATION_TEXT,
        help="explicit physical-area confirmation token",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_parser().parse_args(argv)
    config = load_config(args.config)
    if not config.command_enabled:
        raise SystemExit(
            "Hardware gateway refused: command_enabled is false in the new config"
        )
    if not args.enable_hardware_command:
        raise SystemExit(
            "Hardware gateway refused: --enable-hardware-command was not supplied"
        )
    if args.confirm_robot_area_clear != CONFIRMATION_TEXT:
        raise SystemExit(
            "Hardware gateway refused: pass "
            f"--confirm-robot-area-clear {CONFIRMATION_TEXT} after physical checks"
        )

    gateway = HardwareGateway(config)
    try:
        gateway.setup()
        gateway.run()
    except KeyboardInterrupt:
        LOGGER.info("Gateway interrupted")
    finally:
        gateway.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

