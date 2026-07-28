"""Production gateway entry point used by the launch script.

Heartbeats refresh the watchdog but deliberately do not advance the ordered
action sequence.  This prevents a heartbeat and action sent from separate Unix
datagram sockets from creating a false out-of-order action fault.
"""

from __future__ import annotations

import logging
import time

from .config import load_config
from .gateway import CONFIRMATION_TEXT, HardwareGateway, LOGGER, _build_parser


class SafeHeartbeatHardwareGateway(HardwareGateway):
    def handle_packet(self, packet):
        if str(packet.get("kind", "")) != "heartbeat":
            return super().handle_packet(packet)
        if self.fault_reason:
            return
        try:
            session_id, _sequence, epoch = self._validate_common(packet)
            if not bool(packet.get("deadman", False)):
                raise ValueError("heartbeat deadman is false")
            if not self._pedal_pressed():
                raise ValueError("required local foot pedal is not pressed")
            if not self.active:
                raise ValueError("gateway is not armed")
            if session_id != self.active_session_id or epoch != self.active_epoch:
                raise ValueError("session or epoch differs from active arm")
            self.last_action_receive_ns = time.monotonic_ns()
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            LOGGER.error("Rejected heartbeat packet: %s", reason)
            if self.active:
                self._fail_active(reason)


def main(argv=None):
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
    gateway = SafeHeartbeatHardwareGateway(config)
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

