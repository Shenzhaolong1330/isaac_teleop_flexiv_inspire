"""Publish MANUS SDK ergonomics received from the Isaac plugin over loopback."""

from __future__ import annotations

import argparse
import math
import socket
from dataclasses import dataclass
from typing import Final

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState

PROTOCOL: Final = "MANUS_ERGONOMICS_V1"
DEFAULT_HOST: Final = "127.0.0.1"
DEFAULT_PORT: Final = 15053
FIELD_NAMES: Final = (
    "ThumbMCPSpread",
    "ThumbMCPStretch",
    "ThumbPIPStretch",
    "ThumbDIPStretch",
    "IndexMCPSpread",
    "IndexMCPStretch",
    "IndexPIPStretch",
    "IndexDIPStretch",
    "MiddleMCPSpread",
    "MiddleMCPStretch",
    "MiddlePIPStretch",
    "MiddleDIPStretch",
    "RingMCPSpread",
    "RingMCPStretch",
    "RingPIPStretch",
    "RingDIPStretch",
    "PinkyMCPSpread",
    "PinkyMCPStretch",
    "PinkyPIPStretch",
    "PinkyDIPStretch",
)


@dataclass(frozen=True)
class ErgonomicsPacket:
    side: str
    glove_id: int
    source_time: int
    values_rad: tuple[float, ...]


def parse_packet(payload: bytes) -> ErgonomicsPacket:
    """Validate one compact plugin datagram and convert degrees to radians."""

    try:
        fields = payload.decode("ascii").strip().split(",")
    except UnicodeDecodeError as exc:
        raise ValueError("MANUS ergonomics packet is not ASCII") from exc
    expected = 4 + len(FIELD_NAMES)
    if len(fields) != expected or fields[0] != PROTOCOL:
        raise ValueError(
            f"invalid MANUS ergonomics packet schema/length: {len(fields)}"
        )
    side = fields[1]
    if side not in {"left", "right"}:
        raise ValueError(f"invalid MANUS side: {side}")
    try:
        glove_id = int(fields[2])
        source_time = int(fields[3])
        values_deg = tuple(float(value) for value in fields[4:])
    except ValueError as exc:
        raise ValueError("MANUS ergonomics packet contains invalid numbers") from exc
    if (
        glove_id < 0
        or source_time < 0
        or not all(math.isfinite(value) for value in values_deg)
    ):
        raise ValueError("MANUS ergonomics packet contains non-finite data")
    return ErgonomicsPacket(
        side=side,
        glove_id=glove_id,
        source_time=source_time,
        values_rad=tuple(math.radians(value) for value in values_deg),
    )


class ManusErgonomicsSource(Node):
    def __init__(self) -> None:
        super().__init__("manus_ergonomics_source")
        defaults = {
            "udp_host": DEFAULT_HOST,
            "udp_port": DEFAULT_PORT,
            "left_topic": "/manus/left/ergonomics",
            "right_topic": "/manus/right/ergonomics",
            "poll_rate_hz": 500.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        host = str(self.get_parameter("udp_host").value)
        port = int(self.get_parameter("udp_port").value)
        if host not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("MANUS ergonomics UDP must remain on loopback")
        if not 1 <= port <= 65535:
            raise ValueError("MANUS ergonomics UDP port is invalid")
        rate = float(self.get_parameter("poll_rate_hz").value)
        if not 50.0 <= rate <= 2000.0:
            raise ValueError("MANUS ergonomics poll_rate_hz must be in [50,2000]")

        # Do not shadow rclpy.Node._publishers; Node.destroy_node owns it.
        self._ergonomics_publishers = {
            side: self.create_publisher(
                JointState,
                str(self.get_parameter(f"{side}_topic").value),
                qos_profile_sensor_data,
            )
            for side in ("left", "right")
        }
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # A second source must fail at bind instead of silently sharing glove
        # frames with a stale launcher process.
        self._socket.bind((host, port))
        self._socket.setblocking(False)
        self._received = 0
        self.create_timer(1.0 / rate, self._poll)
        self.get_logger().info(f"MANUS Ergonomics UDP listening on {host}:{port}")

    def _poll(self) -> None:
        # Drain the socket so an old glove frame can never queue behind live input.
        for _ in range(64):
            try:
                payload, _peer = self._socket.recvfrom(4096)
            except BlockingIOError:
                return
            try:
                packet = parse_packet(payload)
            except ValueError as exc:
                self.get_logger().error(
                    f"invalid MANUS Ergonomics datagram: {exc}",
                    throttle_duration_sec=1.0,
                )
                continue
            message = JointState()
            message.header.stamp = self.get_clock().now().to_msg()
            message.header.frame_id = f"manus_glove_{packet.glove_id}"
            message.name = list(FIELD_NAMES)
            message.position = list(packet.values_rad)
            self._ergonomics_publishers[packet.side].publish(message)
            self._received += 1
            if self._received == 1:
                self.get_logger().info(
                    "MANUS_ERGONOMICS_READY: first validated SDK frame published"
                )

    def destroy_node(self) -> bool:
        self._socket.close()
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    _known, ros_args = parser.parse_known_args(args)
    rclpy.init(args=ros_args)
    node = ManusErgonomicsSource()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
