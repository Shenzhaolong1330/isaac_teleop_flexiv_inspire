"""ROS routing for the left/right edge events of the three-key pedal."""

from __future__ import annotations

from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from flexiv_inspire_control.foot_pedal import (
    FootPedalMonitor,
    KEY_DOWN,
    KEY_LEFT,
    KEY_RIGHT,
    PedalEvent,
)


class PedalRouter(Node):
    def __init__(self) -> None:
        super().__init__("flexiv_inspire_pedal_router")
        self.declare_parameter(
            "foot_pedal", "/dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd"
        )
        self.declare_parameter("rerecord_key_code", KEY_LEFT)
        self.declare_parameter("enable_key_code", KEY_DOWN)
        self.declare_parameter("record_toggle_key_code", KEY_RIGHT)
        self._publisher = self.create_publisher(
            String, "/episode/control", QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        )
        self._pedal = FootPedalMonitor(
            Path(str(self.get_parameter("foot_pedal").value)),
            lambda enabled: None,
            self._on_event,
            enable_key_code=int(self.get_parameter("enable_key_code").value),
        )
        self._pedal.start()

    def _on_event(self, event: PedalEvent) -> None:
        if not event.pressed:
            return
        if event.key_code == int(self.get_parameter("rerecord_key_code").value):
            command = "rerecord"
        elif event.key_code == int(self.get_parameter("record_toggle_key_code").value):
            command = "stop"
        else:
            return
        message = String()
        message.data = command
        self._publisher.publish(message)

    def destroy_node(self) -> bool:
        self._pedal.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PedalRouter()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
