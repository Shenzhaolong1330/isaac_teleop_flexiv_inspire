"""ROS routing for the left/right edge events of the three-key pedal."""

from __future__ import annotations

from pathlib import Path

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String

from flexiv_inspire_control.foot_pedal import (
    FootPedalMonitor,
    KEY_DOWN,
    KEY_LEFT,
    KEY_RIGHT,
    KEY_SPACE,
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
        self._deadman_publisher = self.create_publisher(
            Bool, "/teleop/deadman", QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        )
        self._enabled = False
        configured_enable = int(self.get_parameter("enable_key_code").value)
        self._pedal = FootPedalMonitor(
            Path(str(self.get_parameter("foot_pedal").value)),
            self._on_enable,
            self._on_event,
            self._on_status,
            # The current site uses Input Remapper (Space=57).  Accept the
            # physical pedal's native Down=108 as well so a remapper profile
            # change cannot leave the arm clutch permanently released.
            enable_key_codes=(configured_enable, KEY_SPACE, KEY_DOWN),
            grab=True,
        )
        self._on_enable(False)
        # Teleop intentionally treats a stale deadman as released. Publish a
        # small heartbeat while held instead of relying only on edge events.
        self.create_timer(0.05, self._publish_deadman)
        self._pedal.start()

    def _on_status(self, status: str) -> None:
        if status.startswith("connected:"):
            self.get_logger().info(
                f"踏板已连接并独占读取: {status.removeprefix('connected:')}"
            )
        else:
            self.get_logger().error(
                f"踏板不可用，将自动重连: {status.removeprefix('error:')}",
                throttle_duration_sec=2.0,
            )

    def _on_enable(self, enabled: bool) -> None:
        self._enabled = bool(enabled)
        self._publish_deadman()
        self.get_logger().info(
            "中踏板：已踩下，正在等待 Quest 双腕追踪和控制端确认"
            if enabled
            else "中踏板：已松开，机械臂运动已停止"
        )

    def _publish_deadman(self) -> None:
        message = Bool()
        message.data = self._enabled
        try:
            self._deadman_publisher.publish(message)
        except Exception:
            # Process teardown may race with the evdev worker's final
            # release edge. The CLI sends an explicit release before it
            # terminates a standalone replay router.
            return

    def _on_event(self, event: PedalEvent) -> None:
        if not event.pressed:
            return
        if event.key_code == int(self.get_parameter("rerecord_key_code").value):
            command = "rerecord"
            label = "左踏板：丢弃本条并重新录制"
        elif event.key_code == int(self.get_parameter("record_toggle_key_code").value):
            command = "stop"
            label = "右踏板：保存本条并进入下一条"
        else:
            return
        message = String()
        message.data = command
        self._publisher.publish(message)
        self.get_logger().info(label)

    def destroy_node(self) -> bool:
        # Publish the release while the ROS context is still alive; otherwise
        # a killed foreground replay can leave the control bridge believing
        # that the clutch remains pressed.
        self._enabled = False
        self._publish_deadman()
        self._pedal.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PedalRouter()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
