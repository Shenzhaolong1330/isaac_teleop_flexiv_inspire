"""Publish Quest controllers and raw OpenXR/MANUS hands without robot retargeting.

This is the source node for the Flexiv + Inspire stack. It deliberately emits
the 25 OpenXR joints per hand and does not instantiate a Sharpa (or any other
robot-hand) retargeter.
"""

from __future__ import annotations

import time

import msgpack
import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseArray, TransformStamped
from rclpy.node import Node
from std_msgs.msg import ByteMultiArray
from tf2_ros import TransformBroadcaster

from isaacteleop.cloudxr import CloudXRLauncher
from isaacteleop.retargeting_engine.deviceio_source_nodes import (
    ControllersSource,
    HandsSource,
)
from isaacteleop.retargeting_engine.interface import OutputCombiner
from isaacteleop.retargeting_engine.tensor_types.indices import (
    ControllerInputIndex,
    HandInputIndex,
    HandJointIndex,
)
from isaacteleop.teleop_session_manager import (
    SessionMode,
    TeleopSession,
    TeleopSessionConfig,
)
from isaac_teleop_core.octet_sequence import encode_octet_sequence

from .openxr_errors import is_retryable_openxr_session_error


def _pose(position, orientation=(0.0, 0.0, 0.0, 1.0)) -> Pose:
    result = Pose()
    result.position.x, result.position.y, result.position.z = (
        float(value) for value in position
    )
    (
        result.orientation.x,
        result.orientation.y,
        result.orientation.z,
        result.orientation.w,
    ) = (float(value) for value in orientation)
    return result


def _valid(group, index: int, nested: int | None = None) -> bool:
    if group.is_none:
        return False
    value = group[index]
    if nested is not None:
        value = value[nested]
    return bool(value)


def _controller_value(group, index: int, default):
    if group.is_none:
        return default
    value = group[index]
    array = np.asarray(value)
    if array.ndim:
        return [float(item) for item in array]
    return float(value)


def _controller_click(group, index: int) -> bool:
    """Normalize OpenXR's scalar click action to a transport boolean."""
    value = float(_controller_value(group, index, 0.0))
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError("controller click value must be finite and in [0,1]")
    return value >= 0.5


class XrRawRosSource(Node):
    """Isaac DeviceIO to ROS adapter with no robot-hand model dependency."""

    def __init__(self) -> None:
        super().__init__("xr_raw_ros_source")
        defaults = {
            "rate_hz": 60.0,
            "world_frame": "world",
            "left_wrist_frame": "left_wrist",
            "right_wrist_frame": "right_wrist",
            "cloudxr_install_dir": "~/.cloudxr",
            "cloudxr_env_config": "",
            "cloudxr_accept_eula": False,
            "cloudxr_setup_oob": False,
            "cloudxr_usb_local": False,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        rate_hz = float(self.get_parameter("rate_hz").value)
        if not np.isfinite(rate_hz) or rate_hz <= 0.0:
            raise ValueError("rate_hz must be finite and > 0")
        if bool(self.get_parameter("cloudxr_usb_local").value) and not bool(
            self.get_parameter("cloudxr_setup_oob").value
        ):
            raise ValueError("cloudxr_usb_local requires cloudxr_setup_oob")
        self._sleep_s = 1.0 / rate_hz
        self._world = str(self.get_parameter("world_frame").value).strip()
        self._left_wrist = str(
            self.get_parameter("left_wrist_frame").value
        ).strip()
        self._right_wrist = str(
            self.get_parameter("right_wrist_frame").value
        ).strip()
        if (
            not self._world
            or not self._left_wrist
            or not self._right_wrist
            or len({self._world, self._left_wrist, self._right_wrist}) != 3
        ):
            raise ValueError("world and wrist frame names must be non-empty/distinct")

        controllers = ControllersSource(name="controllers")
        hands = HandsSource(name="hands")
        pipeline = OutputCombiner(
            {
                "controller_left": controllers.output(ControllersSource.LEFT),
                "controller_right": controllers.output(ControllersSource.RIGHT),
                "hand_left": hands.output(HandsSource.LEFT),
                "hand_right": hands.output(HandsSource.RIGHT),
            }
        )
        self._session_config = TeleopSessionConfig(
            app_name="FlexivInspireRawSource",
            pipeline=pipeline,
            mode=SessionMode.LIVE,
        )
        self._ee_pub = self.create_publisher(PoseArray, "xr_teleop/ee_poses", 10)
        self._hand_pub = self.create_publisher(PoseArray, "xr_teleop/hand", 10)
        self._controller_pub = self.create_publisher(
            ByteMultiArray, "xr_teleop/controller_data", 10
        )
        self._tf = TransformBroadcaster(self)

    def _publish_controllers(self, result: dict, now) -> None:
        message = PoseArray()
        message.header.stamp = now
        message.header.frame_id = self._world
        transforms = []
        for side, frame in (
            ("left", self._left_wrist),
            ("right", self._right_wrist),
        ):
            controller = result[f"controller_{side}"]
            is_valid = _valid(controller, ControllerInputIndex.AIM_IS_VALID)
            pose = (
                _pose(
                    controller[ControllerInputIndex.AIM_POSITION],
                    controller[ControllerInputIndex.AIM_ORIENTATION],
                )
                if is_valid
                else _pose((0.0, 0.0, 0.0))
            )
            message.poses.append(pose)
            if is_valid:
                transform = TransformStamped()
                transform.header = message.header
                transform.child_frame_id = frame
                transform.transform.translation.x = pose.position.x
                transform.transform.translation.y = pose.position.y
                transform.transform.translation.z = pose.position.z
                transform.transform.rotation = pose.orientation
                transforms.append(transform)
        self._ee_pub.publish(message)
        if transforms:
            self._tf.sendTransform(transforms)

        left = result["controller_left"]
        right = result["controller_right"]
        payload = {
            "timestamp": time.time_ns(),
            "left_squeeze_value": _controller_value(
                left, ControllerInputIndex.SQUEEZE_VALUE, 0.0
            ),
            "right_squeeze_value": _controller_value(
                right, ControllerInputIndex.SQUEEZE_VALUE, 0.0
            ),
            "left_primary_click": _controller_click(
                left, ControllerInputIndex.PRIMARY_CLICK
            ),
            "right_primary_click": _controller_click(
                right, ControllerInputIndex.PRIMARY_CLICK
            ),
            "left_is_active": not left.is_none,
            "right_is_active": not right.is_none,
        }
        controller_message = ByteMultiArray()
        controller_message.data = encode_octet_sequence(
            msgpack.packb(payload, use_bin_type=True)
        )
        self._controller_pub.publish(controller_message)

    def _publish_hands(self, result: dict, now) -> None:
        message = PoseArray()
        message.header.stamp = now
        message.header.frame_id = self._world
        for side in ("left", "right"):
            hand = result[f"hand_{side}"]
            for joint in range(
                HandJointIndex.WRIST, HandJointIndex.LITTLE_TIP + 1
            ):
                if _valid(hand, HandInputIndex.JOINT_VALID, joint):
                    message.poses.append(
                        _pose(
                            hand[HandInputIndex.JOINT_POSITIONS][joint],
                            hand[HandInputIndex.JOINT_ORIENTATIONS][joint],
                        )
                    )
                else:
                    message.poses.append(_pose((0.0, 0.0, 0.0)))
        if len(message.poses) != 50:
            raise RuntimeError(
                f"raw hand transport must contain 50 poses, got {len(message.poses)}"
            )
        self._hand_pub.publish(message)

    def _run_sessions(self, launcher: CloudXRLauncher) -> int:
        while rclpy.ok():
            launcher.health_check()
            try:
                with TeleopSession(self._session_config) as session:
                    self.get_logger().info(
                        "raw Quest + MANUS session started (no hand URDF)"
                    )
                    while rclpy.ok():
                        launcher.health_check()
                        result = session.step()
                        rclpy.spin_once(self, timeout_sec=0.0)
                        now = self.get_clock().now().to_msg()
                        self._publish_controllers(result, now)
                        self._publish_hands(result, now)
                        time.sleep(self._sleep_s)
            except RuntimeError as exc:
                if not is_retryable_openxr_session_error(exc):
                    raise
                self.get_logger().warning(
                    f"OpenXR session not ready ({exc}); retrying in 2 seconds"
                )
                time.sleep(2.0)
        return 0

    def run(self) -> int:
        env_config = str(
            self.get_parameter("cloudxr_env_config").value
        ).strip() or None
        with CloudXRLauncher(
            install_dir=str(self.get_parameter("cloudxr_install_dir").value),
            env_config=env_config,
            accept_eula=bool(
                self.get_parameter("cloudxr_accept_eula").value
            ),
            setup_oob=bool(self.get_parameter("cloudxr_setup_oob").value),
            usb_local=bool(self.get_parameter("cloudxr_usb_local").value),
        ) as launcher:
            self.get_logger().info(
                "CloudXR runtime/WSS started; launch the MANUS plugin using "
                "~/.cloudxr/run/cloudxr.env"
            )
            return self._run_sessions(launcher)


def main(args=None) -> int:
    rclpy.init(args=args)
    node = None
    try:
        node = XrRawRosSource()
        return node.run()
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
