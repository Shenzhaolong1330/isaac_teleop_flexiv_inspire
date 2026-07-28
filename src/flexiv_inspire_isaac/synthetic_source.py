"""Deterministic ROS 2 Isaac-topic fixture for shadow-mode integration tests."""

from __future__ import annotations

import argparse
import math
import sys
import time

import msgpack
import rclpy
from geometry_msgs.msg import Pose, PoseArray, TransformStamped
from rclpy.node import Node
from std_msgs.msg import ByteMultiArray
from tf2_msgs.msg import TFMessage


class SyntheticIsaacSource(Node):
    def __init__(self, *, deadman: bool, duration_s: float):
        super().__init__("synthetic_isaac_source")
        self.deadman = bool(deadman)
        self.duration_s = float(duration_s)
        self.start_ns = time.monotonic_ns()
        self.ee_publisher = self.create_publisher(
            PoseArray, "/xr_teleop/ee_poses", 10
        )
        self.controller_publisher = self.create_publisher(
            ByteMultiArray, "/xr_teleop/controller_data", 10
        )
        self.tf_publisher = self.create_publisher(TFMessage, "/tf", 10)
        self.create_timer(1.0 / 60.0, self.tick)

    def tick(self) -> None:
        elapsed = (time.monotonic_ns() - self.start_ns) * 1e-9
        if elapsed >= self.duration_s:
            rclpy.shutdown()
            return
        stamp = self.get_clock().now().to_msg()
        offset = 0.01 * math.sin(2.0 * math.pi * 0.25 * elapsed)

        poses = PoseArray()
        poses.header.stamp = stamp
        poses.header.frame_id = "world"
        for x, y in ((0.35 + offset, 0.25), (0.35 - offset, -0.25)):
            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = 1.0
            pose.orientation.w = 1.0
            poses.poses.append(pose)
        self.ee_publisher.publish(poses)

        transforms = TFMessage()
        for child, pose in zip(("left_wrist", "right_wrist"), poses.poses):
            transform = TransformStamped()
            transform.header.stamp = stamp
            transform.header.frame_id = "world"
            transform.child_frame_id = child
            transform.transform.translation.x = pose.position.x
            transform.transform.translation.y = pose.position.y
            transform.transform.translation.z = pose.position.z
            transform.transform.rotation.w = 1.0
            transforms.transforms.append(transform)
        self.tf_publisher.publish(transforms)

        squeeze = 0.9 if self.deadman else 0.0
        payload = msgpack.packb(
            {
                "timestamp": time.time_ns(),
                "left_squeeze_value": squeeze,
                "right_squeeze_value": squeeze,
                "left_trigger_value": 0.0,
                "right_trigger_value": 0.0,
                "left_is_active": True,
                "right_is_active": True,
            }
        )
        controller = ByteMultiArray()
        controller.data = tuple(bytes([value]) for value in payload)
        self.controller_publisher.publish(controller)


def main(argv: list[str] | None = None) -> int:
    ros_args = sys.argv if argv is None else [sys.argv[0], *argv]
    clean = rclpy.utilities.remove_ros_args(args=ros_args)[1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deadman", action="store_true")
    parser.add_argument("--duration", type=float, default=5.0)
    args = parser.parse_args(clean)
    rclpy.init(args=ros_args)
    node = SyntheticIsaacSource(
        deadman=args.deadman,
        duration_s=max(0.1, args.duration),
    )
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

