"""Independent Isaac Teleop/Quest/MANUS to canonical BimanualCommand adapter."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any
import uuid

import msgpack
import numpy as np
import rclpy
from geometry_msgs.msg import PoseArray
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Bool, ByteMultiArray, Empty, UInt64
from tf2_msgs.msg import TFMessage
import yaml

from flexiv_inspire_interfaces.msg import (
    BimanualCommand,
    BimanualCommandPoint,
)
from isaac_teleop_core.command import ROTATION_ORDER

from .manus_pose import (
    FEATURE_NAMES,
    pose_message_values,
    split_bimanual_pose_array,
)
from .teleop_mapping import QuestSE3Mapper, TrackedPose, TrackingSample


ACTUATORS = (
    "little",
    "ring",
    "middle",
    "index",
    "thumb_bend",
    "thumb_rotate",
)


def _frame(value: str) -> str:
    return str(value).strip().lstrip("/")


def _bytes(message: ByteMultiArray) -> bytes:
    return bytes(int(item) & 0xFF for item in message.data)


def _validated_squeezes(value: object) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ValueError("controller_data is not a mapping")
    result = {
        "left": float(value.get("left_squeeze_value", 0.0)),
        "right": float(value.get("right_squeeze_value", 0.0)),
    }
    if any(
        not np.isfinite(item) or item < 0.0 or item > 1.0
        for item in result.values()
    ):
        raise ValueError("Quest squeeze values must be finite and in [0,1]")
    return result


@dataclass(frozen=True)
class _Channel:
    sources: tuple[str, ...]
    weights: np.ndarray
    bias: float
    source_open: float
    source_closed: float
    output_open: float
    output_closed: float

    def evaluate(self, joints: dict[str, float]) -> float:
        try:
            value = self.bias + sum(
                float(weight) * float(joints[name])
                for name, weight in zip(self.sources, self.weights, strict=True)
            )
        except KeyError as exc:
            raise ValueError(f"MANUS joint absent: {exc}") from exc
        alpha = (value - self.source_open) / (
            self.source_closed - self.source_open
        )
        alpha = min(1.0, max(0.0, alpha))
        return float(
            self.output_open + alpha * (self.output_closed - self.output_open)
        )


class _Retarget:
    def __init__(self, path: str) -> None:
        self.channels: dict[str, dict[str, _Channel]] = {}
        if not path:
            return
        document = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8"))
        if (
            not isinstance(document, dict)
            or document.get("schema_version") != 1
            or not document.get("calibrated", False)
            or document.get("source_format")
            != "ISAAC_XR_HAND_POSEARRAY_LEFT25_RIGHT25"
            or int(document.get("transport_joint_count_per_side", 0)) != 25
        ):
            raise ValueError(
                "MANUS calibration must be calibrated schema v1 for "
                "Isaac left25+right25 PoseArray"
            )
        for side in ("left", "right"):
            self.channels[side] = {}
            for actuator in ACTUATORS:
                raw = document["sides"][side][actuator]
                sources = tuple(str(value) for value in raw["sources"])
                weights = np.asarray(raw["weights"], dtype=np.float64)
                if not sources or weights.shape != (len(sources),):
                    raise ValueError(f"invalid {side}.{actuator} calibration")
                source_open = float(raw["source_open"])
                source_closed = float(raw["source_closed"])
                if abs(source_closed - source_open) < 1e-9:
                    raise ValueError(f"degenerate {side}.{actuator} calibration")
                unknown = set(sources) - FEATURE_NAMES
                if unknown:
                    raise ValueError(
                        f"unsupported {side}.{actuator} MANUS features: "
                        f"{sorted(unknown)}"
                    )
                self.channels[side][actuator] = _Channel(
                    sources,
                    weights,
                    float(raw.get("bias", 0.0)),
                    source_open,
                    source_closed,
                    float(raw.get("output_open", 1000.0)),
                    float(raw.get("output_closed", 0.0)),
                )
            required = {
                source
                for channel in self.channels[side].values()
                for source in channel.sources
            }
            missing = required - FEATURE_NAMES
            if missing:
                raise ValueError(
                    f"{side} calibration requires unavailable features: "
                    f"{sorted(missing)}"
                )

    @property
    def calibrated(self) -> bool:
        return set(self.channels) == {"left", "right"}

    def apply_features(
        self, side: str, features: dict[str, float]
    ) -> np.ndarray:
        if set(features) != FEATURE_NAMES:
            missing = FEATURE_NAMES - set(features)
            extra = set(features) - FEATURE_NAMES
            raise ValueError(
                f"MANUS feature set mismatch: missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )
        result = np.array(
            [
                self.channels[side][actuator].evaluate(features)
                for actuator in ACTUATORS
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(result)) or np.any(result < 0) or np.any(result > 1000):
            raise ValueError("retargeted hand command outside [0,1000]")
        return result


class TeleopInput(Node):
    def __init__(self) -> None:
        super().__init__("isaac_teleop_input")
        defaults = {
            "command_enabled": False,
            "session_id": "",
            "ee_topic": "/xr_teleop/ee_poses",
            "controller_topic": "/xr_teleop/controller_data",
            "tf_topic": "/tf",
            "external_deadman_topic": "/teleop/deadman",
            "deadman_source": "quest_squeeze_both",
            "squeeze_threshold": 0.65,
            "world_frame": "world",
            "left_wrist_frame": "left_wrist",
            "right_wrist_frame": "right_wrist",
            "left_pose_index": 0,
            "right_pose_index": 1,
            "require_wrist_tf": True,
            "max_tf_age_s": 0.12,
            "control_rate_hz": 60.0,
            "ttl_s": 0.10,
            "translation_gain": 1.0,
            "axis_rotation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "hand_pose_topic": "/xr_teleop/hand",
            "manus_calibration": "",
            "max_manus_age_s": 0.10,
            "home_button_key": "right_primary_click",
            "home_topic": "/control/home_request",
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self._enabled = bool(self.get_parameter("command_enabled").value)
        configured_session = str(self.get_parameter("session_id").value).strip()
        self._session = configured_session or f"teleop-{uuid.uuid4()}"
        axis = np.asarray(
            self.get_parameter("axis_rotation").value, dtype=np.float64
        ).reshape(3, 3)
        self._mapper = QuestSE3Mapper(
            world_frame=str(self.get_parameter("world_frame").value),
            axis_rotation=axis,
            translation_gain=float(self.get_parameter("translation_gain").value),
        )
        self._retarget = _Retarget(
            str(self.get_parameter("manus_calibration").value)
        )
        self._pose_sequence = 0
        self._sample: TrackingSample | None = None
        self._tf_received: dict[str, int] = {}
        self._controller_received = 0
        self._squeeze = {"left": 0.0, "right": 0.0}
        self._external_deadman = False
        self._external_deadman_received = 0
        self._hands: dict[str, np.ndarray | None] = {"left": None, "right": None}
        self._hands_received = {"left": 0, "right": 0}
        self._command_sequence = 0
        self._home_button_down = False

        self.create_subscription(
            PoseArray,
            str(self.get_parameter("ee_topic").value),
            self._on_pose,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            ByteMultiArray,
            str(self.get_parameter("controller_topic").value),
            self._on_controller,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            TFMessage,
            str(self.get_parameter("tf_topic").value),
            self._on_tf,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter("external_deadman_topic").value),
            self._on_external_deadman,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            PoseArray,
            str(self.get_parameter("hand_pose_topic").value),
            self._on_hand_pose,
            qos_profile_sensor_data,
        )
        self._command_pub = self.create_publisher(
            BimanualCommand, "/command_sources/teleop/command", 1
        )
        self._heartbeat_pub = self.create_publisher(
            UInt64, "/command_sources/teleop/heartbeat", qos_profile_sensor_data
        )
        self._home_pub = self.create_publisher(
            Empty, str(self.get_parameter("home_topic").value), 1
        )
        rate = float(self.get_parameter("control_rate_hz").value)
        self.create_timer(1.0 / rate, self._tick)
        self.get_logger().warning(
            f"teleop input started in {'COMMAND' if self._enabled else 'SHADOW'} mode"
        )

    def _on_pose(self, message: PoseArray) -> None:
        receive = time.monotonic_ns()
        try:
            left_index = int(self.get_parameter("left_pose_index").value)
            right_index = int(self.get_parameter("right_pose_index").value)
            left = message.poses[left_index]
            right = message.poses[right_index]
            self._pose_sequence += 1
            source_ns = (
                int(message.header.stamp.sec) * 1_000_000_000
                + int(message.header.stamp.nanosec)
            )
            self._sample = TrackingSample(
                left=TrackedPose.make(
                    [left.position.x, left.position.y, left.position.z],
                    [
                        left.orientation.x,
                        left.orientation.y,
                        left.orientation.z,
                        left.orientation.w,
                    ],
                ),
                right=TrackedPose.make(
                    [right.position.x, right.position.y, right.position.z],
                    [
                        right.orientation.x,
                        right.orientation.y,
                        right.orientation.z,
                        right.orientation.w,
                    ],
                ),
                sequence=self._pose_sequence,
                source_time_ns=source_ns,
                receive_monotonic_ns=receive,
                frame_id=_frame(message.header.frame_id),
            )
        except Exception as exc:
            self._sample = None
            self.get_logger().error(f"invalid Quest ee_poses: {exc}")

    def _on_controller(self, message: ByteMultiArray) -> None:
        try:
            value = msgpack.unpackb(_bytes(message), raw=False)
            self._squeeze = _validated_squeezes(value)
            raw_button = value.get(str(self.get_parameter("home_button_key").value), False)
            if not isinstance(raw_button, (bool, int)):
                raise ValueError("Quest home button must be boolean")
            home_button_down = bool(raw_button)
            # A is a rising-edge Home request. It intentionally does not
            # depend on the down-arrow pedal; the Home supervisor must enforce
            # F/T zero, local permission, collision/limit and daemon checks.
            if home_button_down and not self._home_button_down:
                self._home_pub.publish(Empty())
            self._home_button_down = home_button_down
            self._controller_received = time.monotonic_ns()
        except Exception as exc:
            self._squeeze = {"left": 0.0, "right": 0.0}
            self._controller_received = 0
            self._home_button_down = False
            self.get_logger().error(f"invalid controller_data: {exc}")

    def _on_tf(self, message: TFMessage) -> None:
        now = time.monotonic_ns()
        world = _frame(str(self.get_parameter("world_frame").value))
        watched = {
            _frame(str(self.get_parameter("left_wrist_frame").value)),
            _frame(str(self.get_parameter("right_wrist_frame").value)),
        }
        for transform in message.transforms:
            if (
                _frame(transform.header.frame_id) == world
                and _frame(transform.child_frame_id) in watched
            ):
                self._tf_received[_frame(transform.child_frame_id)] = now

    def _on_external_deadman(self, message: Bool) -> None:
        self._external_deadman = bool(message.data)
        self._external_deadman_received = time.monotonic_ns()

    def _on_hand_pose(self, message: PoseArray) -> None:
        now = time.monotonic_ns()
        if not self._retarget.calibrated:
            self._hands = {"left": None, "right": None}
            self._hands_received = {"left": 0, "right": 0}
            return
        try:
            features = split_bimanual_pose_array(
                pose_message_values(message.poses)
            )
            commands = {
                side: self._retarget.apply_features(side, features[side])
                for side in ("left", "right")
            }
            self._hands.update(commands)
            self._hands_received = {"left": now, "right": now}
        except Exception as exc:
            self._hands = {"left": None, "right": None}
            self._hands_received = {"left": 0, "right": 0}
            self.get_logger().error(
                f"invalid /xr_teleop/hand PoseArray; hand command disabled: {exc}"
            )

    def _deadman(self, now: int) -> bool:
        source = str(self.get_parameter("deadman_source").value)
        max_age = 150_000_000
        if source == "external_bool":
            return (
                self._external_deadman
                and now - self._external_deadman_received <= max_age
            )
        if now - self._controller_received > max_age:
            return False
        threshold = float(self.get_parameter("squeeze_threshold").value)
        if source == "quest_squeeze_either":
            return max(self._squeeze.values()) >= threshold
        if source != "quest_squeeze_both":
            return False
        return min(self._squeeze.values()) >= threshold

    def _tracking_sample(self, now: int) -> TrackingSample | None:
        if not bool(self.get_parameter("require_wrist_tf").value):
            return self._sample
        max_age = int(float(self.get_parameter("max_tf_age_s").value) * 1e9)
        for parameter in ("left_wrist_frame", "right_wrist_frame"):
            frame = _frame(str(self.get_parameter(parameter).value))
            if now - self._tf_received.get(frame, 0) > max_age:
                return None
        return self._sample

    def _tick(self) -> None:
        now = time.monotonic_ns()
        deadman = self._deadman(now)
        delta = self._mapper.update(
            self._tracking_sample(now), deadman=deadman, now_ns=now
        )
        hand_valid = self._retarget.calibrated and all(
            self._hands[side] is not None
            and now - self._hands_received[side]
            <= int(float(self.get_parameter("max_manus_age_s").value) * 1e9)
            for side in ("left", "right")
        )
        authority = self._enabled and deadman and delta.active and not delta.hold_latched
        valid_mask = 0x3 if authority else 0
        if authority and hand_valid:
            valid_mask |= 0xC
        self._command_sequence += 1
        message = BimanualCommand()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "world"
        message.schema_version = 1
        message.session_id = self._session
        message.source = "teleop"
        message.sequence = self._command_sequence
        ttl_ns = int(float(self.get_parameter("ttl_s").value) * 1e9)
        message.ttl.sec = ttl_ns // 1_000_000_000
        message.ttl.nanosec = ttl_ns % 1_000_000_000
        message.representation = BimanualCommand.CARTESIAN_ROT6D
        message.frame_id = "world"
        message.rotation_order = ROTATION_ORDER
        message.valid_mask = valid_mask
        message.deadman = authority
        point = BimanualCommandPoint()
        point.left_delta_xyz = delta.left_xyz.tolist()
        point.left_delta_rotation6d = delta.left_rotation6d.tolist()
        point.right_delta_xyz = delta.right_xyz.tolist()
        point.right_delta_rotation6d = delta.right_rotation6d.tolist()
        point.left_delta_quaternion_xyzw = [0.0, 0.0, 0.0, 1.0]
        point.right_delta_quaternion_xyzw = [0.0, 0.0, 0.0, 1.0]
        point.left_hand_targets = (
            self._hands["left"].tolist() if hand_valid else [0.0] * 6
        )
        point.right_hand_targets = (
            self._hands["right"].tolist() if hand_valid else [0.0] * 6
        )
        message.trajectory = [point]
        message.metadata_keys = ["teleop_mode", "mapping_reason"]
        message.metadata_values = [
            "command" if self._enabled else "shadow",
            delta.reason,
        ]
        self._command_pub.publish(message)
        heartbeat = UInt64()
        heartbeat.data = now
        self._heartbeat_pub.publish(heartbeat)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TeleopInput()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
