"""Independent Isaac Teleop/Quest/MANUS to canonical BimanualCommand adapter."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
import rclpy
import yaml
from flexiv_inspire_interfaces.msg import (
    BimanualCommand,
    BimanualCommandPoint,
)
from geometry_msgs.msg import PoseArray
from isaac_teleop_core.command import ROTATION_ORDER
from isaac_teleop_core.octet_sequence import decode_octet_sequence
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, ByteMultiArray, String, UInt64
from tf2_msgs.msg import TFMessage

from .foot_pedal import KEY_DOWN, KEY_SPACE, FootPedalMonitor
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


def _initial_command_sequence(monotonic_ns: int | None = None) -> int:
    """Seed source ordering above packets from an earlier teleop process.

    The control bridge deliberately survives between ``robot record`` runs.
    Starting every replacement teleop publisher at sequence 1 therefore makes
    its first actuating packet look older than the previous process.  Host
    monotonic microseconds form a process-independent epoch while retaining
    ample uint64 headroom for the bridge's six-bit trajectory expansion.
    """

    now = time.monotonic_ns() if monotonic_ns is None else int(monotonic_ns)
    if now < 0:
        raise ValueError("monotonic_ns cannot be negative")
    sequence = now // 1_000
    if sequence > ((1 << 64) - 1) >> 6:
        raise OverflowError("monotonic command sequence exceeds uint64 headroom")
    return sequence


def _frame(value: str) -> str:
    return str(value).strip().lstrip("/")


def _bytes(message: ByteMultiArray) -> bytes:
    return decode_octet_sequence(message.data)


def _validated_squeezes(value: object) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ValueError("controller_data is not a mapping")
    result = {
        "left": float(value.get("left_squeeze_value", 0.0)),
        "right": float(value.get("right_squeeze_value", 0.0)),
    }
    if any(
        not np.isfinite(item) or item < 0.0 or item > 1.0 for item in result.values()
    ):
        raise ValueError("Quest squeeze values must be finite and in [0,1]")
    return result


def _validated_button(value: object) -> bool:
    """Accept boolean clicks and legacy normalized numeric click actions."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        numeric = float(value)
        if np.isfinite(numeric) and numeric in (0.0, 1.0):
            return bool(numeric)
    raise ValueError("Quest home button must be boolean or 0/1")


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
        alpha = (value - self.source_open) / (self.source_closed - self.source_open)
        alpha = min(1.0, max(0.0, alpha))
        return float(self.output_open + alpha * (self.output_closed - self.output_open))


@dataclass(frozen=True)
class _ErgonomicsChannel:
    source: str
    source_points: np.ndarray
    output_points: np.ndarray

    def evaluate(self, values: dict[str, float]) -> float:
        try:
            value = float(values[self.source])
        except KeyError as exc:
            raise ValueError(f"MANUS Ergonomics field absent: {exc}") from exc
        if not np.isfinite(value):
            raise ValueError(f"MANUS Ergonomics {self.source} is not finite")
        return float(np.interp(value, self.source_points, self.output_points))


class _HandCommandFilter:
    def __init__(
        self,
        *,
        alpha: float,
        deadband: float,
        max_rate_per_s: float,
    ) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("hand low_pass_alpha must be in (0,1]")
        if deadband < 0.0 or max_rate_per_s <= 0.0:
            raise ValueError("hand filter deadband/rate is invalid")
        self.alpha = float(alpha)
        self.deadband = float(deadband)
        self.max_rate_per_s = float(max_rate_per_s)
        self._low_pass: dict[str, np.ndarray] = {}
        self._output: dict[str, np.ndarray] = {}
        self._updated_ns: dict[str, int] = {}

    def apply(self, side: str, target: np.ndarray, now_ns: int) -> np.ndarray:
        target = np.asarray(target, dtype=np.float64)
        previous_low_pass = self._low_pass.get(side)
        if previous_low_pass is None:
            filtered = target.copy()
        else:
            filtered = previous_low_pass + self.alpha * (target - previous_low_pass)
        self._low_pass[side] = filtered

        previous_output = self._output.get(side)
        previous_time = self._updated_ns.get(side)
        if previous_output is None or previous_time is None:
            output = filtered.copy()
        else:
            delta = filtered - previous_output
            delta[np.abs(delta) < self.deadband] = 0.0
            elapsed_s = max(0.0, (int(now_ns) - previous_time) * 1.0e-9)
            max_delta = self.max_rate_per_s * elapsed_s
            output = previous_output + np.clip(delta, -max_delta, max_delta)
        output = np.clip(output, 0.0, 1000.0)
        self._output[side] = output
        self._updated_ns[side] = int(now_ns)
        return output.copy()


class _Retarget:
    def __init__(self, path: str) -> None:
        self.channels: dict[str, dict[str, _Channel | _ErgonomicsChannel]] = {}
        self.source_format = ""
        self._filter = _HandCommandFilter(
            alpha=1.0,
            deadband=0.0,
            max_rate_per_s=1.0e9,
        )
        if not path:
            return
        document = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8"))
        if not isinstance(document, dict) or not document.get("calibrated", False):
            raise ValueError("MANUS calibration must be a calibrated mapping")
        self.source_format = str(document.get("source_format", ""))
        if self.source_format == "MANUS_SDK_ERGONOMICS_RADIANS":
            self._load_ergonomics(document)
            runtime_filter = document.get("filter", {})
            if not isinstance(runtime_filter, dict):
                raise ValueError("MANUS Ergonomics filter must be a mapping")
            self._filter = _HandCommandFilter(
                alpha=float(runtime_filter.get("low_pass_alpha", 0.35)),
                deadband=float(runtime_filter.get("output_deadband", 2.0)),
                max_rate_per_s=float(
                    runtime_filter.get("max_output_rate_per_s", 3000.0)
                ),
            )
            return
        if (
            document.get("schema_version") != 1
            or self.source_format != "ISAAC_XR_HAND_POSEARRAY_LEFT25_RIGHT25"
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

    def _load_ergonomics(self, document: dict[str, Any]) -> None:
        if document.get("schema_version") != 2:
            raise ValueError("MANUS Ergonomics calibration must use schema v2")
        sides = document.get("sides")
        if not isinstance(sides, dict):
            raise TypeError("MANUS Ergonomics calibration sides are missing")
        for side in ("left", "right"):
            raw_side = sides.get(side)
            if not isinstance(raw_side, dict):
                raise TypeError(f"MANUS Ergonomics {side} mapping is missing")
            self.channels[side] = {}
            for actuator in ACTUATORS:
                raw = raw_side.get(actuator)
                if not isinstance(raw, dict):
                    raise TypeError(f"invalid {side}.{actuator} calibration")
                source = str(raw.get("source", "")).strip()
                raw_points = raw.get("points")
                if (
                    not source
                    or not isinstance(raw_points, list)
                    or len(raw_points) < 2
                ):
                    raise ValueError(f"invalid {side}.{actuator} calibration")
                try:
                    points = np.asarray(raw_points, dtype=np.float64)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"invalid {side}.{actuator} calibration points"
                    ) from exc
                if (
                    points.ndim != 2
                    or points.shape[1] != 2
                    or not np.all(np.isfinite(points))
                ):
                    raise ValueError(f"invalid {side}.{actuator} calibration points")
                order = np.argsort(points[:, 0])
                source_points = points[order, 0]
                output_points = points[order, 1]
                if np.any(np.diff(source_points) <= 1.0e-6):
                    raise ValueError(f"degenerate {side}.{actuator} calibration points")
                if np.any(output_points < 0.0) or np.any(output_points > 1000.0):
                    raise ValueError(
                        f"{side}.{actuator} output points outside [0,1000]"
                    )
                output_deltas = np.diff(output_points)
                if not (np.all(output_deltas >= 0.0) or np.all(output_deltas <= 0.0)):
                    raise ValueError(f"{side}.{actuator} mapping must be monotonic")
                self.channels[side][actuator] = _ErgonomicsChannel(
                    source=source,
                    source_points=source_points,
                    output_points=output_points,
                )

    @property
    def calibrated(self) -> bool:
        return set(self.channels) == {"left", "right"}

    @property
    def uses_ergonomics(self) -> bool:
        return self.source_format == "MANUS_SDK_ERGONOMICS_RADIANS"

    def apply_features(self, side: str, features: dict[str, float]) -> np.ndarray:
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
        if (
            not np.all(np.isfinite(result))
            or np.any(result < 0)
            or np.any(result > 1000)
        ):
            raise ValueError("retargeted hand command outside [0,1000]")
        return result

    def apply_ergonomics(
        self,
        side: str,
        values: dict[str, float],
        *,
        now_ns: int,
    ) -> np.ndarray:
        if not self.uses_ergonomics:
            raise ValueError("active MANUS calibration does not use Ergonomics")
        result = np.asarray(
            [self.channels[side][actuator].evaluate(values) for actuator in ACTUATORS],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(result)):
            raise ValueError("retargeted hand command contains NaN or Inf")
        return self._filter.apply(side, result, now_ns)


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
            "deadman_source": "external_bool",
            "foot_pedal": "name:input-remapper keyboard",
            "enable_key_code": KEY_SPACE,
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
            "rotation_gain": 1.0,
            "max_translation_step_m": 0.01,
            "max_rotation_step_rad": 0.10,
            "axis_rotation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "hand_pose_topic": "/xr_teleop/hand",
            "manus_left_ergonomics_topic": "/manus/left/ergonomics",
            "manus_right_ergonomics_topic": "/manus/right/ergonomics",
            "manus_calibration": "",
            "max_manus_age_s": 0.10,
            "home_button_key": "right_primary_click",
            "home_topic": "/episode/control",
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
            rotation_gain=float(self.get_parameter("rotation_gain").value),
            max_translation_step_m=float(
                self.get_parameter("max_translation_step_m").value
            ),
            max_rotation_step_rad=float(
                self.get_parameter("max_rotation_step_rad").value
            ),
        )
        self._retarget = _Retarget(str(self.get_parameter("manus_calibration").value))
        self._pose_sequence = 0
        self._sample: TrackingSample | None = None
        self._tf_received: dict[str, int] = {}
        self._controller_received = 0
        self._squeeze = {"left": 0.0, "right": 0.0}
        self._external_deadman = False
        self._external_deadman_received = 0
        self._hands: dict[str, np.ndarray | None] = {"left": None, "right": None}
        self._hands_received = {"left": 0, "right": 0}
        self._manus_ready = False
        self._command_sequence = _initial_command_sequence()
        self._home_button_down = False
        self._pedal_pressed = False
        self._tracking_unavailable_reason = ""

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
        if self._retarget.uses_ergonomics:
            for side in ("left", "right"):
                self.create_subscription(
                    JointState,
                    str(self.get_parameter(f"manus_{side}_ergonomics_topic").value),
                    lambda message, selected=side: self._on_ergonomics(
                        selected, message
                    ),
                    qos_profile_sensor_data,
                )
        else:
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
            String, str(self.get_parameter("home_topic").value), 1
        )
        self._pedal: FootPedalMonitor | None = None
        if str(self.get_parameter("deadman_source").value) == "pedal":
            configured_enable = int(self.get_parameter("enable_key_code").value)
            self._pedal = FootPedalMonitor(
                Path(str(self.get_parameter("foot_pedal").value)),
                self._on_pedal_state,
                enable_key_codes=(configured_enable, KEY_SPACE, KEY_DOWN),
            )
            self._pedal.start()
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
            source_ns = int(message.header.stamp.sec) * 1_000_000_000 + int(
                message.header.stamp.nanosec
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
            raw_button = value.get(
                str(self.get_parameter("home_button_key").value), False
            )
            home_button_down = _validated_button(raw_button)
            if home_button_down and not self._home_button_down:
                request = String()
                request.data = "home"
                self._home_pub.publish(request)
            self._home_button_down = home_button_down
            self._controller_received = time.monotonic_ns()
        except Exception as exc:
            self._squeeze = {"left": 0.0, "right": 0.0}
            self._controller_received = 0
            self._home_button_down = False
            self.get_logger().error(
                f"invalid controller_data: {exc}", throttle_duration_sec=1.0
            )

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

    def _on_pedal_state(self, pressed: bool) -> None:
        self._pedal_pressed = bool(pressed)

    def _on_hand_pose(self, message: PoseArray) -> None:
        now = time.monotonic_ns()
        if not self._retarget.calibrated:
            self._hands = {"left": None, "right": None}
            self._hands_received = {"left": 0, "right": 0}
            return
        try:
            features = split_bimanual_pose_array(pose_message_values(message.poses))
            commands = {
                side: self._retarget.apply_features(side, features[side])
                for side in ("left", "right")
            }
            self._hands.update(commands)
            self._hands_received = {"left": now, "right": now}
            self._refresh_manus_ready(now)
        except Exception as exc:
            self._hands = {"left": None, "right": None}
            self._hands_received = {"left": 0, "right": 0}
            if self._manus_ready:
                self.get_logger().warning(
                    "MANUS_HANDS_LOST: glove tracking became invalid"
                )
            self._manus_ready = False
            self.get_logger().error(
                f"invalid /xr_teleop/hand PoseArray; hand command disabled: {exc}",
                throttle_duration_sec=1.0,
            )

    def _on_ergonomics(self, side: str, message: JointState) -> None:
        now = time.monotonic_ns()
        try:
            if len(message.name) != len(message.position):
                raise ValueError("JointState names/positions have different lengths")
            if len(set(message.name)) != len(message.name):
                raise ValueError("JointState contains duplicate field names")
            values = {
                str(name): float(value)
                for name, value in zip(message.name, message.position, strict=True)
            }
            self._hands[side] = self._retarget.apply_ergonomics(
                side, values, now_ns=now
            )
            self._hands_received[side] = now
            self._refresh_manus_ready(now)
        except (KeyError, TypeError, ValueError) as exc:
            self._hands[side] = None
            self._hands_received[side] = 0
            if self._manus_ready:
                self.get_logger().warning(
                    f"MANUS_HANDS_LOST: {side} Ergonomics became invalid"
                )
            self._manus_ready = False
            self.get_logger().error(
                f"invalid MANUS {side} Ergonomics; hand command disabled: {exc}",
                throttle_duration_sec=1.0,
            )

    def _refresh_manus_ready(self, now: int) -> None:
        max_age = int(float(self.get_parameter("max_manus_age_s").value) * 1e9)
        ready = all(
            self._hands[side] is not None
            and now - self._hands_received[side] <= max_age
            for side in ("left", "right")
        )
        if ready and not self._manus_ready:
            source = "Ergonomics" if self._retarget.uses_ergonomics else "skeleton"
            self.get_logger().info(
                "MANUS_HANDS_READY: both gloves are valid; "
                f"{source} retargeting is active"
            )
        self._manus_ready = ready

    def _deadman(self, now: int) -> bool:
        source = str(self.get_parameter("deadman_source").value)
        if source == "pedal":
            return self._pedal_pressed
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
            self._tracking_unavailable_reason = (
                "quest_pose_missing" if self._sample is None else ""
            )
            return self._sample
        max_age = int(float(self.get_parameter("max_tf_age_s").value) * 1e9)
        missing = []
        for parameter in ("left_wrist_frame", "right_wrist_frame"):
            frame = _frame(str(self.get_parameter(parameter).value))
            if now - self._tf_received.get(frame, 0) > max_age:
                missing.append(frame)
        if missing:
            self._tracking_unavailable_reason = (
                "controller_tracking_missing:" + ",".join(missing)
            )
            return None
        if self._sample is None:
            self._tracking_unavailable_reason = "quest_pose_missing"
            return None
        self._tracking_unavailable_reason = ""
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
        authority = (
            self._enabled and deadman and delta.active and not delta.hold_latched
        )
        if self._enabled and deadman and not authority:
            reason = self._tracking_unavailable_reason or delta.reason
            self.get_logger().warning(
                "中踏板已踩下，但机械臂未使能："
                f"{reason}；请确认左右 Quest 控制器均被追踪，"
                "松开中踏板后再踩下",
                throttle_duration_sec=1.0,
            )
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
            self._tracking_unavailable_reason or delta.reason,
        ]
        self._command_pub.publish(message)
        heartbeat = UInt64()
        heartbeat.data = now
        self._heartbeat_pub.publish(heartbeat)

    def destroy_node(self) -> bool:
        if self._pedal is not None:
            self._pedal.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TeleopInput()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
