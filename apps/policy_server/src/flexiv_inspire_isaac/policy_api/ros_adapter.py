"""Runnable ROS 2 adapter for TLS PolicyService v1.

This process never enables hardware or performs F/T zeroing.  It mirrors the
local observation graph into protobuf snapshots and publishes only to the
policy command-source topics; the local control supervisor remains authoritative.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np

from .broker import LatestActionBuffer, ObservationBroker, PolicyStreamLiveness
from .channel_broker import (
    ChannelBroker,
    ChannelDataError,
    SampleTiming,
    image_sample,
    tensor_sample,
)
from .lease import ControlLeaseManager, LocalControlState
from .models import ActionChunk
from .server import TlsFiles, serve
from .system_schema import build_system_schema
from .generated import policy_service_v1_pb2 as pb
from flexiv_inspire_isaac.dftp.protocol import TACTILE_LAYOUT, TOTAL_TAXELS
from isaac_teleop_core.command import ROTATION_ORDER


def initial_policy_command_sequence(monotonic_ns: int | None = None) -> int:
    """Seed ROS policy ordering above earlier clients and adapter processes.

    RPC action sequences are scoped to a lease and restart at one after a
    reconnect. The control bridge instead owns one long-lived hardware session
    and requires every command from a source to increase monotonically. Use the
    host monotonic clock as the ROS-side epoch while retaining six bits for the
    bridge's trajectory expansion.
    """

    now = time.monotonic_ns() if monotonic_ns is None else int(monotonic_ns)
    if now < 0:
        raise ValueError("monotonic_ns cannot be negative")
    sequence = now // 1_000
    if sequence > ((1 << 64) - 1) >> 6:
        raise OverflowError("monotonic policy sequence exceeds uint64 headroom")
    return sequence


def _time_ns(value: Any) -> int:
    return int(value.sec) * 1_000_000_000 + int(value.nanosec)


def _metadata(acquisition: Any | None, *, now_ns: int | None = None) -> pb.TimedMetadata:
    snapshot_ns = time.monotonic_ns() if now_ns is None else now_ns
    if acquisition is None:
        return pb.TimedMetadata(
            host_receive_time_ns=snapshot_ns,
            valid=False,
            invalid_reason="observation-not-yet-received",
            host_clock_domain="host_monotonic",
            timing_valid=False,
        )
    mapped_ns = _time_ns(acquisition.mapped_host_time)
    timing_valid = bool(acquisition.timing_valid) and mapped_ns > 0
    age_ns = snapshot_ns - mapped_ns if timing_valid else 0
    reason = str(acquisition.invalid_reason)
    if not timing_valid:
        reason = ";".join(filter(None, (reason, "source-clock-unmapped")))
    elif age_ns < 0:
        timing_valid = False
        age_ns = 0
        reason = ";".join(filter(None, (reason, "mapped-source-after-snapshot")))
    valid = bool(acquisition.valid) and timing_valid
    return pb.TimedMetadata(
        source_time_ns=_time_ns(acquisition.source_time),
        host_receive_time_ns=_time_ns(acquisition.host_receive_time),
        mapped_host_time_ns=mapped_ns,
        acquisition_start_ns=_time_ns(acquisition.acquisition_start),
        acquisition_end_ns=_time_ns(acquisition.acquisition_end),
        sequence=int(acquisition.source_sequence),
        valid=valid,
        age_ns=age_ns,
        invalid_reason=reason,
        source_clock_domain=str(acquisition.source_clock_domain),
        host_clock_domain=str(acquisition.host_clock_domain),
        timing_valid=timing_valid,
    )


def _sample_timing(acquisition: Any, *, now_ns: int | None = None) -> SampleTiming:
    snapshot_ns = time.monotonic_ns() if now_ns is None else now_ns
    if acquisition is None:
        return SampleTiming(
            source_time_ns=0,
            host_receive_time_ns=snapshot_ns,
            mapped_host_time_ns=0,
            acquisition_start_ns=0,
            acquisition_end_ns=0,
            sequence=0,
            valid=False,
            invalid_reason="observation-not-yet-received",
            source_clock_domain="",
            host_clock_domain="host_monotonic",
            timing_valid=False,
        )
    mapped_ns = _time_ns(acquisition.mapped_host_time)
    timing_valid = bool(acquisition.timing_valid) and mapped_ns > 0
    reason = str(acquisition.invalid_reason)
    if not timing_valid:
        reason = ";".join(filter(None, (reason, "source-clock-unmapped")))
    elif mapped_ns > snapshot_ns:
        timing_valid = False
        reason = ";".join(filter(None, (reason, "mapped-source-after-receive")))
    return SampleTiming(
        source_time_ns=_time_ns(acquisition.source_time),
        host_receive_time_ns=_time_ns(acquisition.host_receive_time),
        mapped_host_time_ns=mapped_ns,
        acquisition_start_ns=_time_ns(acquisition.acquisition_start),
        acquisition_end_ns=_time_ns(acquisition.acquisition_end),
        sequence=int(acquisition.source_sequence),
        valid=bool(acquisition.valid) and timing_valid,
        invalid_reason=reason,
        source_clock_domain=str(acquisition.source_clock_domain),
        host_clock_domain=str(acquisition.host_clock_domain),
        timing_valid=timing_valid,
    )


def _vector(values: Any) -> list[float]:
    return [float(value) for value in values]


def _wrench(message: Any) -> list[float]:
    return [
        float(message.force.x), float(message.force.y), float(message.force.z),
        float(message.torque.x), float(message.torque.y), float(message.torque.z),
    ]


def _arm_observation(message: Any | None, *, now_ns: int | None = None) -> pb.ArmObservation:
    if message is None:
        result = pb.ArmObservation()
        result.timing.CopyFrom(_metadata(None, now_ns=now_ns))
        return result
    pose = message.tcp_pose
    twist = message.tcp_twist
    result = pb.ArmObservation(
        q=_vector(message.q),
        dq=_vector(message.dq),
        tau=_vector(message.tau),
        tau_des=_vector(message.tau_des),
        tau_ext=_vector(message.tau_ext),
        tau_interact=_vector(message.tau_interact),
        tcp_pose_xyzw=[
            float(pose.position.x), float(pose.position.y), float(pose.position.z),
            float(pose.orientation.x), float(pose.orientation.y),
            float(pose.orientation.z), float(pose.orientation.w),
        ],
        tcp_twist=[
            float(twist.linear.x), float(twist.linear.y), float(twist.linear.z),
            float(twist.angular.x), float(twist.angular.y), float(twist.angular.z),
        ],
        raw_ft=_wrench(message.raw_ft),
        tcp_wrench=_wrench(message.tcp_wrench),
        temperature=_vector(message.temperature),
    )
    timing = _metadata(message.acquisition, now_ns=now_ns)
    result.timing.CopyFrom(timing)
    for field in (
        "q", "dq", "tau", "tau_des", "tau_ext", "tau_interact",
        "tcp_pose_xyzw", "tcp_twist", "raw_ft", "tcp_wrench", "temperature",
    ):
        result.field_timing[field].CopyFrom(timing)
    return result


def _hand_observation(message: Any | None, *, now_ns: int | None = None) -> pb.HandObservation:
    if message is None:
        result = pb.HandObservation()
        result.timing.CopyFrom(_metadata(None, now_ns=now_ns))
        return result
    result = pb.HandObservation(
        angle=_vector(message.angle),
        position=_vector(message.position),
        actual_force=_vector(message.actual_force),
        current=_vector(message.current),
        temperature=_vector(message.temperature),
        error=[int(value) for value in message.error],
        status=[int(value) for value in message.status],
    )
    timing = _metadata(message.acquisition, now_ns=now_ns)
    result.timing.CopyFrom(timing)
    for field in (
        "angle", "position", "actual_force", "current", "temperature", "error", "status"
    ):
        field_acquisition = getattr(message, f"{field}_timing", None)
        result.field_timing[field].CopyFrom(
            _metadata(field_acquisition, now_ns=now_ns)
        )
    return result


def _tactile_observation(message: Any | None, *, now_ns: int | None = None) -> pb.TactileObservation:
    if message is None:
        result = pb.TactileObservation()
        result.frame_timing.CopyFrom(_metadata(None, now_ns=now_ns))
        return result
    result = pb.TactileObservation(
        frame_start_ns=_time_ns(message.acquisition.acquisition_start),
        frame_end_ns=_time_ns(message.acquisition.acquisition_end),
        taxel_count=int(message.taxel_count),
    )
    result.frame_timing.CopyFrom(_metadata(message.acquisition, now_ns=now_ns))
    for surface in message.surfaces:
        target = result.surfaces.add(
            name=str(surface.name),
            rows=int(surface.rows),
            cols=int(surface.columns),
            values=[int(value) for value in surface.taxels],
        )
        target.timing.CopyFrom(_metadata(surface.acquisition, now_ns=now_ns))
    expected_layout = [(spec.name, spec.rows, spec.cols, spec.taxels) for spec in TACTILE_LAYOUT]
    observed_layout = [
        (surface.name, surface.rows, surface.cols, len(surface.values))
        for surface in result.surfaces
    ]
    if (
        len(result.surfaces) != 17
        or result.taxel_count != TOTAL_TAXELS
        or observed_layout != expected_layout
    ):
        result.frame_timing.valid = False
        result.frame_timing.invalid_reason = ";".join(filter(None, (
            result.frame_timing.invalid_reason, "tactile-layout-or-taxel-count-invalid"
        )))
    return result


def remaining_action_timing(chunk: ActionChunk, now_ns: int) -> tuple[int, tuple[int, ...]]:
    elapsed_ns = max(0, now_ns - chunk.server_receive_monotonic_ns)
    remaining_ns = chunk.ttl_from_server_receive_ns - elapsed_ns
    if remaining_ns <= 0:
        raise ValueError("expired-policy-action-before-ros")
    offsets = tuple(
        max(0, int(round(point.execute_after_s * 1e9)) - elapsed_ns)
        for point in chunk.points
    )
    if any(offset >= remaining_ns for offset in offsets):
        raise ValueError("policy-point-outside-remaining-ttl")
    return remaining_ns, offsets


def policy_heartbeat_sequence(
    action_liveness: PolicyStreamLiveness,
    lease_manager: ControlLeaseManager,
    state: LocalControlState,
) -> int | None:
    """Return the live policy sequence or disarm on any local/lease failure."""
    active = action_liveness.current()
    if active is None:
        return None
    if (
        not state.policy_lease_allowed
        or lease_manager.clock_ns() >= active.action_deadline_ns
        or state.session_id != active.session_id
        or not lease_manager.current_valid(active.lease_id, active.session_id)
    ):
        action_liveness.clear()
        return None
    return active.sequence


@dataclass
class Snapshot:
    control: Any | None = None
    arms: dict[str, Any] | None = None
    hands: dict[str, Any] | None = None
    tactile: dict[str, Any] | None = None
    camera_frames: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.arms = {} if self.arms is None else self.arms
        self.hands = {} if self.hands is None else self.hands
        self.tactile = {} if self.tactile is None else self.tactile
        self.camera_frames = {} if self.camera_frames is None else self.camera_frames


class SharedRosState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.snapshot = Snapshot()

    def local_control_state(self) -> LocalControlState:
        with self.lock:
            message = self.snapshot.control
            if message is None:
                return LocalControlState("", False, False, False, False, False, "DISABLED")
            name = str(message.state_name).upper()
            active_source = str(message.active_source)
            locally_armed = name == "POLICY_ARMED" or (
                name == "ACTIVE" and active_source == "policy"
            )
            return LocalControlState(
                session_id=str(message.session_id),
                ft_zeroed=bool(message.ft_zeroed_for_session),
                local_policy_authorized=bool(message.local_permission) and locally_armed,
                pedal_valid=bool(message.physical_pedal),
                arms_online=bool(message.arms_online),
                hands_online=bool(message.hands_online),
                state=name,
            )


def build_ros_node(
    shared: SharedRosState,
    broker: ObservationBroker,
    action_buffer: LatestActionBuffer,
    grpc_loop,
    lease_manager: ControlLeaseManager,
    action_liveness: PolicyStreamLiveness,
    channel_broker: ChannelBroker | None = None,
):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from std_msgs.msg import Empty, UInt64
    from flexiv_inspire_interfaces.msg import (
        ArmState, BimanualCommand, BimanualCommandPoint, CameraFrame,
        ControlState, HandState, TactileFrame,
    )

    class PolicyRosAdapter(Node):
        def __init__(self) -> None:
            super().__init__("flexiv_inspire_policy_service")
            self._sequence = 0
            self._command_sequence = initial_policy_command_sequence()
            self._broker = broker
            self._action_buffer = action_buffer
            self._grpc_loop = grpc_loop
            self._lease_manager = lease_manager
            self._action_liveness = action_liveness
            self._channel_broker = channel_broker
            self._channel_futures: set[Any] = set()
            self._command_pub = self.create_publisher(
                BimanualCommand, "/command_sources/policy/command", 1
            )
            self._heartbeat_pub = self.create_publisher(
                UInt64, "/command_sources/policy/heartbeat", qos_profile_sensor_data
            )
            self._stop_pub = self.create_publisher(Empty, "/control/stop", 1)
            self.create_subscription(ControlState, "/control/state", self._on_control, 1)
            for side in ("left", "right"):
                self.create_subscription(
                    ArmState, f"/robot/{side}_arm/state",
                    lambda msg, selected=side: self._on_arm(selected, msg),
                    qos_profile_sensor_data,
                )
                self.create_subscription(
                    HandState, f"/robot/{side}_hand/state",
                    lambda msg, selected=side: self._on_hand(selected, msg),
                    qos_profile_sensor_data,
                )
                self.create_subscription(
                    TactileFrame, f"/robot/{side}_hand/tactile_raw",
                    lambda msg, selected=side: self._on_tactile(selected, msg),
                    qos_profile_sensor_data,
                )
            for camera in ("head", "left_wrist", "right_wrist"):
                root = f"/camera/{camera}/color"
                self.create_subscription(
                    CameraFrame, f"{root}/frame",
                    lambda msg, selected=camera: self._on_camera(selected, msg),
                    qos_profile_sensor_data,
                )
            self.create_timer(1.0 / 30.0, self._publish_observation)
            self.create_timer(0.005, self._publish_action)
            self.create_timer(0.05, self._publish_stream_heartbeat)

        def _store(self, collection: str, key: str, message: Any) -> None:
            with shared.lock:
                getattr(shared.snapshot, collection)[key] = message

        def _submit_channel_samples(self, samples) -> None:
            if self._channel_broker is None:
                return
            try:
                materialized = tuple(samples)
            except ChannelDataError as exc:
                self.get_logger().warning(f"PolicyData v2 sample rejected: {exc}")
                return
            future = asyncio.run_coroutine_threadsafe(
                self._channel_broker.publish_many(materialized), self._grpc_loop
            )
            self._channel_futures.add(future)

            def completed(item) -> None:
                self._channel_futures.discard(item)
                try:
                    item.result()
                except Exception as exc:  # noqa: BLE001
                    self.get_logger().warning(f"PolicyData v2 publish failed: {exc}")

            future.add_done_callback(completed)

        def _on_arm(self, side: str, message: Any) -> None:
            self._store("arms", side, message)
            if self._channel_broker is None:
                return
            timing = _sample_timing(message.acquisition)
            pose = message.tcp_pose
            twist = message.tcp_twist
            samples = []
            values_by_suffix = {
                "q": message.q,
                "dq": message.dq,
                "tau": message.tau,
                "tau_des": message.tau_des,
                "tau_ext": message.tau_ext,
                "tau_interact": message.tau_interact,
                "tcp_pose": (
                    pose.position.x, pose.position.y, pose.position.z,
                    pose.orientation.x, pose.orientation.y,
                    pose.orientation.z, pose.orientation.w,
                ),
                "tcp_twist": (
                    twist.linear.x, twist.linear.y, twist.linear.z,
                    twist.angular.x, twist.angular.y, twist.angular.z,
                ),
                "raw_ft": _wrench(message.raw_ft),
                "tcp_wrench": _wrench(message.tcp_wrench),
            }
            try:
                for suffix, values in values_by_suffix.items():
                    channel_id = f"arm.{side}.{suffix}"
                    samples.append(
                        tensor_sample(
                            channel_id,
                            values,
                            timing,
                            self._channel_broker.descriptor(channel_id),
                        )
                    )
            except ChannelDataError as exc:
                self.get_logger().warning(f"PolicyData v2 {side} arm rejected: {exc}")
                return
            self._submit_channel_samples(samples)

        def _on_hand(self, side: str, message: Any) -> None:
            self._store("hands", side, message)
            if self._channel_broker is None:
                return
            timing = _sample_timing(message.acquisition)
            samples = []
            try:
                for suffix, values in (
                    ("angle", message.angle),
                    ("actual_force", message.actual_force),
                ):
                    channel_id = f"hand.{side}.{suffix}"
                    samples.append(
                        tensor_sample(
                            channel_id,
                            values,
                            timing,
                            self._channel_broker.descriptor(channel_id),
                        )
                    )
            except ChannelDataError as exc:
                self.get_logger().warning(f"PolicyData v2 {side} hand rejected: {exc}")
                return
            self._submit_channel_samples(samples)

        def _on_tactile(self, side: str, message: Any) -> None:
            self._store("tactile", side, message)
            if self._channel_broker is None:
                return
            channel_id = f"hand.{side}.tactile"
            values = np.asarray(
                [taxel for surface in message.surfaces for taxel in surface.taxels],
                dtype=np.uint16,
            )
            try:
                sample = tensor_sample(
                    channel_id,
                    values,
                    _sample_timing(message.acquisition),
                    self._channel_broker.descriptor(channel_id),
                )
            except ChannelDataError as exc:
                self.get_logger().warning(f"PolicyData v2 {side} tactile rejected: {exc}")
                return
            self._submit_channel_samples((sample,))

        def _on_camera(self, camera: str, message: Any) -> None:
            self._store("camera_frames", camera, message)
            if self._channel_broker is None:
                return
            image = message.image
            encoding = str(image.format).lower()
            if encoding.startswith("jpeg"):
                encoding = "jpeg"
            channel_id = f"camera.{camera}.rgb"
            try:
                sample = image_sample(
                    channel_id,
                    bytes(image.data),
                    encoding=encoding,
                    width=int(message.width),
                    height=int(message.height),
                    channels=3,
                    timing=_sample_timing(message.acquisition),
                )
            except ChannelDataError as exc:
                self.get_logger().warning(f"PolicyData v2 {camera} image rejected: {exc}")
                return
            self._submit_channel_samples((sample,))

        def _on_control(self, message: Any) -> None:
            with shared.lock:
                shared.snapshot.control = message

        def request_safe_stop(self, _reason: str) -> None:
            self._stop_pub.publish(Empty())

        def _publish_observation(self) -> None:
            with shared.lock:
                snap = Snapshot(
                    control=shared.snapshot.control,
                    arms=dict(shared.snapshot.arms),
                    hands=dict(shared.snapshot.hands),
                    tactile=dict(shared.snapshot.tactile),
                    camera_frames=dict(shared.snapshot.camera_frames),
                )
            self._sequence += 1
            snapshot_ns = time.monotonic_ns()
            control_state = "DISABLED" if snap.control is None else str(snap.control.state_name)
            session_id = "" if snap.control is None else str(snap.control.session_id)
            observation = pb.Observation(
                schema_version=1,
                session_id=session_id,
                sequence=self._sequence,
                snapshot_monotonic_ns=snapshot_ns,
                control_state=control_state,
            )
            observation.left_arm.CopyFrom(_arm_observation(snap.arms.get("left"), now_ns=snapshot_ns))
            observation.right_arm.CopyFrom(_arm_observation(snap.arms.get("right"), now_ns=snapshot_ns))
            observation.left_hand.CopyFrom(_hand_observation(snap.hands.get("left"), now_ns=snapshot_ns))
            observation.right_hand.CopyFrom(_hand_observation(snap.hands.get("right"), now_ns=snapshot_ns))
            observation.left_tactile.CopyFrom(_tactile_observation(snap.tactile.get("left"), now_ns=snapshot_ns))
            observation.right_tactile.CopyFrom(_tactile_observation(snap.tactile.get("right"), now_ns=snapshot_ns))
            camera_enum = {
                "head": pb.CAMERA_HEAD,
                "left_wrist": pb.CAMERA_LEFT_WRIST,
                "right_wrist": pb.CAMERA_RIGHT_WRIST,
            }
            for camera in ("head", "left_wrist", "right_wrist"):
                frame = snap.camera_frames.get(camera)
                if frame is None:
                    target = observation.camera_images.add(
                        camera=camera_enum[camera], encoding="jpeg", data=b"",
                        width=424, height=240,
                    )
                    target.timing.CopyFrom(_metadata(None, now_ns=snapshot_ns))
                    continue
                image = frame.image
                target = observation.camera_images.add(
                    camera=camera_enum[camera], encoding="jpeg", data=bytes(image.data),
                    width=int(frame.width), height=int(frame.height),
                )
                target.timing.CopyFrom(_metadata(frame.acquisition, now_ns=snapshot_ns))
                if (
                    str(frame.camera) != camera
                    or not str(image.format).lower().startswith("jpeg")
                    or not image.data
                    or (int(frame.width), int(frame.height)) != (424, 240)
                ):
                    target.timing.valid = False
                    target.timing.invalid_reason = ";".join(filter(None, (
                        target.timing.invalid_reason, "camera-name-or-encoding-mismatch"
                    )))
            asyncio.run_coroutine_threadsafe(
                self._broker.publish(observation), self._grpc_loop
            )

        def _publish_stream_heartbeat(self) -> None:
            sequence = policy_heartbeat_sequence(
                self._action_liveness,
                self._lease_manager,
                shared.local_control_state(),
            )
            if sequence is None:
                return
            heartbeat = UInt64()
            heartbeat.data = sequence
            self._heartbeat_pub.publish(heartbeat)

        def _publish_action(self) -> None:
            chunk = self._action_buffer.take()
            if chunk is None:
                return
            now_ns = time.monotonic_ns()
            try:
                remaining_ns, offsets_ns = remaining_action_timing(chunk, now_ns)
            except ValueError as exc:
                self.request_safe_stop(str(exc))
                return
            message = BimanualCommand()
            message.header.stamp = self.get_clock().now().to_msg()
            message.header.frame_id = "world"
            message.schema_version = 1
            message.session_id = chunk.session_id
            message.source = "policy"
            # The RPC sequence is lease-local. Reusing it after reconnect would
            # look older than an earlier command to the persistent bridge.
            self._command_sequence += 1
            message.sequence = self._command_sequence
            message.ttl.sec = int(remaining_ns // 1_000_000_000)
            message.ttl.nanosec = int(remaining_ns % 1_000_000_000)
            message.representation = BimanualCommand.CARTESIAN_ROT6D
            message.frame_id = "world"
            message.rotation_order = ROTATION_ORDER
            message.valid_mask = (
                BimanualCommand.LEFT_ARM_VALID | BimanualCommand.RIGHT_ARM_VALID
                | BimanualCommand.LEFT_HAND_VALID | BimanualCommand.RIGHT_HAND_VALID
            )
            message.deadman = chunk.deadman
            for point, offset_ns in zip(chunk.points, offsets_ns):
                target = BimanualCommandPoint()
                target.execute_after.sec = offset_ns // 1_000_000_000
                target.execute_after.nanosec = offset_ns % 1_000_000_000
                values = point.values
                target.left_delta_xyz = values[0:3]
                target.left_delta_rotation6d = values[3:9]
                target.right_delta_xyz = values[9:12]
                target.right_delta_rotation6d = values[12:18]
                target.left_hand_targets = values[18:24]
                target.right_hand_targets = values[24:30]
                message.trajectory.append(target)
            self._command_pub.publish(message)
            heartbeat = UInt64()
            heartbeat.data = chunk.sequence
            self._heartbeat_pub.publish(heartbeat)

    return PolicyRosAdapter()


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="ROS 2 + TLS PolicyService v1 adapter")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--server-cert", type=Path, required=True)
    parser.add_argument("--server-key", type=Path, required=True)
    parser.add_argument("--client-ca", type=Path)
    parser.add_argument("--arm-rate-hz", type=float, default=300.0)
    parser.add_argument("--hand-rate-hz", type=float, default=200.0)
    parser.add_argument("--tactile-rate-hz", type=float, default=15.0)
    parser.add_argument("--camera-rate-hz", type=float, default=15.0)
    parser.add_argument("--action-rate-hz", type=float, default=30.0)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    options = _parse_args(argv)
    import rclpy
    from rclpy.executors import MultiThreadedExecutor

    async def run() -> None:
        loop = asyncio.get_running_loop()
        broker = ObservationBroker()
        actions: LatestActionBuffer[ActionChunk] = LatestActionBuffer()
        shared = SharedRosState()
        lease = ControlLeaseManager()
        action_liveness = PolicyStreamLiveness()
        system_schema = build_system_schema(
            arm_rate_hz=options.arm_rate_hz,
            hand_rate_hz=options.hand_rate_hz,
            tactile_rate_hz=options.tactile_rate_hz,
            camera_rate_hz=options.camera_rate_hz,
            action_rate_hz=options.action_rate_hz,
        )
        channel_broker = ChannelBroker(system_schema)
        rclpy.init(args=None)
        node = build_ros_node(
            shared,
            broker,
            actions,
            loop,
            lease,
            action_liveness,
            channel_broker,
        )
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(node)
        thread = threading.Thread(target=executor.spin, name="policy-ros-executor", daemon=True)
        thread.start()
        try:
            await serve(
                bind_host=options.bind,
                port=options.port,
                tls=TlsFiles(options.server_cert, options.server_key, options.client_ca),
                lease_manager=lease,
                local_state=shared.local_control_state,
                observation_broker=broker,
                action_buffer=actions,
                stop_callback=node.request_safe_stop,
                action_liveness=action_liveness,
                channel_broker=channel_broker,
                system_schema=system_schema,
            )
        finally:
            executor.shutdown(timeout_sec=2.0)
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
            thread.join(timeout=2.0)

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
