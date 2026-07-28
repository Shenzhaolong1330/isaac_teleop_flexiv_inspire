"""ROS 2 bridge from Isaac Teleop topics to the isolated hardware gateway."""

from __future__ import annotations

import argparse
import json
import logging
import socket
import stat
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import PoseArray, PoseStamped, TwistStamped
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, ByteMultiArray, String
from tf2_msgs.msg import TFMessage

from .config import BridgeConfig, load_config
from .mapping import (
    BridgeDecision,
    BridgeState,
    DualArmAbsoluteMapper,
    Pose,
    PosePairSample,
    action_vectors,
)
from .math3d import rotvec_to_quat
from .protocol import decode_packet, encode_packet

LOGGER = logging.getLogger("isaac_flexiv_ros2")


@dataclass(frozen=True)
class _RawPosePair:
    left: Pose
    right: Pose
    sequence: int
    receive_monotonic_ns: int
    source_stamp_ns: int
    frame_id: str


def _safe_remove_socket(path_text: str) -> None:
    path = Path(path_text)
    if path.parent.resolve() != Path("/tmp"):
        raise ValueError(f"IPC socket must be directly under /tmp: {path}")
    if not path.exists() and not path.is_symlink():
        return
    if not stat.S_ISSOCK(path.lstat().st_mode):
        raise RuntimeError(f"refusing to remove non-socket IPC path: {path}")
    path.unlink()


def _byte_multiarray_payload(message: ByteMultiArray) -> bytes:
    output = bytearray()
    for item in message.data:
        if isinstance(item, int):
            output.append(item & 0xFF)
        elif isinstance(item, (bytes, bytearray, memoryview)):
            output.extend(bytes(item))
        elif isinstance(item, (tuple, list)):
            output.extend(int(value) & 0xFF for value in item)
        else:
            raise ValueError(f"unexpected ByteMultiArray item: {type(item)!r}")
    return bytes(output)


def _frame_name(value: str) -> str:
    return str(value).strip().lstrip("/")


class IsaacFlexivBridgeNode(Node):
    def __init__(self, config: BridgeConfig, *, enable_command: bool):
        super().__init__("isaac_flexiv_bridge")
        self.config = config
        self.command_enabled = bool(enable_command and config.command_enabled)
        self.mapper = DualArmAbsoluteMapper(config.mapping, config.safety)
        self.session_id = uuid.uuid4().hex
        self._protocol_sequence = 0
        self._pose_sequence = 0
        self._last_ingested_pose_sequence = -1
        self._latest_pose: _RawPosePair | None = None
        self._latest_pose_error = ""
        self._tf_receive_ns: dict[str, int] = {}
        self._controller_receive_ns = 0
        self._controller_data: dict[str, Any] = {}
        self._external_deadman_receive_ns = 0
        self._external_deadman = False
        self._last_heartbeat_ns = 0
        self._last_status_text = ""
        self._last_status_publish_ns = 0
        self._last_gateway_status: dict[str, Any] = {}
        self._last_gateway_instance_id = ""

        self._telemetry_socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        _safe_remove_socket(config.ipc.ros_socket)
        self._telemetry_socket.bind(config.ipc.ros_socket)
        self._telemetry_socket.setblocking(False)

        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        reliable_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            PoseArray, config.ros.ee_topic, self._on_ee_poses, sensor_qos
        )
        self.create_subscription(
            ByteMultiArray,
            config.ros.controller_topic,
            self._on_controller_data,
            sensor_qos,
        )
        self.create_subscription(
            TFMessage, config.ros.tf_topic, self._on_tf, sensor_qos
        )
        self.create_subscription(
            Bool,
            config.deadman.external_topic,
            self._on_external_deadman,
            reliable_qos,
        )

        self._status_publisher = self.create_publisher(
            String, "/isaac_flexiv/status", reliable_qos
        )
        self._gateway_status_publisher = self.create_publisher(
            String, "/isaac_flexiv/gateway_status", reliable_qos
        )
        self._diagnostics_publisher = self.create_publisher(
            DiagnosticArray, "/isaac_flexiv/diagnostics", reliable_qos
        )
        self._requested_publishers = {
            side: self.create_publisher(
                TwistStamped,
                f"/isaac_flexiv/requested/{side}_delta",
                reliable_qos,
            )
            for side in ("left", "right")
        }
        self._applied_publishers = {
            side: self.create_publisher(
                TwistStamped,
                f"/isaac_flexiv/applied/{side}_delta",
                reliable_qos,
            )
            for side in ("left", "right")
        }
        self._joint_publishers = {
            side: self.create_publisher(
                JointState, f"/flexiv/{side}/joint_states", reliable_qos
            )
            for side in ("left", "right")
        }
        self._tcp_publishers = {
            side: self.create_publisher(
                PoseStamped, f"/flexiv/{side}/tcp_pose", reliable_qos
            )
            for side in ("left", "right")
        }
        self._hand_publishers = {
            side: self.create_publisher(
                JointState, f"/inspire/{side}/actuator_states", reliable_qos
            )
            for side in ("left", "right")
        }
        self.create_timer(1.0 / config.ros.control_rate_hz, self._control_tick)
        self.get_logger().warning(
            "Bridge started in %s mode; session_id=%s",
            "COMMAND" if self.command_enabled else "SHADOW",
            self.session_id,
        )

    def _on_ee_poses(self, message: PoseArray) -> None:
        receive_ns = time.monotonic_ns()
        self._pose_sequence += 1
        try:
            max_index = max(
                self.config.mapping.left.pose_index,
                self.config.mapping.right.pose_index,
            )
            if len(message.poses) <= max_index:
                raise ValueError(
                    f"PoseArray has {len(message.poses)} poses; need index {max_index}"
                )
            poses: dict[str, Pose] = {}
            for side, side_config in (
                ("left", self.config.mapping.left),
                ("right", self.config.mapping.right),
            ):
                source = message.poses[side_config.pose_index]
                poses[side] = Pose.from_values(
                    [source.position.x, source.position.y, source.position.z],
                    [
                        source.orientation.x,
                        source.orientation.y,
                        source.orientation.z,
                        source.orientation.w,
                    ],
                )
            stamp = message.header.stamp
            source_stamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
            self._latest_pose = _RawPosePair(
                left=poses["left"],
                right=poses["right"],
                sequence=self._pose_sequence,
                receive_monotonic_ns=receive_ns,
                source_stamp_ns=source_stamp_ns,
                frame_id=_frame_name(message.header.frame_id),
            )
            self._latest_pose_error = ""
        except Exception as exc:
            self._latest_pose_error = f"invalid ee_poses: {exc}"
            if self.mapper.state in {
                BridgeState.ARMING,
                BridgeState.WAIT_ARM_ACK,
                BridgeState.ACTIVE,
            }:
                self.mapper.force_hold(self._latest_pose_error)

    def _on_tf(self, message: TFMessage) -> None:
        receive_ns = time.monotonic_ns()
        world = _frame_name(self.config.ros.world_frame)
        watched = {
            _frame_name(self.config.ros.left_wrist_frame),
            _frame_name(self.config.ros.right_wrist_frame),
        }
        for transform in message.transforms:
            child = _frame_name(transform.child_frame_id)
            parent = _frame_name(transform.header.frame_id)
            if child in watched and parent == world:
                self._tf_receive_ns[child] = receive_ns

    def _on_controller_data(self, message: ByteMultiArray) -> None:
        try:
            unpacked = msgpack.unpackb(
                _byte_multiarray_payload(message),
                raw=False,
                strict_map_key=False,
            )
            if not isinstance(unpacked, dict):
                raise ValueError("controller payload is not a mapping")
            self._controller_data = unpacked
            self._controller_receive_ns = time.monotonic_ns()
        except Exception as exc:
            self.get_logger().error("Cannot decode controller_data: %s", exc)
            if self.mapper.deadman_pressed:
                self.mapper.force_hold(f"controller_data decode failed: {exc}")

    def _on_external_deadman(self, message: Bool) -> None:
        self._external_deadman = bool(message.data)
        self._external_deadman_receive_ns = time.monotonic_ns()

    def _tracking_valid(self, now_ns: int) -> tuple[bool, str]:
        if self._latest_pose_error:
            return False, self._latest_pose_error
        if self._latest_pose is None:
            return False, "no ee_poses"
        if self._latest_pose.frame_id != _frame_name(self.config.ros.world_frame):
            return (
                False,
                f"unexpected ee frame {self._latest_pose.frame_id!r}; expected "
                f"{_frame_name(self.config.ros.world_frame)!r}",
            )
        max_age = self.config.safety.max_tf_age_s
        for side, frame in (
            ("left", self.config.ros.left_wrist_frame),
            ("right", self.config.ros.right_wrist_frame),
        ):
            receive_ns = self._tf_receive_ns.get(_frame_name(frame), 0)
            age_s = (now_ns - receive_ns) * 1e-9 if receive_ns else float("inf")
            if age_s > max_age:
                return False, f"{side} wrist TF stale or absent ({age_s:.4f}s)"
        return True, "valid"

    def _resolve_deadman(self, now_ns: int) -> tuple[bool, bool, str]:
        cfg = self.config.deadman
        max_age = self.config.safety.max_deadman_age_s
        if cfg.source == "external_bool":
            age_s = (
                (now_ns - self._external_deadman_receive_ns) * 1e-9
                if self._external_deadman_receive_ns
                else float("inf")
            )
            return (
                bool(self._external_deadman),
                age_s <= max_age,
                f"external_bool age={age_s:.4f}s",
            )
        age_s = (
            (now_ns - self._controller_receive_ns) * 1e-9
            if self._controller_receive_ns
            else float("inf")
        )
        left = float(self._controller_data.get("left_squeeze_value", 0.0))
        right = float(self._controller_data.get("right_squeeze_value", 0.0))
        if cfg.source == "quest_squeeze_both":
            pressed = left >= cfg.squeeze_threshold and right >= cfg.squeeze_threshold
        else:
            pressed = left >= cfg.squeeze_threshold or right >= cfg.squeeze_threshold
        return pressed, age_s <= max_age, (
            f"{cfg.source} left={left:.3f} right={right:.3f} age={age_s:.4f}s"
        )

    def _ingest_latest_pose(self, now_ns: int) -> None:
        raw = self._latest_pose
        if raw is None or raw.sequence == self._last_ingested_pose_sequence:
            return
        valid, _ = self._tracking_valid(now_ns)
        self.mapper.ingest_sample(
            PosePairSample(
                left=raw.left,
                right=raw.right,
                sequence=raw.sequence,
                receive_monotonic_ns=raw.receive_monotonic_ns,
                source_stamp_ns=raw.source_stamp_ns,
                frame_id=raw.frame_id,
                valid=valid,
            )
        )
        self._last_ingested_pose_sequence = raw.sequence

    def _drain_gateway_packets(self, now_ns: int) -> None:
        while True:
            try:
                data = self._telemetry_socket.recv(
                    self.config.ipc.max_packet_bytes
                )
            except BlockingIOError:
                return
            try:
                packet = decode_packet(
                    data, max_packet_bytes=self.config.ipc.max_packet_bytes
                )
                self._handle_gateway_packet(packet, now_ns)
            except Exception as exc:
                self.mapper.force_hold(f"invalid gateway packet: {exc}")

    def _handle_gateway_packet(self, packet: dict[str, Any], now_ns: int) -> None:
        instance_id = str(packet.get("gateway_instance_id", ""))
        if (
            self._last_gateway_instance_id
            and instance_id
            and instance_id != self._last_gateway_instance_id
            and self.mapper.state
            in {BridgeState.WAIT_ARM_ACK, BridgeState.ACTIVE}
        ):
            self.mapper.force_hold("hardware gateway process restarted")
        if instance_id:
            self._last_gateway_instance_id = instance_id
        kind = str(packet.get("kind", ""))
        if kind == "ack":
            if str(packet.get("session_id", "")) != self.session_id:
                return
            ack_kind = str(packet.get("ack_kind", ""))
            success = bool(packet.get("ok", False))
            reason = str(packet.get("reason", ""))
            if ack_kind == "arm":
                self.mapper.acknowledge_arm(
                    epoch=int(packet.get("epoch", -1)),
                    success=success,
                    now_ns=now_ns,
                    reason=reason,
                )
            elif ack_kind == "action":
                proposal_id = packet.get("proposal_id")
                if proposal_id is not None:
                    self.mapper.acknowledge_action(
                        epoch=int(packet.get("epoch", -1)),
                        proposal_id=int(proposal_id),
                        success=success,
                        now_ns=now_ns,
                        reason=reason,
                    )
            return
        if kind == "gateway_status":
            self._last_gateway_status = packet
            message = String()
            message.data = json.dumps(packet, ensure_ascii=False, sort_keys=True)
            self._gateway_status_publisher.publish(message)
            fault = str(packet.get("fault_reason", ""))
            if fault and self.mapper.state in {
                BridgeState.WAIT_ARM_ACK,
                BridgeState.ACTIVE,
            }:
                self.mapper.force_hold(f"gateway fault: {fault}")
            return
        if kind == "observation":
            self._publish_observation(packet.get("observation", {}))

    def _next_packet_sequence(self) -> int:
        self._protocol_sequence += 1
        return self._protocol_sequence

    def _send_gateway(
        self,
        *,
        kind: str,
        epoch: int,
        now_ns: int,
        proposal_id: int | None = None,
        action: dict[str, Any] | None = None,
        reason: str = "",
    ) -> None:
        packet: dict[str, Any] = {
            "kind": kind,
            "session_id": self.session_id,
            "sequence": self._next_packet_sequence(),
            "epoch": int(epoch),
            "sent_monotonic_ns": int(now_ns),
            "deadman": bool(self.mapper.deadman_pressed),
            "reason": str(reason),
        }
        if proposal_id is not None:
            packet["proposal_id"] = int(proposal_id)
        if action is not None:
            packet["action"] = action
        payload = encode_packet(packet)
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as client:
            client.sendto(payload, self.config.ipc.gateway_socket)

    def _handle_decision(self, decision: BridgeDecision, now_ns: int) -> None:
        if decision.requested_action is not None:
            self._publish_action(
                decision.requested_action, self._requested_publishers
            )
        if decision.applied_action is not None:
            self._publish_action(decision.applied_action, self._applied_publishers)

        if decision.kind == "arm":
            if self.command_enabled:
                try:
                    self._send_gateway(
                        kind="arm",
                        epoch=decision.epoch,
                        now_ns=now_ns,
                        reason=decision.reason,
                    )
                except Exception as exc:
                    self.mapper.acknowledge_arm(
                        epoch=decision.epoch,
                        success=False,
                        now_ns=now_ns,
                        reason=f"cannot reach gateway: {exc}",
                    )
            else:
                self.mapper.acknowledge_arm(
                    epoch=decision.epoch,
                    success=True,
                    now_ns=now_ns,
                    reason="shadow-mode synthetic arm ACK",
                )
            return
        if decision.kind == "action":
            assert decision.proposal_id is not None
            assert decision.applied_action is not None
            if self.command_enabled:
                try:
                    self._send_gateway(
                        kind="action",
                        epoch=decision.epoch,
                        now_ns=now_ns,
                        proposal_id=decision.proposal_id,
                        action=decision.applied_action,
                        reason=decision.reason,
                    )
                except Exception as exc:
                    self.mapper.acknowledge_action(
                        epoch=decision.epoch,
                        proposal_id=decision.proposal_id,
                        success=False,
                        now_ns=now_ns,
                        reason=f"cannot reach gateway: {exc}",
                    )
            else:
                self.mapper.acknowledge_action(
                    epoch=decision.epoch,
                    proposal_id=decision.proposal_id,
                    success=True,
                    now_ns=now_ns,
                    reason="shadow-mode synthetic action ACK",
                )
            return
        if decision.kind == "hold" and self.command_enabled:
            try:
                self._send_gateway(
                    kind="hold",
                    epoch=max(0, decision.epoch),
                    now_ns=now_ns,
                    reason=decision.reason,
                )
            except Exception as exc:
                self.get_logger().error("Cannot send gateway hold: %s", exc)

    def _heartbeat(self, now_ns: int) -> None:
        if (
            not self.command_enabled
            or self.mapper.state != BridgeState.ACTIVE
            or not self.mapper.deadman_pressed
        ):
            return
        period_ns = int(self.config.safety.gateway_watchdog_s * 0.35 * 1e9)
        if now_ns - self._last_heartbeat_ns < period_ns:
            return
        try:
            self._send_gateway(
                kind="heartbeat",
                epoch=self.mapper.epoch,
                now_ns=now_ns,
                reason="control-loop heartbeat",
            )
            self._last_heartbeat_ns = now_ns
        except Exception as exc:
            self.mapper.force_hold(f"cannot send gateway heartbeat: {exc}")

    def _control_tick(self) -> None:
        now_ns = time.monotonic_ns()
        self._drain_gateway_packets(now_ns)
        self._ingest_latest_pose(now_ns)

        deadman, deadman_fresh, deadman_detail = self._resolve_deadman(now_ns)
        if not deadman_fresh:
            if self.mapper.deadman_pressed:
                self.mapper.force_hold(f"deadman source stale: {deadman_detail}")
        else:
            self.mapper.set_deadman(deadman, now_ns)

        decision = self.mapper.tick(now_ns)
        if decision is not None:
            self._handle_decision(decision, now_ns)
        self._heartbeat(now_ns)
        self._publish_status(now_ns, deadman_detail)

    def _publish_action(
        self,
        action: dict[str, Any],
        publishers: dict[str, Any],
    ) -> None:
        left_dp, left_dr, right_dp, right_dr = action_vectors(action)
        stamp = self.get_clock().now().to_msg()
        for side, translation, rotation in (
            ("left", left_dp, left_dr),
            ("right", right_dp, right_dr),
        ):
            message = TwistStamped()
            message.header.stamp = stamp
            message.header.frame_id = self.config.ros.world_frame
            message.twist.linear.x = float(translation[0])
            message.twist.linear.y = float(translation[1])
            message.twist.linear.z = float(translation[2])
            message.twist.angular.x = float(rotation[0])
            message.twist.angular.y = float(rotation[1])
            message.twist.angular.z = float(rotation[2])
            publishers[side].publish(message)

    def _publish_observation(self, observation: Any) -> None:
        if not isinstance(observation, dict):
            return
        stamp = self.get_clock().now().to_msg()
        for side in ("left", "right"):
            joint_values = [
                float(observation.get(f"{side}_joint_{index}.pos", 0.0))
                for index in range(1, 8)
            ]
            joints = JointState()
            joints.header.stamp = stamp
            joints.header.frame_id = f"{side}_base"
            joints.name = [f"{side}_joint_{index}" for index in range(1, 8)]
            joints.position = joint_values
            self._joint_publishers[side].publish(joints)

            pose_values = np.array(
                [
                    float(observation.get(f"{side}_ee_pose.{axis}", 0.0))
                    for axis in ("x", "y", "z", "rx", "ry", "rz")
                ],
                dtype=float,
            )
            quaternion = rotvec_to_quat(pose_values[3:])
            pose = PoseStamped()
            pose.header.stamp = stamp
            pose.header.frame_id = self.config.ros.world_frame
            pose.pose.position.x = float(pose_values[0])
            pose.pose.position.y = float(pose_values[1])
            pose.pose.position.z = float(pose_values[2])
            pose.pose.orientation.x = float(quaternion[0])
            pose.pose.orientation.y = float(quaternion[1])
            pose.pose.orientation.z = float(quaternion[2])
            pose.pose.orientation.w = float(quaternion[3])
            self._tcp_publishers[side].publish(pose)

            hand = JointState()
            hand.header.stamp = stamp
            hand.header.frame_id = f"{side}_inspire_base"
            hand.name = [f"{side}_inspire_actuator_{index}" for index in range(6)]
            hand.position = [
                float(observation.get(f"{side}_hand_state_{index}", 0.0))
                for index in range(6)
            ]
            self._hand_publishers[side].publish(hand)

    def _publish_status(self, now_ns: int, deadman_detail: str) -> None:
        valid, tracking_detail = self._tracking_valid(now_ns)
        status = {
            "session_id": self.session_id,
            "mode": "command" if self.command_enabled else "shadow",
            "state": self.mapper.state.value,
            "reason": self.mapper.last_reason,
            "deadman_pressed": self.mapper.deadman_pressed,
            "deadman": deadman_detail,
            "tracking_valid": valid,
            "tracking": tracking_detail,
            "epoch": self.mapper.epoch,
            "pending_proposal_id": self.mapper.pending_proposal_id,
        }
        text = json.dumps(status, ensure_ascii=False, sort_keys=True)
        if (
            text == self._last_status_text
            and now_ns - self._last_status_publish_ns < 500_000_000
        ):
            return
        self._last_status_text = text
        self._last_status_publish_ns = now_ns
        message = String()
        message.data = text
        self._status_publisher.publish(message)

        diagnostic = DiagnosticArray()
        diagnostic.header.stamp = self.get_clock().now().to_msg()
        item = DiagnosticStatus()
        item.name = "isaac_flexiv_bridge"
        item.hardware_id = "flexiv_dual_rizon4s_inspire"
        item.level = (
            DiagnosticStatus.ERROR
            if self.mapper.state == BridgeState.HOLD_LATCHED
            else DiagnosticStatus.OK
        )
        item.message = self.mapper.last_reason
        item.values = [
            KeyValue(key=str(key), value=str(value))
            for key, value in status.items()
        ]
        diagnostic.status = [item]
        self._diagnostics_publisher.publish(diagnostic)

    def close(self) -> None:
        if self.command_enabled:
            try:
                now_ns = time.monotonic_ns()
                self._send_gateway(
                    kind="hold",
                    epoch=max(0, self.mapper.epoch),
                    now_ns=now_ns,
                    reason="ROS bridge shutdown",
                )
            except Exception:
                pass
        self._telemetry_socket.close()
        try:
            _safe_remove_socket(self.config.ipc.ros_socket)
        except Exception:
            pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--enable-command",
        action="store_true",
        help="requires command_enabled: true in the new config as a second gate",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ros_args = sys.argv if argv is None else [sys.argv[0], *argv]
    clean_args = rclpy.utilities.remove_ros_args(args=ros_args)[1:]
    args = _build_parser().parse_args(clean_args)
    config = load_config(args.config)
    rclpy.init(args=ros_args)
    node: IsaacFlexivBridgeNode | None = None
    try:
        node = IsaacFlexivBridgeNode(
            config, enable_command=bool(args.enable_command)
        )
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

