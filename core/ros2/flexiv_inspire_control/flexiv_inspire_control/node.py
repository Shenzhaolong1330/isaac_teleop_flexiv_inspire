"""Independent ROS 2 control/observation bridge.

Startup is intentionally motionless: local permission, collision-clear, foot
pedal, source heartbeat, a local control-arm token and per-session F/T zero are
all absent by default.
"""

from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import threading
import time
import uuid

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration, Time
from geometry_msgs.msg import PoseStamped, TwistStamped, WrenchStamped
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Empty, String, UInt64
from rosidl_runtime_py.convert import message_to_ordereddict

from flexiv_inspire_interfaces.action import ZeroFTSensors
from flexiv_inspire_interfaces.msg import (
    ArmState as ArmStateMsg,
    BimanualCommand as BimanualCommandMsg,
    CommandTrace,
    ControlState as ControlStateMsg,
    ForceTorqueStatistics,
    HandState,
)
from isaac_teleop_core.command import CommandSource, ValidMask
from isaac_teleop_core.deviceio import AsyncDeviceIOEmitter, record_envelope
from isaac_teleop_core.control import (
    ControlArbiter,
    ControlState,
    GateInputs,
    TransitionError,
)

from .clock_mapper import OnlineClockMapper
from .conversion import cartesian_target_from_point, command_from_ros
from .foot_pedal import FootPedalMonitor
from .ipc_client import RDKIPCClient


def _time_from_ns(value: int) -> Time:
    message = Time()
    message.sec = int(value // 1_000_000_000)
    message.nanosec = int(value % 1_000_000_000)
    return message


def _duration_from_ns(value: int) -> Duration:
    message = Duration()
    message.sec = int(value // 1_000_000_000)
    message.nanosec = int(value % 1_000_000_000)
    return message


def _require_matching_ft_zero_epoch(
    *,
    result_generation: int,
    result_instance_id: str,
    current_generation: int | None,
    current_instance_id: str,
) -> None:
    """Reject an F/T result if the observed RDK connection changed."""

    if current_generation is None:
        raise RuntimeError("current RDK connection generation is unavailable")
    if not current_instance_id:
        raise RuntimeError("current RDK daemon instance is unavailable")
    if result_instance_id != current_instance_id:
        raise RuntimeError("daemon instance changed during F/T zero")
    if result_generation != current_generation:
        raise RuntimeError("RDK connection generation changed during F/T zero")


class ControlBridge(Node):
    def __init__(self) -> None:
        super().__init__("flexiv_inspire_control_bridge")
        runtime = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"))
        self.declare_parameter("session_id", "")
        self.declare_parameter("rdk_socket", str(runtime / "isaac_teleop" / "rdk.sock"))
        self.declare_parameter(
            "foot_pedal",
            "/dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd",
        )
        self.declare_parameter("observation_rate_hz", 200.0)
        self.declare_parameter("max_translation_step_m", 0.01)
        self.declare_parameter("max_rotation_step_rad", 0.10)
        self.declare_parameter("max_linear_velocity_m_s", 0.05)
        self.declare_parameter("max_angular_velocity_rad_s", 0.15)
        self.declare_parameter("max_linear_acceleration_m_s2", 0.25)
        self.declare_parameter("max_angular_acceleration_rad_s2", 0.50)
        self.declare_parameter("joint_lower_limits_rad", [])
        self.declare_parameter("joint_upper_limits_rad", [])
        self.declare_parameter("max_joint_velocity_rad_s", 0.75)
        self.declare_parameter("max_tcp_linear_speed_m_s", 0.35)
        self.declare_parameter("max_tcp_angular_speed_rad_s", 1.0)
        self.declare_parameter("max_external_force_n", 60.0)
        self.declare_parameter("max_external_torque_nm", 8.0)
        self.declare_parameter("max_joint_temperature_c", 75.0)
        self.declare_parameter("hand_reference_tolerance", 1.0)
        session = str(self.get_parameter("session_id").value).strip()
        self._session_id = session or f"hardware-{uuid.uuid4()}"
        socket_path = Path(str(self.get_parameter("rdk_socket").value))
        self._ipc_observation = RDKIPCClient(socket_path)
        self._ipc_command = RDKIPCClient(socket_path)
        self._ipc_maintenance = RDKIPCClient(socket_path)
        self._ipc_hand = RDKIPCClient(socket_path)
        self._arbiter = ControlArbiter()
        self._arbiter.begin_hardware_session(self._session_id)
        self._state_lock = threading.RLock()
        self._stop = threading.Event()
        self._poll_thread: threading.Thread | None = None
        self._chunk_generation = 0
        self._chunk_lock = threading.Lock()
        self._latest_wire: dict[str, dict] = {}
        self._safe_pose_rdk: dict[str, np.ndarray] = {}
        self._previous_output_quaternion: dict[str, np.ndarray | None] = {
            "left": None,
            "right": None,
        }
        self._connection_generation: int | None = None
        self._daemon_instance_id = ""
        self._clock_mappers = {
            "left": OnlineClockMapper(), "right": OnlineClockMapper()
        }
        self._last_arm_observation_ns = {"left": 0, "right": 0}
        self._last_hand_observation_ns = {"left": 0, "right": 0}
        self._hand_connected = {"left": False, "right": False}
        self._hand_angles: dict[str, np.ndarray | None] = {
            "left": None, "right": None
        }
        self._arm_safety_ok = {"left": False, "right": False}
        self._local_permission = False
        self._physical_pedal = False
        self._collision_clear = False
        self._limits_ok = False
        self._pending_arm_token: str | None = None
        self._pending_arm_token_expiry_ns = 0
        self._pending_arm_source: CommandSource | None = None
        self._rdk_control_lease_active = False
        self._hold_sent_for_latch = False
        self._hold_request_inflight = False
        self._zero_goal_reserved = False
        self._deviceio = AsyncDeviceIOEmitter("control")

        command_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._requested_pub = self.create_publisher(
            BimanualCommandMsg, "/control/requested_command", command_qos
        )
        self._safe_pub = self.create_publisher(
            BimanualCommandMsg, "/control/safe_command", command_qos
        )
        self._sent_pub = self.create_publisher(
            BimanualCommandMsg, "/control/sent_command", command_qos
        )
        self._trace_pub = self.create_publisher(
            CommandTrace, "/control/command_trace", command_qos
        )
        self._control_state_pub = self.create_publisher(
            ControlStateMsg, "/control/state", command_qos
        )
        self._arm_publishers = {
            side: self._create_arm_publishers(side) for side in ("left", "right")
        }

        for source in CommandSource:
            self.create_subscription(
                BimanualCommandMsg,
                f"/command_sources/{source.value}/command",
                lambda message, selected=source: self._on_command(selected, message),
                command_qos,
            )
            self.create_subscription(
                UInt64,
                f"/command_sources/{source.value}/heartbeat",
                lambda message, selected=source: self._on_heartbeat(selected),
                qos_profile_sensor_data,
            )
        for side in ("left", "right"):
            self.create_subscription(
                HandState,
                f"/robot/{side}_hand/state",
                lambda message, selected=side: self._on_hand_state(selected, message),
                qos_profile_sensor_data,
            )
        self.create_subscription(Bool, "/control/local_permission", self._on_local_permission, command_qos)
        self.create_subscription(Bool, "/safety/collision_clear", self._on_collision_clear, command_qos)
        self.create_subscription(String, "/control/local_arm_authorization", self._on_arm_authorization, command_qos)
        self.create_subscription(Empty, "/control/stop", lambda message: self._stop_control(), command_qos)
        self._zero_action = ActionServer(
            self,
            ZeroFTSensors,
            "/maintenance/zero_ft_sensors",
            execute_callback=self._execute_zero_ft,
            goal_callback=self._zero_goal,
            cancel_callback=lambda request: CancelResponse.REJECT,
        )
        self._watchdog_timer = self.create_timer(0.01, self._watchdog_tick)
        self._pedal = FootPedalMonitor(
            Path(str(self.get_parameter("foot_pedal").value)),
            self._on_pedal_state,
        )
        self._pedal.start()
        self._poll_thread = threading.Thread(
            target=self._observation_loop,
            name="rdk-observation-bridge",
            daemon=True,
        )
        self._poll_thread.start()
        self._publish_control_state()

    def _create_arm_publishers(self, side: str) -> dict[str, object]:
        prefix = f"/robot/{side}_arm"
        return {
            "state": self.create_publisher(ArmStateMsg, f"{prefix}/state", qos_profile_sensor_data),
            "joint": self.create_publisher(JointState, f"{prefix}/joint_states", qos_profile_sensor_data),
            "pose": self.create_publisher(PoseStamped, f"{prefix}/tcp_pose", qos_profile_sensor_data),
            "twist": self.create_publisher(TwistStamped, f"{prefix}/tcp_twist", qos_profile_sensor_data),
            "raw_ft": self.create_publisher(WrenchStamped, f"{prefix}/raw_ft", qos_profile_sensor_data),
            "wrench": self.create_publisher(WrenchStamped, f"{prefix}/tcp_wrench", qos_profile_sensor_data),
        }

    def destroy_node(self) -> bool:
        self._stop.set()
        self._pedal.close()
        for client in (self._ipc_observation, self._ipc_command, self._ipc_maintenance, self._ipc_hand):
            client.close()
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=1.0)
        self._zero_action.destroy()
        self._deviceio.close()
        return super().destroy_node()

    def _observation_loop(self) -> None:
        observation_rate_hz = float(self.get_parameter("observation_rate_hz").value)
        if not 1.0 <= observation_rate_hz <= 1000.0:
            raise RuntimeError("observation_rate_hz must be in [1,1000]")
        period = 1.0 / observation_rate_hz
        deadline = time.monotonic()
        while not self._stop.is_set():
            try:
                kind, payload = self._ipc_observation.request("observe", {})
                if kind != "dual_arm_state":
                    raise RuntimeError(f"unexpected RDK response {kind}")
                now = time.monotonic_ns()
                instance_id = str(payload.get("daemon_instance_id", ""))
                with self._state_lock:
                    if self._daemon_instance_id and instance_id != self._daemon_instance_id:
                        self._arbiter.on_rdk_reconnect()
                        self._safe_pose_rdk.clear()
                        self._pending_arm_token = None
                        self._pending_arm_token_expiry_ns = 0
                        self._rdk_control_lease_active = False
                        for mapper in self._clock_mappers.values():
                            mapper.reset()
                    self._daemon_instance_id = instance_id
                for side in ("left", "right"):
                    self._publish_arm(side, payload[side], now)
            except Exception as exc:
                with self._state_lock:
                    self._last_arm_observation_ns = {"left": 0, "right": 0}
                    self._limits_ok = False
                    self._arm_safety_ok["left"] = False
                    self._arm_safety_ok["right"] = False
                self.get_logger().error(f"RDK observation failed: {exc}", throttle_duration_sec=1.0)
            deadline += period
            delay = deadline - time.monotonic()
            if delay > 0:
                self._stop.wait(delay)
            else:
                deadline = time.monotonic()

    def _publish_arm(self, side: str, wire: dict, host_receive_ns: int) -> None:
        generation = int(wire.get("connection_generation", "0"))
        with self._state_lock:
            if self._connection_generation is None:
                self._connection_generation = generation
            elif generation != self._connection_generation:
                self._connection_generation = generation
                self._arbiter.on_rdk_reconnect()
                self._safe_pose_rdk.clear()
                self._pending_arm_token = None
                self._pending_arm_token_expiry_ns = 0
                self._rdk_control_lease_active = False
                for mapper in self._clock_mappers.values():
                    mapper.reset()
            self._latest_wire[side] = wire
            self._last_arm_observation_ns[side] = host_receive_ns
            pose_rdk = np.asarray(wire["tcp_pose_rdk_xyz_wxyz"], dtype=float)
            if self._arbiter.snapshot.state is not ControlState.ACTIVE:
                self._safe_pose_rdk[side] = pose_rdk.copy()

        robot_time_ns = int(wire.get("robot_time_ns", "0"))
        host_monotonic_ns = int(wire.get("host_receive_monotonic_ns", "0"))
        host_unix_ns = int(wire.get("host_receive_unix_ns", "0")) or time.time_ns()
        mapping = self._clock_mappers[side].update(
            robot_time_ns, host_monotonic_ns, host_unix_ns
        )
        mapped_unix_ns = (
            mapping.mapped_host_unix_ns if mapping.timing_valid else host_unix_ns
        )
        stamp = _time_from_ns(mapped_unix_ns)
        source_stamp = _time_from_ns(robot_time_ns)
        safe, safety_reason = self._arm_wire_safe(wire)
        with self._state_lock:
            self._arm_safety_ok[side] = safe
            self._limits_ok = all(self._arm_safety_ok.values())
        state = ArmStateMsg()
        state.header.stamp = stamp
        state.header.frame_id = "world"
        state.side = side
        state.robot_time_sec = int(wire.get("robot_time_sec", "0"))
        state.robot_time_nsec = int(wire.get("robot_time_nsec", 0))
        state.robot_clock_domain = str(wire.get("clock_domain", "flexiv_controller"))
        state.acquisition.source_time = source_stamp
        # Header stamps are mapped into ROS/Unix time for normal ROS consumers,
        # while AcquisitionInfo remains entirely in the monotonic domain.
        state.acquisition.host_receive_time = _time_from_ns(host_monotonic_ns)
        state.acquisition.acquisition_start = state.acquisition.host_receive_time
        state.acquisition.acquisition_end = state.acquisition.host_receive_time
        state.acquisition.source_sequence = robot_time_ns
        state.acquisition.valid = (
            bool(wire.get("connected", False))
            and not wire.get("fault", "")
            and safe
        )
        state.acquisition.age = _duration_from_ns(
            max(0, host_monotonic_ns - mapping.mapped_host_monotonic_ns)
            if mapping.timing_valid else 0
        )
        state.acquisition.invalid_reason = (
            str(wire.get("fault", "")) or safety_reason
        )
        state.acquisition.source_clock_domain = state.robot_clock_domain
        state.acquisition.host_clock_domain = "host_monotonic"
        state.acquisition.mapped_host_time = (
            _time_from_ns(mapping.mapped_host_monotonic_ns)
            if mapping.timing_valid else Time()
        )
        state.acquisition.timing_valid = mapping.timing_valid
        for field in ("q", "dq", "tau", "tau_des", "tau_ext", "tau_interact", "temperature"):
            setattr(state, field, [float(value) for value in wire.get(field, [])])
        state.connected = bool(wire.get("connected", False))
        state.fault = str(wire.get("fault", ""))
        state.rdk_connection_generation = generation
        self._fill_pose(state.tcp_pose, wire["tcp_pose_rdk_xyz_wxyz"])
        self._fill_twist(state.tcp_twist, wire["tcp_velocity"])
        self._fill_wrench(state.raw_ft, wire["raw_ft"])
        self._fill_wrench(state.tcp_wrench, wire["external_wrench"])
        pubs = self._arm_publishers[side]
        pubs["state"].publish(state)

        joints = JointState()
        joints.header = state.header
        joints.name = [f"{side}_joint_{index + 1}" for index in range(7)]
        joints.position = state.q
        joints.velocity = state.dq
        joints.effort = state.tau
        pubs["joint"].publish(joints)
        pose = PoseStamped()
        pose.header = state.header
        pose.pose = state.tcp_pose
        pubs["pose"].publish(pose)
        twist = TwistStamped()
        twist.header = state.header
        twist.twist = state.tcp_twist
        pubs["twist"].publish(twist)
        raw = WrenchStamped()
        raw.header = state.header
        raw.wrench = state.raw_ft
        pubs["raw_ft"].publish(raw)
        wrench = WrenchStamped()
        wrench.header = state.header
        wrench.wrench = state.tcp_wrench
        pubs["wrench"].publish(wrench)

    @staticmethod
    def _fill_pose(output, values) -> None:
        output.position.x, output.position.y, output.position.z = map(float, values[:3])
        output.orientation.w = float(values[3])
        output.orientation.x = float(values[4])
        output.orientation.y = float(values[5])
        output.orientation.z = float(values[6])

    @staticmethod
    def _fill_twist(output, values) -> None:
        output.linear.x, output.linear.y, output.linear.z = map(float, values[:3])
        output.angular.x, output.angular.y, output.angular.z = map(float, values[3:])

    @staticmethod
    def _fill_wrench(output, values) -> None:
        output.force.x, output.force.y, output.force.z = map(float, values[:3])
        output.torque.x, output.torque.y, output.torque.z = map(float, values[3:])

    def _arm_wire_safe(self, wire: dict) -> tuple[bool, str]:
        try:
            if not bool(wire.get("connected", False)):
                return False, "arm_disconnected"
            if str(wire.get("fault", "")):
                return False, f"arm_fault:{wire['fault']}"
            q = np.asarray(wire["q"], dtype=np.float64)
            dq = np.asarray(wire["dq"], dtype=np.float64)
            tcp = np.asarray(wire["tcp_velocity"], dtype=np.float64)
            wrench = np.asarray(wire["external_wrench"], dtype=np.float64)
            temperature = np.asarray(wire["temperature"], dtype=np.float64)
            if (
                q.shape != (7,)
                or dq.shape != (7,)
                or tcp.shape != (6,)
                or wrench.shape != (6,)
                or temperature.size == 0
                or not all(
                    np.all(np.isfinite(value))
                    for value in (q, dq, tcp, wrench, temperature)
                )
            ):
                return False, "nonfinite_or_malformed_arm_observation"
            lower = np.asarray(
                self.get_parameter("joint_lower_limits_rad").value,
                dtype=np.float64,
            )
            upper = np.asarray(
                self.get_parameter("joint_upper_limits_rad").value,
                dtype=np.float64,
            )
            if lower.shape != (7,) or upper.shape != (7,) or np.any(lower >= upper):
                return False, "audited_joint_limits_not_configured"
            if np.any(q < lower) or np.any(q > upper):
                return False, "joint_position_limit"
            if np.max(np.abs(dq)) > float(
                self.get_parameter("max_joint_velocity_rad_s").value
            ):
                return False, "joint_velocity_limit"
            if np.linalg.norm(tcp[:3]) > float(
                self.get_parameter("max_tcp_linear_speed_m_s").value
            ):
                return False, "tcp_linear_speed_limit"
            if np.linalg.norm(tcp[3:]) > float(
                self.get_parameter("max_tcp_angular_speed_rad_s").value
            ):
                return False, "tcp_angular_speed_limit"
            if np.linalg.norm(wrench[:3]) > float(
                self.get_parameter("max_external_force_n").value
            ):
                return False, "external_force_limit"
            if np.linalg.norm(wrench[3:]) > float(
                self.get_parameter("max_external_torque_nm").value
            ):
                return False, "external_torque_limit"
            if np.max(temperature) > float(
                self.get_parameter("max_joint_temperature_c").value
            ):
                return False, "joint_temperature_limit"
            return True, ""
        except Exception as exc:
            return False, f"arm_safety_check_failed:{type(exc).__name__}"

    def _on_command(self, source: CommandSource, message: BimanualCommandMsg) -> None:
        receive_ns = time.monotonic_ns()
        if not self._emit_deviceio(
            "/control/requested_command", message, receive_ns, critical=True
        ):
            self._arbiter.reject_invalid_command(source, now_monotonic_ns=receive_ns)
            self._send_hold_once("deviceio_requested_command_not_recorded")
            self._publish_control_state()
            return
        self._requested_pub.publish(message)
        try:
            command = command_from_ros(
                message,
                expected_source=source,
                received_monotonic_ns=receive_ns,
            )
            self._validate_hand_targets(command)
            self._update_gates(receive_ns)
            self._arbiter.submit(command, now_monotonic_ns=receive_ns)
        except Exception as exc:
            self._arbiter.reject_invalid_command(source, now_monotonic_ns=receive_ns)
            # Teleop emits an empty-mask packet when its physical deadman is
            # released. Envelope validation rejects that packet by design, but
            # the release must still be observed after latching so a local
            # operator can acknowledge and clear HOLD_LATCHED.
            if not bool(message.deadman):
                self._arbiter.observe_deadman_released(source)
            self._publish_trace(message, None, None, str(exc), receive_ns)
            self._publish_control_state()
            return
        with self._chunk_lock:
            self._chunk_generation += 1
            generation = self._chunk_generation
        threading.Thread(
            target=self._execute_chunk,
            args=(generation, command, message, receive_ns),
            daemon=True,
            name=f"command-chunk-{source.value}-{command.sequence}",
        ).start()

    def _execute_chunk(
        self,
        generation: int,
        command,
        message: BimanualCommandMsg,
        receive_ns: int,
    ) -> None:
        for index, point in enumerate(command.points):
            safe_message = None
            deadline = receive_ns + int(point.execute_after_s * 1e9)
            while time.monotonic_ns() < deadline:
                with self._chunk_lock:
                    if generation != self._chunk_generation:
                        return
                if self._arbiter.snapshot.state is not ControlState.ACTIVE:
                    return
                time.sleep(0.001)
            try:
                now = time.monotonic_ns()
                with self._chunk_lock:
                    if generation != self._chunk_generation:
                        return
                # Every scheduled point is revalidated immediately before send.
                self._update_gates(now)
                snapshot = self._arbiter.tick(now_monotonic_ns=now)
                if snapshot.state is not ControlState.ACTIVE:
                    raise RuntimeError(
                        f"scheduled point gate rejected: {snapshot.state.value}"
                    )
                remaining_ttl_ns = int(command.expires_monotonic_ns - now)
                if remaining_ttl_ns <= 0:
                    raise RuntimeError("scheduled point expired before execution")
                if command.sequence > ((1 << 64) - 1) >> 6:
                    raise ValueError("source sequence is too large for chunk expansion")
                internal_sequence = (int(command.sequence) << 6) | index
                point_message = self._single_point_message(
                    message,
                    index,
                    internal_sequence,
                    stamp=self.get_clock().now().to_msg(),
                    remaining_ttl_ns=remaining_ttl_ns,
                )
                targets, candidate_poses, candidate_quaternions = (
                    self._targets_for_point(command, point)
                )
                payload = None
                if targets:
                    with self._state_lock:
                        token = self._pending_arm_token or ""
                        token_expiry = self._pending_arm_token_expiry_ns
                        lease_active = self._rdk_control_lease_active
                    if not lease_active and (not token or now >= token_expiry):
                        raise RuntimeError("local control authorization is absent or expired")
                    payload = {
                        "session_id": command.session_id,
                        "source": command.source.value,
                        "source_sequence": str(internal_sequence),
                        "expires_monotonic_ns": str(command.expires_monotonic_ns),
                        "valid_mask": int(command.valid_mask & (ValidMask.LEFT_ARM | ValidMask.RIGHT_ARM)),
                        "safety_validated": True,
                        "local_permission": self._local_permission,
                        "physical_pedal": self._physical_pedal,
                        "local_arm_token": token,
                        **targets,
                    }
                # A safe command has passed every bridge-side safety check and
                # local authorization gate and is ready for its hardware
                # boundary. It does not imply that the RDK accepted it.
                if not self._emit_deviceio(
                    "/control/safe_command", point_message, now, critical=True
                ):
                    raise RuntimeError("safe command could not be recorded")
                self._safe_pub.publish(point_message)
                safe_message = point_message
                if targets:
                    kind, response = self._ipc_command.request("cartesian_command", payload)
                    if kind != "command_ack" or not response.get("accepted", False):
                        raise RuntimeError(f"RDK rejected command: {response.get('reason', kind)}")
                    # The token establishes a daemon lease. Later points remain
                    # bound to this PID/session/source and do not reuse it.
                    with self._state_lock:
                        # Commit both candidate frames atomically only after the
                        # daemon accepted the complete dual-arm transaction.
                        self._safe_pose_rdk.update(candidate_poses)
                        self._previous_output_quaternion.update(
                            candidate_quaternions
                        )
                        self._rdk_control_lease_active = True
                        self._pending_arm_token = None
                        self._pending_arm_token_expiry_ns = 0
                self._arbiter.mark_sent(command)
                if not self._emit_deviceio(
                    "/control/sent_command", point_message, time.monotonic_ns(), critical=True
                ):
                    raise RuntimeError("sent command acknowledgement could not be recorded")
                self._sent_pub.publish(point_message)
                self._publish_trace(point_message, point_message, point_message, "", receive_ns)
            except Exception as exc:
                if safe_message is None:
                    # A point rejected before the hardware boundary is an
                    # invalid active-chain command and enters recoverable,
                    # latched hold. A point rejected after publication as safe
                    # crossed the local boundary and is an execution fault.
                    self._arbiter.reject_invalid_command(
                        command.source,
                        now_monotonic_ns=time.monotonic_ns(),
                    )
                else:
                    self._arbiter.fault()
                with self._state_lock:
                    # Never retain an unacknowledged target as the next delta
                    # origin. Observation polling will rebase from measurement.
                    self._safe_pose_rdk.clear()
                    self._previous_output_quaternion = {
                        "left": None, "right": None
                    }
                    self._rdk_control_lease_active = False
                    self._pending_arm_source = None
                    self._pending_arm_token = None
                    self._pending_arm_token_expiry_ns = 0
                self._send_hold_once(f"send_failure:{exc}")
                self._publish_trace(message, safe_message, None, str(exc), receive_ns)
                self._publish_control_state()
                return

    @staticmethod
    def _single_point_message(
        message: BimanualCommandMsg,
        index: int,
        internal_sequence: int,
        *,
        stamp: Time,
        remaining_ttl_ns: int,
    ) -> BimanualCommandMsg:
        if remaining_ttl_ns <= 0 or remaining_ttl_ns > 1_000_000_000:
            raise ValueError("remaining TTL must be in (0,1 second]")
        output = copy.deepcopy(message)
        output.sequence = internal_sequence
        point = copy.deepcopy(message.trajectory[index])
        point.execute_after.sec = 0
        point.execute_after.nanosec = 0
        output.trajectory = [point]
        output.header.stamp = stamp
        output.ttl = _duration_from_ns(remaining_ttl_ns)
        return output

    def _targets_for_point(
        self, command, point
    ) -> tuple[dict[str, dict], dict[str, np.ndarray], dict[str, np.ndarray]]:
        result: dict[str, dict] = {}
        candidate_poses: dict[str, np.ndarray] = {}
        candidate_quaternions: dict[str, np.ndarray] = {}
        for side, bit in (("left", ValidMask.LEFT_ARM), ("right", ValidMask.RIGHT_ARM)):
            if not command.valid_mask & bit:
                continue
            with self._state_lock:
                previous = self._safe_pose_rdk.get(side)
                previous_quaternion = self._previous_output_quaternion[side]
            if previous is None:
                raise RuntimeError(f"no valid {side} measured pose for rebase")
            target, output_quaternion = cartesian_target_from_point(
                point,
                side=side,
                representation=command.representation,
                previous_safe_pose_rdk=previous,
                max_translation_step_m=float(self.get_parameter("max_translation_step_m").value),
                max_rotation_step_rad=float(self.get_parameter("max_rotation_step_rad").value),
                previous_output_quaternion_xyzw=previous_quaternion,
            )
            candidate_poses[side] = target.copy()
            candidate_quaternions[side] = output_quaternion.copy()
            result[side] = {
                "tcp_pose_rdk": target.tolist(),
                "max_linear_velocity": float(self.get_parameter("max_linear_velocity_m_s").value),
                "max_angular_velocity": float(self.get_parameter("max_angular_velocity_rad_s").value),
                "max_linear_acceleration": float(self.get_parameter("max_linear_acceleration_m_s2").value),
                "max_angular_acceleration": float(self.get_parameter("max_angular_acceleration_rad_s2").value),
            }
        return result, candidate_poses, candidate_quaternions

    @staticmethod
    def _validate_hand_targets(command) -> None:
        for point in command.points:
            for name, values, bit in (
                ("left", point.left_hand_targets, ValidMask.LEFT_HAND),
                ("right", point.right_hand_targets, ValidMask.RIGHT_HAND),
            ):
                if command.valid_mask & bit and (
                    np.any(values < 0.0) or np.any(values > 1000.0)
                ):
                    raise ValueError(f"{name} hand target is outside [0,1000]")

    def _on_heartbeat(self, source: CommandSource) -> None:
        self._arbiter.heartbeat(source, now_monotonic_ns=time.monotonic_ns())

    def _on_hand_state(self, side: str, message: HandState) -> None:
        now = time.monotonic_ns()
        angle = np.asarray(message.angle, dtype=np.float64)
        valid = (
            bool(message.connected)
            and not message.fault
            and angle.shape == (6,)
            and np.all(np.isfinite(angle))
        )
        with self._state_lock:
            self._hand_connected[side] = valid
            self._last_hand_observation_ns[side] = now
            self._hand_angles[side] = angle.copy() if valid else None
        if not valid:
            self._update_gates(now)
            return
        source_time_ns = (
            int(message.acquisition.source_time.sec) * 1_000_000_000
            + int(message.acquisition.source_time.nanosec)
        )
        try:
            kind, response = self._ipc_hand.request(
                "hand_observation",
                {
                    "left": self._current_hand_angles("left").tolist(),
                    "right": self._current_hand_angles("right").tolist(),
                    "host_receive_monotonic_ns": str(now),
                    "source_sequence": str(message.sequence),
                    "source_time_ns": str(source_time_ns),
                    "source_clock_domain": str(
                        message.acquisition.source_clock_domain
                    ),
                },
            )
            if kind != "command_ack" or not response.get("accepted", False):
                raise RuntimeError(response.get("reason", kind))
        except Exception as exc:
            with self._state_lock:
                self._hand_connected[side] = False
            self.get_logger().error(f"hand observation IPC failed: {exc}")
        self._update_gates(now)

    def _current_hand_angles(self, side: str) -> np.ndarray:
        with self._state_lock:
            value = self._hand_angles[side]
            received = self._last_hand_observation_ns[side]
        if value is None or time.monotonic_ns() - received > 100_000_000:
            raise RuntimeError(f"{side} hand measurement is absent or stale")
        return value.copy()

    def _on_local_permission(self, message: Bool) -> None:
        self._local_permission = bool(message.data)
        self._update_gates(time.monotonic_ns())

    def _on_collision_clear(self, message: Bool) -> None:
        self._collision_clear = bool(message.data)
        self._update_gates(time.monotonic_ns())

    def _on_pedal_state(self, pressed: bool) -> None:
        self._physical_pedal = pressed
        self._update_gates(time.monotonic_ns())

    def _on_arm_authorization(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
            if payload["session_id"] != self._session_id:
                raise ValueError("authorization session mismatch")
            source = CommandSource(payload["source"])
            token = str(payload["one_time_token"])
            expires = int(payload["expires_monotonic_ns"])
            if not token:
                raise ValueError("empty authorization token")
            if expires <= time.monotonic_ns():
                raise ValueError("authorization token is already expired")
            if self._arbiter.snapshot.state is ControlState.HOLD_LATCHED:
                self._arbiter.clear_hold(local_acknowledged=True)
            self._arbiter.arm(source)
            self._pending_arm_source = source
            self._pending_arm_token = token
            self._pending_arm_token_expiry_ns = expires
            self._rdk_control_lease_active = False
            self._hold_sent_for_latch = False
        except Exception as exc:
            self.get_logger().error(f"control authorization rejected: {exc}")
        self._publish_control_state()

    def _update_gates(self, now: int) -> None:
        hands_online = all(
            self._hand_connected[side]
            and now - self._last_hand_observation_ns[side] <= 200_000_000
            for side in ("left", "right")
        )
        arms_online = all(
            self._last_arm_observation_ns[side] > 0
            and now - self._last_arm_observation_ns[side] <= 100_000_000
            and bool(self._latest_wire.get(side, {}).get("connected", False))
            and not str(self._latest_wire.get(side, {}).get("fault", ""))
            for side in ("left", "right")
        )
        hardware_fault = any(
            bool(str(self._latest_wire.get(side, {}).get("fault", "")))
            for side in ("left", "right")
        )
        self._arbiter.update_gates(
            GateInputs(
                local_permission=self._local_permission,
                physical_pedal=self._physical_pedal,
                arms_online=arms_online,
                hands_online=hands_online,
                limits_ok=self._limits_ok,
                collision_clear=self._collision_clear,
                hardware_fault=hardware_fault,
            ),
            now_monotonic_ns=now,
        )

    def _watchdog_tick(self) -> None:
        now = time.monotonic_ns()
        self._update_gates(now)
        snapshot = self._arbiter.tick(now_monotonic_ns=now)
        if snapshot.state in {ControlState.HOLD_LATCHED, ControlState.FAULT}:
            self._send_hold_once(snapshot.hold_reason.value)
        self._publish_control_state()

    def _send_hold_once(self, reason: str) -> None:
        with self._state_lock:
            if self._hold_sent_for_latch or self._hold_request_inflight:
                return
            self._hold_request_inflight = True
        try:
            kind, response = self._ipc_command.request(
                "hold", {"reason": reason, "latch": True}
            )
            if kind != "command_ack" or not response.get("accepted", False):
                raise RuntimeError(response.get("reason", kind))
            with self._state_lock:
                self._hold_sent_for_latch = True
        except Exception as exc:
            self.get_logger().error(
                f"RDK hold request failed (watchdog will retry): {exc}",
                throttle_duration_sec=1.0,
            )
        finally:
            with self._state_lock:
                self._hold_request_inflight = False

    def _stop_control(self) -> None:
        self._arbiter.stop()
        self._send_hold_once("local_stop")
        self._publish_control_state()

    def _zero_goal(self, request: ZeroFTSensors.Goal) -> GoalResponse:
        with self._state_lock:
            if self._zero_goal_reserved:
                return GoalResponse.REJECT
        if self._arbiter.snapshot.state is not ControlState.MAINTENANCE:
            return GoalResponse.REJECT
        if request.session_id != self._session_id:
            return GoalResponse.REJECT
        if not request.local_console or not request.local_authorization_token:
            return GoalResponse.REJECT
        if request.operator_confirmation != "FLEXIV-FT-UNLOADED":
            return GoalResponse.REJECT
        if len(request.tool_payload_config_hash) != 64:
            return GoalResponse.REJECT
        try:
            left = self._current_hand_angles("left")
            right = self._current_hand_angles("right")
        except Exception:
            return GoalResponse.REJECT
        tolerance = float(self.get_parameter("hand_reference_tolerance").value)
        if (
            np.max(np.abs(left - np.asarray(request.left_hand_position))) > tolerance
            or np.max(np.abs(right - np.asarray(request.right_hand_position))) > tolerance
        ):
            return GoalResponse.REJECT
        with self._state_lock:
            if self._zero_goal_reserved:
                return GoalResponse.REJECT
            self._zero_goal_reserved = True
        return GoalResponse.ACCEPT

    def _execute_zero_ft(self, goal_handle):
        goal = goal_handle.request
        result = ZeroFTSensors.Result()
        feedback = ZeroFTSensors.Feedback()
        feedback.phase = "daemon_transaction"
        feedback.active_arm = ""
        feedback.progress = 0.0
        feedback.status = "waiting for protected local daemon"
        goal_handle.publish_feedback(feedback)
        payload: dict = {}
        try:
            left_measured = self._current_hand_angles("left")
            right_measured = self._current_hand_angles("right")
            tolerance = float(self.get_parameter("hand_reference_tolerance").value)
            if (
                np.max(np.abs(left_measured - np.asarray(goal.left_hand_position))) > tolerance
                or np.max(np.abs(right_measured - np.asarray(goal.right_hand_position))) > tolerance
            ):
                raise RuntimeError("hand position changed after Action goal acceptance")
            kind, payload = self._ipc_maintenance.request(
                "zero_ft",
                {
                    "session_id": goal.session_id,
                    "operator_confirmation": goal.operator_confirmation,
                    "local_console": bool(goal.local_console),
                    "tool_payload_config_hash": goal.tool_payload_config_hash,
                    "local_authorization_token": goal.local_authorization_token,
                    "left_hand_position": left_measured.tolist(),
                    "right_hand_position": right_measured.tolist(),
                },
                timeout_s=180.0,
            )
            if kind != "zero_ft_result":
                raise RuntimeError(f"unexpected {kind}")
            self._populate_ft_result(result, payload)
            if not payload.get("success", False):
                raise RuntimeError(payload.get("failure_reason", f"unexpected {kind}"))
            generation = int(payload["connection_generation"])
            instance_id = str(payload.get("daemon_instance_id", ""))
            # The observation thread updates both epoch fields under this same
            # lock. Keep validation and READY transition atomic with respect to
            # a reconnect arriving after the daemon result.
            with self._state_lock:
                _require_matching_ft_zero_epoch(
                    result_generation=generation,
                    result_instance_id=instance_id,
                    current_generation=self._connection_generation,
                    current_instance_id=self._daemon_instance_id,
                )
                self._arbiter.mark_ft_zeroed(
                    session_id=self._session_id,
                    connection_generation=generation,
                )
                self._arbiter.declare_ready(connection_generation=generation)
            result.success = True
            result.final_state = "READY"
            goal_handle.succeed()
        except Exception as exc:
            result.success = False
            result.final_state = self._arbiter.snapshot.state.value
            result.failure_reason = str(exc)
            if payload:
                self._populate_ft_result(result, payload)
            goal_handle.abort()
        finally:
            with self._state_lock:
                self._zero_goal_reserved = False
        self._publish_control_state()
        return result

    def _populate_ft_result(self, result, payload: dict) -> None:
        result.event_id = str(payload.get("event_json", ""))
        result.failure_reason = str(payload.get("failure_reason", ""))
        available = payload.get("available_statistics", {})
        for key, field_name in (
            ("left_before", "left_before"),
            ("left_after", "left_after"),
            ("right_before", "right_before"),
            ("right_after", "right_after"),
        ):
            value = available.get(key)
            arm = payload.get(key.split("_")[0], {})
            value = value or arm.get(key.split("_")[1])
            if value:
                self._fill_ft_statistics(
                    getattr(result, field_name), key.split("_")[0], value
                )

    @staticmethod
    def _fill_ft_statistics(output, side: str, value: dict) -> None:
        output.side = side
        output.sample_count = int(value.get("samples", "0"))
        duration_ns = int(float(value.get("duration_s", 0.0)) * 1e9)
        output.window = _duration_from_ns(duration_ns)
        for field_name in (
            "mean",
            "standard_deviation",
            "peak_absolute",
            "external_mean",
            "external_standard_deviation",
            "external_peak_absolute",
        ):
            values = [float(item) for item in value.get(field_name, [])]
            if len(values) != 6:
                raise ValueError(f"F/T statistics {field_name} is not length six")
            setattr(output, field_name, values)
        output.max_dq_norm = float(value.get("max_dq_norm", 0.0))
        output.max_tcp_velocity_norm = float(
            value.get("max_tcp_velocity_norm", 0.0)
        )
        output.stable = bool(value.get("stable", False))
        output.rejection_reason = str(
            value.get("rejection_reason", value.get("reason", ""))
        )

    def _emit_deviceio(
        self, topic: str, message, host_ns: int, *, critical: bool
    ) -> bool:
        emitter = getattr(self, "_deviceio", None)
        if emitter is None:
            # Unit harnesses construct a bridge without running its ROS constructor.
            return True
        try:
            sequence = int(getattr(message, "sequence", 0))
            emitter.emit(
                record_envelope(
                    producer="control",
                    topic=topic,
                    source_time_ns=host_ns,
                    host_receive_time_ns=host_ns,
                    sequence=sequence,
                    valid=True,
                    source_clock_domain="host_monotonic",
                    host_clock_domain="host_monotonic",
                    payload=message_to_ordereddict(message),
                ),
                critical=critical,
            )
            return True
        except (BufferError, RuntimeError, ValueError) as exc:
            self.get_logger().error(f"critical DeviceIO capture failed for {topic}: {exc}")
            return False

    def _publish_trace(self, requested, safe, sent, rejection: str, started_ns: int) -> None:
        trace = CommandTrace()
        trace.header.stamp = self.get_clock().now().to_msg()
        trace.session_id = self._session_id
        trace.trace_sequence = int(getattr(requested, "sequence", 0))
        trace.requested = requested
        trace.requested_valid = True
        if safe is not None:
            trace.safe = safe
            trace.safe_valid = True
        if sent is not None:
            trace.sent = sent
            trace.sent_valid = True
        trace.rejection_reason = rejection
        trace.validation_latency = _duration_from_ns(max(0, time.monotonic_ns() - started_ns))
        self._emit_deviceio(
            "/control/command_trace", trace, time.monotonic_ns(), critical=True
        )
        self._trace_pub.publish(trace)

    def _publish_control_state(self) -> None:
        snapshot = self._arbiter.snapshot
        state = ControlStateMsg()
        state.header.stamp = self.get_clock().now().to_msg()
        state.state_name = snapshot.state.value
        state.state = list(ControlState).index(snapshot.state)
        state.session_id = self._session_id
        state.active_source = "" if snapshot.active_source is None else snapshot.active_source.value
        state.hold_reason = snapshot.hold_reason.value
        state.local_permission = self._local_permission
        state.physical_pedal = self._physical_pedal
        state.ft_zeroed_for_session = snapshot.ft_zero_generation is not None
        now = time.monotonic_ns()
        state.arms_online = all(
            self._last_arm_observation_ns[side] > 0
            and now - self._last_arm_observation_ns[side] <= 100_000_000
            and bool(self._latest_wire.get(side, {}).get("connected", False))
            and not str(self._latest_wire.get(side, {}).get("fault", ""))
            for side in ("left", "right")
        )
        state.hands_online = all(
            self._hand_connected[side]
            and now - self._last_hand_observation_ns[side] <= 200_000_000
            for side in ("left", "right")
        )
        state.rdk_connection_generation = self._connection_generation or 0
        self._emit_deviceio(
            "/control/state", state, time.monotonic_ns(), critical=False
        )
        self._control_state_pub.publish(state)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ControlBridge()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
