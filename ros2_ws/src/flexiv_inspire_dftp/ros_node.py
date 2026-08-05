"""ROS 2 adapter for two independent Inspire RH56DFTP-2 workers.

The driver is read-only unless all three local launch parameters are supplied:
``hardware_write_enabled=true``, a non-empty ``local_session_id``, and the
literal local confirmation ``DFTP-LOCAL-CONTROL-AUTHORIZED``. Normal motion only
subscribes to ``/control/sent_command`` (the safety supervisor's output), and
requires a fresh ``ACTIVE`` control state, matching session, deadman, validity
mask, monotonic sequence, and TTL before filling a worker's latest-only mailbox.
A separate local Reset service can run open-close-open only in ``READY``. The
driver never subscribes directly to teleop, policy, or replay.
"""

from __future__ import annotations

from collections import deque
import math
import threading
import time
from typing import Any

from .models import HandCommand, HandState, TactileFrame
from .modbus import (
    CommandCapableModbusTcpClient,
    LocalWritePermit,
    ReadOnlyModbusTcpClient,
)
from .protocol import ACTUATOR_NAMES, TACTILE_LAYOUT
from .worker import DftpCommandSink, DftpHandWorker, DftpProtocolReader
from isaac_teleop_core.deviceio import AsyncDeviceIOEmitter, record_envelope


ANGLE_RANGES_DEG = (
    (20.0, 176.0),
    (20.0, 176.0),
    (20.0, 176.0),
    (20.0, 176.0),
    (-13.0, 70.0),
    (90.0, 165.0),
)

VALID_COMMAND_SOURCES = frozenset({"teleop", "policy", "replay"})
HAND_FIELD_TIMINGS = {
    "angle": "angle",
    "position": "position",
    "actual_force": "force",
    "current": "current",
    "temperature": "temperature",
    "error": "error",
    "status": "status",
}


def _replace_fixed_tactile_surfaces(target, surfaces) -> None:
    """Replace every slot in the fixed ROS tactile-surface array.

    ``TactileFrame.surfaces`` is ``TactileSurface[17]`` and rclpy constructs
    all 17 default messages before publication. Appending puts real surfaces
    after those defaults, while serialization retains the first 17 blanks.
    """

    if len(target) != len(surfaces):
        raise ValueError(
            f"fixed tactile surface array has {len(target)} slots, "
            f"received {len(surfaces)} surfaces"
        )
    for index, surface in enumerate(surfaces):
        target[index] = surface


def hand_reset_targets(
    open_angle: int = 1000, closed_angle: int = 0
) -> tuple[tuple[int, ...], ...]:
    """Return the legacy Inspire connection-check motion: open-close-open."""

    opened = int(open_angle)
    closed = int(closed_angle)
    if not 0 <= opened <= 1000 or not 0 <= closed <= 1000:
        raise ValueError("hand reset angles must be in 0..1000")
    if opened == closed:
        raise ValueError("hand reset open and closed angles must differ")
    return ((opened,) * 6, (closed,) * 6, (opened,) * 6)


def hand_target_reached(
    measured: tuple[int, ...] | None,
    target: tuple[int, ...],
    tolerance: int,
) -> bool:
    """Return whether all six measured actuator angles reached a target."""

    if measured is None or len(measured) != 6 or len(target) != 6:
        return False
    if tolerance < 0:
        raise ValueError("hand target tolerance must be non-negative")
    return all(
        abs(int(actual) - int(expected)) <= tolerance
        for actual, expected in zip(measured, target)
    )


def command_source_matches(active_source: str, command_source: str) -> bool:
    return active_source in VALID_COMMAND_SOURCES and command_source == active_source


def angle_registers_to_radians(values: tuple[int, ...]) -> tuple[float, ...]:
    return tuple(
        math.radians(low + (max(0, min(1000, value)) / 1000.0) * (high - low))
        for value, (low, high) in zip(values, ANGLE_RANGES_DEG)
    )


def _assign_time(message: Any, time_ns: int) -> None:
    message.sec = int(time_ns // 1_000_000_000)
    message.nanosec = int(time_ns % 1_000_000_000)


def _time_ns(message: Any) -> int:
    return int(message.sec) * 1_000_000_000 + int(message.nanosec)


def _fill_acquisition(
    message: Any,
    *,
    source: int,
    receive: int,
    start: int,
    end: int,
    sequence: int,
    valid: bool,
    reason: str = "",
    source_clock_domain: str = "host_monotonic",
    host_clock_domain: str = "host_monotonic",
    mapped_host_time: int | None = None,
) -> None:
    _assign_time(message.source_time, source)
    _assign_time(message.host_receive_time, receive)
    _assign_time(message.acquisition_start, start)
    _assign_time(message.acquisition_end, end)
    message.source_sequence = sequence
    message.valid = valid
    message.source_clock_domain = source_clock_domain
    message.host_clock_domain = host_clock_domain
    if mapped_host_time is None and source_clock_domain == host_clock_domain:
        mapped_host_time = source
    if mapped_host_time is None:
        _assign_time(message.mapped_host_time, 0)
        message.timing_valid = False
        age = 0
    else:
        _assign_time(message.mapped_host_time, mapped_host_time)
        age = receive - mapped_host_time
        message.timing_valid = age >= 0
        if age < 0:
            age = 0
            timing_reason = "mapped-source-is-after-host-receive"
            reason = (
                f"{reason};{timing_reason}" if reason else timing_reason
            )
    message.age.sec = int(age // 1_000_000_000)
    message.age.nanosec = int(age % 1_000_000_000)
    message.invalid_reason = reason


def main(args=None) -> int:
    import rclpy
    from rclpy.node import Node
    from control_msgs.msg import DynamicJointState, InterfaceValue
    from sensor_msgs.msg import JointState
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from rosidl_runtime_py.convert import message_to_ordereddict
    from std_srvs.srv import Trigger
    from flexiv_inspire_interfaces.msg import (
        BimanualCommand,
        ControlState,
        HandState as HandStateMsg,
        TactileFrame as TactileFrameMsg,
        TactileSurface as TactileSurfaceMsg,
    )

    class DualDftpNode(Node):
        def __init__(self) -> None:
            super().__init__("flexiv_inspire_dftp_driver")
            self._sensor_qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.BEST_EFFORT,
            )
            self._control_qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
            )
            self.declare_parameter("left_host", "192.168.5.11")
            self.declare_parameter("right_host", "192.168.5.12")
            self.declare_parameter("port", 6000)
            self.declare_parameter("state_hz", 50.0)
            self.declare_parameter("tactile_hz", 30.0)
            self.declare_parameter("socket_timeout_s", 0.25)
            self.declare_parameter("hardware_write_enabled", False)
            self.declare_parameter("local_session_id", "")
            self.declare_parameter("local_write_confirmation", "")
            self.declare_parameter(
                "safe_force_limits", [600, 600, 600, 600, 500, 500]
            )
            self.declare_parameter("control_state_timeout_ms", 200.0)
            self.declare_parameter("hand_state_timeout_ms", 200.0)
            self.declare_parameter("hand_reset_enabled", False)
            self.declare_parameter("hand_reset_open_angle", 1000)
            self.declare_parameter("hand_reset_closed_angle", 0)
            self.declare_parameter("hand_reset_pause_s", 0.35)
            self.declare_parameter("hand_reset_command_timeout_s", 3.0)
            self.declare_parameter("hand_reset_open_tolerance", 30)
            self.declare_parameter("hand_reset_open_timeout_s", 8.0)

            self._write_enabled = bool(
                self.get_parameter("hardware_write_enabled").value
            )
            self._configured_session = str(
                self.get_parameter("local_session_id").value
            )
            confirmation = str(
                self.get_parameter("local_write_confirmation").value
            )
            force_limits = tuple(
                int(value) for value in self.get_parameter("safe_force_limits").value
            )
            if len(force_limits) != 6:
                raise ValueError("safe_force_limits must contain six values")
            permit = None
            if self._write_enabled:
                permit = LocalWritePermit.issue_after_local_authorization(
                    self._configured_session, confirmation
                )

            self._control_active = False
            self._control_session = ""
            self._control_source = ""
            self._control_state_received_ns = 0
            self._control_state_timeout_ns = int(
                float(self.get_parameter("control_state_timeout_ms").value) * 1e6
            )
            self._hand_state_timeout_ns = int(
                float(self.get_parameter("hand_state_timeout_ms").value) * 1e6
            )
            self._hand_reset_enabled = bool(
                self.get_parameter("hand_reset_enabled").value
            )
            self._hand_reset_targets = hand_reset_targets(
                int(self.get_parameter("hand_reset_open_angle").value),
                int(self.get_parameter("hand_reset_closed_angle").value),
            )
            self._hand_reset_pause_s = float(
                self.get_parameter("hand_reset_pause_s").value
            )
            self._hand_reset_command_timeout_s = float(
                self.get_parameter("hand_reset_command_timeout_s").value
            )
            self._hand_reset_open_tolerance = int(
                self.get_parameter("hand_reset_open_tolerance").value
            )
            self._hand_reset_open_timeout_s = float(
                self.get_parameter("hand_reset_open_timeout_s").value
            )
            if not 0.05 <= self._hand_reset_pause_s <= 2.0:
                raise ValueError("hand_reset_pause_s must be in [0.05,2.0]")
            if not 1.0 <= self._hand_reset_command_timeout_s <= 10.0:
                raise ValueError(
                    "hand_reset_command_timeout_s must be in [1.0,10.0]"
                )
            if not 0 <= self._hand_reset_open_tolerance <= 200:
                raise ValueError("hand_reset_open_tolerance must be in [0,200]")
            if not 1.0 <= self._hand_reset_open_timeout_s <= 30.0:
                raise ValueError("hand_reset_open_timeout_s must be in [1.0,30.0]")
            self._last_hand_state_ns = {"left": 0, "right": 0}
            self._last_hand_sequence = {"left": 0, "right": 0}
            self._latest_hand_angles: dict[str, tuple[int, ...] | None] = {
                "left": None,
                "right": None,
            }
            # Source sequences belong to the control bridge. Worker sequences
            # also include local Reset/hold commands, so the counters must not
            # be coupled or Reset can suppress the first teleop frames.
            self._last_source_command_sequence = -1
            self._worker_command_sequence = -1
            self._sequence_lock = threading.Lock()
            self._hand_reset_lock = threading.Lock()
            self._fault_latched = False
            self._fault_reason_by_side = {"left": "", "right": ""}
            self._control_state_name = ""
            self._offline_queue = {
                "left": deque(maxlen=1),
                "right": deque(maxlen=1),
            }
            self._state_queue = {
                "left": deque(maxlen=1),
                "right": deque(maxlen=1),
            }
            self._tactile_queue = {
                "left": deque(maxlen=1),
                "right": deque(maxlen=1),
            }
            # ``rclpy.node.Node`` owns ``self._publishers`` and expects it to
            # remain a list.  Keep the application's topic lookup separate so
            # create_publisher() can register publishers with the base class.
            self._topic_publishers = {}
            self._workers = {}
            self._deviceio = AsyncDeviceIOEmitter("dftp")

            for side in ("left", "right"):
                self._topic_publishers[(side, "joint")] = self.create_publisher(
                    JointState, f"/robot/{side}_hand/joint_states", self._sensor_qos
                )
                self._topic_publishers[(side, "dynamic")] = self.create_publisher(
                    DynamicJointState,
                    f"/robot/{side}_hand/dynamic_joint_states",
                    self._sensor_qos,
                )
                self._topic_publishers[(side, "state")] = self.create_publisher(
                    HandStateMsg, f"/robot/{side}_hand/state", self._sensor_qos
                )
                self._topic_publishers[(side, "tactile")] = self.create_publisher(
                    TactileFrameMsg, f"/robot/{side}_hand/tactile_raw", self._sensor_qos
                )

                transport_kwargs = dict(
                    port=int(self.get_parameter("port").value),
                    timeout_s=float(self.get_parameter("socket_timeout_s").value),
                )
                host = str(self.get_parameter(f"{side}_host").value)
                if self._write_enabled:
                    transport = CommandCapableModbusTcpClient(
                        host, permit=permit, **transport_kwargs
                    )
                    command_sink = DftpCommandSink(transport)
                else:
                    transport = ReadOnlyModbusTcpClient(host, **transport_kwargs)
                    command_sink = None
                reader = DftpProtocolReader(
                    transport, side, clock_ns=time.monotonic_ns
                )
                # Tactile acquisition is 17 Modbus transactions per frame.
                # Keep it on a second read-only connection so it cannot block
                # the 200 Hz non-tactile state and command path.
                tactile_transport = ReadOnlyModbusTcpClient(
                    host, **transport_kwargs
                )
                tactile_reader = DftpProtocolReader(
                    tactile_transport, side, clock_ns=time.monotonic_ns
                )
                self._workers[side] = DftpHandWorker(
                    side,
                    reader,
                    tactile_reader=tactile_reader,
                    command_sink=command_sink,
                    state_hz=float(self.get_parameter("state_hz").value),
                    tactile_hz=float(self.get_parameter("tactile_hz").value),
                    safe_force_limits=force_limits,
                    on_state=lambda value, side=side: self._on_state(side, value),
                    on_tactile=lambda value, side=side: self._tactile_queue[
                        side
                    ].append(value),
                    on_fault=lambda reason, side=side: self._on_fault(
                        side, reason),
                )

            self.create_subscription(
                ControlState, "/control/state", self._on_control_state, self._control_qos
            )
            self.create_subscription(
                BimanualCommand,
                "/control/sent_command",
                self._on_safe_command,
                self._control_qos,
            )
            self.create_service(
                Trigger, "/maintenance/cycle_hands", self._on_cycle_hands
            )
            self.create_timer(0.002, self._drain)
            for worker in self._workers.values():
                worker.start()
            mode = "command-capable" if self._write_enabled else "read-only"
            self.get_logger().info(
                f"DFTP project Modbus driver started {mode}; it is not vendor "
                "ROS 2 source and has no legacy-workspace dependency"
            )

        def destroy_node(self):
            for worker in self._workers.values():
                worker.stop()
            self._deviceio.close()
            return super().destroy_node()

        def _native_emit(self, topic: str, message: Any) -> None:
            acquisition = message.acquisition
            source_ns = _time_ns(acquisition.source_time)
            receive_ns = _time_ns(acquisition.host_receive_time)
            mapped_ns = _time_ns(acquisition.mapped_host_time)
            timing_valid = bool(acquisition.timing_valid) and mapped_ns > 0
            try:
                self._deviceio.emit(record_envelope(
                    producer="dftp",
                    topic=topic,
                    source_time_ns=source_ns,
                    host_receive_time_ns=receive_ns,
                    sequence=int(acquisition.source_sequence),
                    valid=bool(acquisition.valid),
                    invalid_reason=str(acquisition.invalid_reason),
                    source_clock_domain=str(acquisition.source_clock_domain),
                    host_clock_domain=str(acquisition.host_clock_domain),
                    mapped_host_time_ns=mapped_ns if timing_valid else None,
                    timing_valid=timing_valid,
                    payload=message_to_ordereddict(message),
                ))
            except (BufferError, RuntimeError, ValueError):
                # Recorder availability must not perturb the Modbus workers.
                pass

        def _on_state(self, side: str, state: HandState) -> None:
            self._last_hand_state_ns[side] = time.monotonic_ns()
            self._last_hand_sequence[side] = state.acquisition.sequence
            self._latest_hand_angles[side] = state.actuator_angle
            self._state_queue[side].append(state)

        def _on_fault(self, side: str, reason: str) -> None:
            self._fault_latched = True
            self._control_active = False
            self._fault_reason_by_side[side] = reason
            self._offline_queue[side].append((time.monotonic_ns(), reason))
            self.get_logger().error(reason)
            self._request_both_holds(f"driver-fault:{reason}")

        def _on_control_state(self, message) -> None:
            now = time.monotonic_ns()
            raw_state = getattr(message, "state", None)
            state_name = str(getattr(message, "state_name", "")).upper()
            active_constant = getattr(type(message), "ACTIVE", None)
            active = state_name == "ACTIVE" or (
                active_constant is not None and raw_state == active_constant
            )
            session_id = str(getattr(message, "session_id", ""))
            control_source = str(getattr(message, "active_source", ""))
            if control_source not in VALID_COMMAND_SOURCES:
                active = False
            if self._write_enabled and session_id != self._configured_session:
                active = False
            was_active = self._control_active
            self._control_active = active and not self._fault_latched
            self._control_state_name = state_name
            self._control_session = session_id
            self._control_source = control_source if active else ""
            self._control_state_received_ns = now
            if was_active and not self._control_active:
                self._request_both_holds("control-left-active")

        def _on_safe_command(self, message) -> None:
            if not self._write_enabled:
                return
            now_mono = time.monotonic_ns()
            now_ros = self.get_clock().now().nanoseconds
            if self._fault_latched or not self._control_active:
                return
            if (
                now_mono - self._control_state_received_ns
                > self._control_state_timeout_ns
            ):
                self._control_active = False
                self._request_both_holds("control-state-timeout")
                return
            if (
                message.schema_version != 1
                or message.session_id != self._configured_session
                or message.session_id != self._control_session
                or not command_source_matches(self._control_source, message.source)
                or not message.deadman
                or message.sequence <= self._last_source_command_sequence
            ):
                return
            ttl_ns = _time_ns(message.ttl)
            stamp_ns = _time_ns(message.header.stamp)
            age_ns = now_ros - stamp_ns
            if (
                ttl_ns <= 0
                or ttl_ns > 1_000_000_000
                or age_ns < -50_000_000
                or age_ns >= ttl_ns
                or len(message.trajectory) != 1
            ):
                return
            point = message.trajectory[0]
            if _time_ns(point.execute_after) >= ttl_ns:
                return
            remaining_ns = ttl_ns - max(0, age_ns)
            valid_mask = int(message.valid_mask)
            submitted = False
            worker_sequence: int | None = None
            for side, bit_name, fallback_bit in (
                ("left", "LEFT_HAND_VALID", 4),
                ("right", "RIGHT_HAND_VALID", 8),
            ):
                bit = int(getattr(type(message), bit_name, fallback_bit))
                if not valid_mask & bit:
                    continue
                if (
                    now_mono - self._last_hand_state_ns[side]
                    > self._hand_state_timeout_ns
                ):
                    self._fault_latched = True
                    self._request_both_holds(f"{side}-hand-state-stale")
                    return
                targets = tuple(
                    int(round(value))
                    for value in getattr(point, f"{side}_hand_targets")
                )
                try:
                    if worker_sequence is None:
                        worker_sequence = self._next_worker_sequence()
                    hand_command = HandCommand(
                        sequence=worker_sequence,
                        angles=targets,
                        force_limits=self._workers[side].safe_force_limits,
                        deadline_ns=now_mono + remaining_ns,
                        source=f"safe:{message.source}",
                    )
                except ValueError as exc:
                    self._fault_latched = True
                    self._request_both_holds(f"invalid-hand-command:{exc}")
                    return
                submitted = self._workers[side].submit(hand_command) or submitted
            if submitted:
                self._last_source_command_sequence = int(message.sequence)

        def _next_worker_sequence(self) -> int:
            with self._sequence_lock:
                self._worker_command_sequence += 1
                return self._worker_command_sequence

        def _on_cycle_hands(self, _request, response):
            """Run the installed-hardware Reset gesture after arm Home."""

            if not self._hand_reset_enabled:
                response.success = False
                response.message = "hand reset is disabled by configuration"
                return response
            if not self._write_enabled:
                response.success = False
                response.message = "DFTP driver is read-only"
                return response
            if self._fault_latched:
                response.success = False
                response.message = "DFTP fault is latched"
                return response
            if self._control_state_name != "READY":
                response.success = False
                response.message = (
                    "hand reset requires control state READY; got "
                    + (self._control_state_name or "no state")
                )
                return response
            if self._control_session != self._configured_session:
                response.success = False
                response.message = "control session does not match DFTP session"
                return response
            now = time.monotonic_ns()
            stale = [
                side
                for side in ("left", "right")
                if now - self._last_hand_state_ns[side]
                > self._hand_state_timeout_ns
            ]
            if stale:
                response.success = False
                response.message = "hand state is stale: " + ",".join(stale)
                return response
            if not self._hand_reset_lock.acquire(blocking=False):
                response.success = False
                response.message = "hand reset is already running"
                return response
            try:
                # The first two targets retain the proven legacy gesture.  The
                # final open target is handled separately and verified against
                # ANGLE_ACTUAL; queue acceptance alone is not motion success.
                labels = ("open", "closed")
                for label, target in zip(labels, self._hand_reset_targets[:2]):
                    sequence = self._next_worker_sequence()
                    deadline_ns = time.monotonic_ns() + int(
                        self._hand_reset_command_timeout_s * 1e9
                    )
                    accepted = {}
                    for side, worker in self._workers.items():
                        accepted[side] = worker.submit(
                            HandCommand(
                                sequence=sequence,
                                angles=target,
                                force_limits=worker.safe_force_limits,
                                deadline_ns=deadline_ns,
                                source=f"local-reset:{label}",
                            )
                        )
                    if not all(accepted.values()):
                        self._request_both_holds("hand-reset-submit-failed")
                        response.success = False
                        response.message = (
                            "failed to submit reset target: " + label
                        )
                        return response
                    time.sleep(self._hand_reset_pause_s)

                open_target = self._hand_reset_targets[-1]
                verification_deadline = (
                    time.monotonic() + self._hand_reset_open_timeout_s
                )
                retry_period_s = min(0.5, self._hand_reset_pause_s)
                # Always enqueue the final open before examining measurements.
                # Otherwise an observation left over from the first open phase
                # can make the service return while the close command is still
                # the newest command in a worker.
                baseline_sequences = dict(self._last_hand_sequence)
                sequence = self._next_worker_sequence()
                command_deadline_ns = time.monotonic_ns() + int(
                    self._hand_reset_command_timeout_s * 1e9
                )
                accepted = {
                    side: worker.submit(
                        HandCommand(
                            sequence=sequence,
                            angles=open_target,
                            force_limits=worker.safe_force_limits,
                            deadline_ns=command_deadline_ns,
                            source="local-reset:open-final",
                        )
                    )
                    for side, worker in self._workers.items()
                }
                if not all(accepted.values()):
                    response.success = False
                    response.message = "failed to submit final open target"
                    return response
                next_submit = time.monotonic() + retry_period_s
                while True:
                    now_s = time.monotonic()
                    reached = {
                        side: (
                            self._last_hand_sequence[side]
                            > baseline_sequences[side]
                            and hand_target_reached(
                                self._latest_hand_angles[side],
                                open_target,
                                self._hand_reset_open_tolerance,
                            )
                        )
                        for side in ("left", "right")
                    }
                    if all(reached.values()):
                        response.success = True
                        response.message = (
                            "both Inspire hands completed open-close-open and "
                            "measured open: "
                            + ", ".join(
                                f"{side}={list(self._latest_hand_angles[side] or ())}"
                                for side in ("left", "right")
                            )
                        )
                        return response
                    if self._fault_latched:
                        response.success = False
                        response.message = (
                            "DFTP fault while reopening hands: "
                            + "; ".join(
                                filter(None, self._fault_reason_by_side.values())
                            )
                        )
                        return response
                    if now_s >= verification_deadline:
                        response.success = False
                        response.message = (
                            "hands did not reach open target before timeout: "
                            + ", ".join(
                                f"{side}={list(self._latest_hand_angles[side] or ())}"
                                for side in ("left", "right")
                            )
                        )
                        return response
                    if now_s >= next_submit:
                        sequence = self._next_worker_sequence()
                        command_deadline_ns = time.monotonic_ns() + int(
                            self._hand_reset_command_timeout_s * 1e9
                        )
                        accepted = {
                            side: worker.submit(
                                HandCommand(
                                    sequence=sequence,
                                    angles=open_target,
                                    force_limits=worker.safe_force_limits,
                                    deadline_ns=command_deadline_ns,
                                    source="local-reset:open-final",
                                )
                            )
                            for side, worker in self._workers.items()
                        }
                        if not all(accepted.values()):
                            response.success = False
                            response.message = "failed to submit final open target"
                            return response
                        next_submit = now_s + retry_period_s
                    time.sleep(0.05)
            except (RuntimeError, ValueError) as exc:
                self._request_both_holds("hand-reset-error")
                response.success = False
                response.message = str(exc)
                return response
            finally:
                self._hand_reset_lock.release()

        def _request_both_holds(self, reason: str) -> None:
            sequence = self._next_worker_sequence()
            for worker in self._workers.values():
                worker.request_hold(sequence, reason)

        def _drain(self) -> None:
            now = time.monotonic_ns()
            if (
                self._control_active
                and now - self._control_state_received_ns
                > self._control_state_timeout_ns
            ):
                self._control_active = False
                self._request_both_holds("control-state-watchdog")
            for side in ("left", "right"):
                if self._offline_queue[side]:
                    timestamp, reason = self._offline_queue[side].pop()
                    self._offline_queue[side].clear()
                    self._publish_offline_state(side, timestamp, reason)

                if self._state_queue[side]:
                    self._publish_state(self._state_queue[side].pop())
                    self._state_queue[side].clear()
                if self._tactile_queue[side]:
                    self._publish_tactile(self._tactile_queue[side].pop())
                    self._tactile_queue[side].clear()

        def _header(self, message, side: str) -> None:
            message.header.stamp = self.get_clock().now().to_msg()
            message.header.frame_id = f"{side}_hand"

        def _publish_offline_state(
            self, side: str, timestamp_ns: int, reason: str
        ) -> None:
            message = HandStateMsg()
            self._header(message, side)
            message.side = side
            message.sequence = self._last_hand_sequence[side] + 1
            self._last_hand_sequence[side] = message.sequence
            _fill_acquisition(
                message.acquisition,
                source=timestamp_ns,
                receive=timestamp_ns,
                start=timestamp_ns,
                end=timestamp_ns,
                sequence=message.sequence,
                valid=False,
                reason=reason,
            )
            for ros_field in HAND_FIELD_TIMINGS:
                _fill_acquisition(
                    getattr(message, f"{ros_field}_timing"),
                    source=timestamp_ns,
                    receive=timestamp_ns,
                    start=timestamp_ns,
                    end=timestamp_ns,
                    sequence=message.sequence,
                    valid=False,
                    reason=reason,
                )
            unavailable = float("nan")
            message.angle = [unavailable] * 6
            message.position = [unavailable] * 6
            message.actual_force = [unavailable] * 6
            message.current = [unavailable] * 6
            message.temperature = [unavailable] * 6
            message.error = [65535] * 6
            message.status = [65535] * 6
            message.connected = False
            message.fault = True
            message.fault_reason = reason
            self._native_emit(f"/robot/{side}_hand/state", message)
            self._topic_publishers[(side, "state")].publish(message)

        def _publish_state(self, state: HandState) -> None:
            names = [f"{state.side}_{name}" for name in ACTUATOR_NAMES]
            joint = JointState()
            self._header(joint, state.side)
            joint.name = names
            joint.position = list(angle_registers_to_radians(state.actuator_angle))
            self._topic_publishers[(state.side, "joint")].publish(joint)

            dynamic = DynamicJointState()
            self._header(dynamic, state.side)
            dynamic.joint_names = names
            for index in range(6):
                interfaces = InterfaceValue()
                interfaces.interface_names = [
                    "actuator_position_raw",
                    "actuator_angle_raw",
                    "actual_force_g",
                    "current_mA",
                    "temperature_C",
                    "error_code",
                    "status_code",
                ]
                interfaces.values = [
                    float(state.actuator_position[index]),
                    float(state.actuator_angle[index]),
                    float(state.actual_force_g[index]),
                    float(state.current_ma[index]),
                    float(state.temperature_c[index]),
                    float(state.error_code[index]),
                    float(state.status_code[index]),
                ]
                dynamic.interface_values.append(interfaces)
            self._topic_publishers[(state.side, "dynamic")].publish(dynamic)

            custom = HandStateMsg()
            self._header(custom, state.side)
            custom.side = state.side
            custom.sequence = state.acquisition.sequence
            windows = tuple(state.field_times_ns.values())
            start = min(
                (value[0] for value in windows),
                default=state.acquisition.source_time_ns,
            )
            end = max(
                (value[1] for value in windows),
                default=state.acquisition.host_receive_time_ns,
            )
            _fill_acquisition(
                custom.acquisition,
                source=state.acquisition.source_time_ns,
                receive=state.acquisition.host_receive_time_ns,
                start=start,
                end=end,
                sequence=state.acquisition.sequence,
                valid=state.acquisition.valid,
                reason=state.acquisition.invalid_reason,
            )
            for ros_field, model_field in HAND_FIELD_TIMINGS.items():
                field_start, field_end = state.field_times_ns.get(
                    model_field, (start, end)
                )
                _fill_acquisition(
                    getattr(custom, f"{ros_field}_timing"),
                    source=(field_start + field_end) // 2,
                    receive=field_end,
                    start=field_start,
                    end=field_end,
                    sequence=state.acquisition.sequence,
                    valid=state.acquisition.valid,
                    reason=state.acquisition.invalid_reason,
                )
            custom.angle = [float(value) for value in state.actuator_angle]
            custom.position = [float(value) for value in state.actuator_position]
            custom.actual_force = [float(value) for value in state.actual_force_g]
            custom.current = [float(value) for value in state.current_ma]
            custom.temperature = [float(value) for value in state.temperature_c]
            custom.error = list(state.error_code)
            custom.status = list(state.status_code)
            custom.connected = True
            hardware_reason = ""
            if any(state.error_code):
                hardware_reason = "hardware-error:" + ",".join(
                    str(value) for value in state.error_code
                )
            driver_reason = self._fault_reason_by_side[state.side]
            custom.fault = bool(hardware_reason or driver_reason)
            custom.fault_reason = ";".join(filter(None, (hardware_reason, driver_reason)))
            self._native_emit(f"/robot/{state.side}_hand/state", custom)
            self._topic_publishers[(state.side, "state")].publish(custom)

        def _publish_tactile(self, frame: TactileFrame) -> None:
            message = TactileFrameMsg()
            self._header(message, frame.side)
            message.side = frame.side
            message.sequence = frame.sequence
            message.taxel_count = frame.taxel_count
            message.valid = frame.valid
            message.invalid_reason = frame.invalid_reason
            _fill_acquisition(
                message.acquisition,
                source=(frame.acquisition_start_ns + frame.acquisition_end_ns) // 2,
                receive=frame.acquisition_end_ns,
                start=frame.acquisition_start_ns,
                end=frame.acquisition_end_ns,
                sequence=frame.sequence,
                valid=frame.valid,
                reason=frame.invalid_reason,
            )
            specs = {spec.name: spec for spec in TACTILE_LAYOUT}
            surface_messages = []
            for surface in frame.surfaces:
                item = TactileSurfaceMsg()
                item.name = surface.name
                item.rows = surface.rows
                item.columns = surface.cols
                item.modbus_start_address = specs[surface.name].start_address
                item.taxels = list(surface.values)
                _fill_acquisition(
                    item.acquisition,
                    source=(
                        surface.acquisition_start_ns + surface.acquisition_end_ns
                    )
                    // 2,
                    receive=surface.acquisition_end_ns,
                    start=surface.acquisition_start_ns,
                    end=surface.acquisition_end_ns,
                    sequence=frame.sequence,
                    valid=surface.valid,
                    reason=surface.invalid_reason,
                )
                surface_messages.append(item)
            _replace_fixed_tactile_surfaces(
                message.surfaces, surface_messages
            )
            self._native_emit(f"/robot/{frame.side}_hand/tactile_raw", message)
            self._topic_publishers[(frame.side, "tactile")].publish(message)

    rclpy.init(args=args)
    node = DualDftpNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
