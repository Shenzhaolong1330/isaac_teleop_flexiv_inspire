"""Locally authorized, timing-faithful replay of accepted episode commands.

This module deliberately reads DeviceIO MCAP instead of replaying a ROS bag.
Every outgoing command is rebuilt with the current session, clock and TTL and
is sent through the normal replay-source control bridge gates.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
import uuid
from typing import Any

from flexiv_inspire_isaac.data_pipeline.manifest import canonical_yaml_sha256
from flexiv_inspire_isaac.data_pipeline.playback import (
    PlaybackConfigError,
    RecordedCommand,
    extract_replay_commands,
    load_playback_config,
    read_deviceio_records,
    resolve_episode,
    validate_replay_home_origin,
    validate_replay_timing,
)
from .system_config import SystemConfigError, load_system_config


REPLAY_CONFIRMATION = "FLEXIV-REPLAY-EXECUTE"
_TERMINAL_CONTROL_STATES = {"HOLD_LATCHED", "FAULT", "MAINTENANCE", "DISABLED"}


def _default_config() -> Path:
    return Path(__file__).resolve().parents[4] / "config" / "playback.yaml"


def _validate_replay_request(
    config_path: str | Path,
) -> tuple[Any, Any, Any, list[RecordedCommand]]:
    """Complete all side-effect-free validation before ROS publishers exist."""

    spec = load_playback_config(config_path)
    if not spec.replay.enabled:
        raise PlaybackConfigError(
            "hardware replay is disabled; set replay.enabled=true deliberately"
        )
    if spec.replay.operator_confirmation != REPLAY_CONFIRMATION:
        raise PlaybackConfigError(
            f"replay.operator_confirmation must equal {REPLAY_CONFIRMATION}"
        )
    if not spec.replay.home_before_start:
        raise PlaybackConfigError("hardware replay requires replay.home_before_start=true")
    if not sys.stdin.isatty():
        raise PlaybackConfigError(
            "hardware replay authorization requires an interactive local TTY"
        )

    episode = resolve_episode(spec, for_hardware=True)
    records = read_deviceio_records(episode.deviceio_mcap)
    commands = extract_replay_commands(records, spec.replay)
    validate_replay_timing(commands, spec.replay)
    site = load_system_config(spec.site_config)
    session_id = str(site.document["session"]["id"]).strip()
    if not session_id:
        raise PlaybackConfigError("current site session.id is empty")

    tool_path = site.resolve(site.document["flexiv"]["tool_payload_config"])
    current_tool_hash = canonical_yaml_sha256(tool_path)
    recorded_tool_hash = str(episode.manifest.get("tool_configuration_hash", ""))
    if not recorded_tool_hash or current_tool_hash != recorded_tool_hash:
        raise PlaybackConfigError(
            "recorded and current tool/payload configuration hashes differ"
        )
    home = site.document["flexiv"]["home"]
    validate_replay_home_origin(
        records,
        commands,
        left_home=home["left_joints_rad"],
        right_home=home["right_joints_rad"],
        tolerance_rad=float(home["tolerance_rad"]),
    )
    return spec, site, episode, commands


def _fill_command_message(
    message: Any,
    point: Any,
    command: RecordedCommand,
    *,
    session_id: str,
    sequence: int,
    ttl_s: float,
    episode_uuid: str,
) -> None:
    """Populate ROS-like message objects; kept pure for boundary tests."""

    from isaac_teleop_core.command import ROTATION_ORDER

    action = command.action
    if len(action) != 30:
        raise ValueError("replay action must contain 30 values")
    message.header.frame_id = "world"
    message.schema_version = 1
    message.session_id = session_id
    message.source = "replay"
    message.sequence = sequence
    ttl_ns = int(round(ttl_s * 1e9))
    message.ttl.sec = ttl_ns // 1_000_000_000
    message.ttl.nanosec = ttl_ns % 1_000_000_000
    message.representation = 1
    message.frame_id = "world"
    message.rotation_order = ROTATION_ORDER
    message.valid_mask = command.valid_mask
    message.deadman = True

    point.execute_after.sec = 0
    point.execute_after.nanosec = 0
    point.left_delta_xyz = list(action[0:3])
    point.left_delta_rotation6d = list(action[3:9])
    point.right_delta_xyz = list(action[9:12])
    point.right_delta_rotation6d = list(action[12:18])
    point.left_delta_quaternion_xyzw = [0.0, 0.0, 0.0, 1.0]
    point.right_delta_quaternion_xyzw = [0.0, 0.0, 0.0, 1.0]
    point.left_hand_targets = list(action[18:24])
    point.right_hand_targets = list(action[24:30])
    message.trajectory = [point]
    message.metadata_keys = ["replay_episode_uuid", "recorded_sequence"]
    message.metadata_values = [episode_uuid, str(command.original_sequence)]


class ReplayNode:
    """Small ROS facade around the existing control bridge safety contract."""

    def __init__(self, *, session_id: str, rdk_socket: Path) -> None:
        import rclpy
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import Empty, String, UInt64
        from flexiv_inspire_interfaces.msg import BimanualCommand, ControlState

        self._rclpy = rclpy
        self._String = String
        self._UInt64 = UInt64
        self._Empty = Empty
        self._BimanualCommand = BimanualCommand
        self._session_id = session_id
        self._rdk_socket = rdk_socket
        self._control_state: Any | None = None
        self._home_events: list[dict[str, Any]] = []
        self._stopped = False
        self._owns_replay_control = False
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.node = rclpy.create_node("flexiv_inspire_replay_controller")
        self.command = self.node.create_publisher(
            BimanualCommand, "/command_sources/replay/command", qos
        )
        self.heartbeat = self.node.create_publisher(
            UInt64, "/command_sources/replay/heartbeat", qos
        )
        self.stop = self.node.create_publisher(Empty, "/control/stop", qos)
        self.home_authorization = self.node.create_publisher(
            String, "/control/local_home_authorization", qos
        )
        self.home_request = self.node.create_publisher(
            String, "/control/home_request_context", qos
        )
        self.arm_authorization = self.node.create_publisher(
            String, "/control/local_arm_authorization", qos
        )
        self.node.create_subscription(
            ControlState, "/control/state", self._on_control_state, qos
        )
        self.node.create_subscription(
            String, "/control/home_status", self._on_home_status, qos
        )

    def _on_control_state(self, message: Any) -> None:
        self._control_state = message

    def _on_home_status(self, message: Any) -> None:
        try:
            payload = json.loads(message.data)
        except (TypeError, json.JSONDecodeError):
            return
        if isinstance(payload, dict):
            self._home_events.append(payload)
            self._home_events = self._home_events[-32:]

    def spin(self, timeout_s: float = 0.02) -> None:
        self._rclpy.spin_once(self.node, timeout_sec=timeout_s)

    def wait_for_bridge(self, timeout_s: float = 5.0) -> None:
        publishers = (
            self.command,
            self.heartbeat,
            self.stop,
            self.home_authorization,
            self.home_request,
            self.arm_authorization,
        )
        deadline = time.monotonic() + timeout_s
        stable_since: float | None = None
        while time.monotonic() < deadline:
            self.spin(0.05)
            subscriber_counts = [
                publisher.get_subscription_count() for publisher in publishers
            ]
            if any(count > 1 for count in subscriber_counts):
                raise RuntimeError(
                    "unexpected command/control subscribers; stop recording before replay"
                )
            if self.node.count_subscribers("/episode/control") > 0:
                raise RuntimeError(
                    "an episode controller is active; stop collection before replay"
                )
            if self.node.count_publishers("/command_sources/replay/command") > 1:
                raise RuntimeError("another replay publisher is already active")
            control_publishers = self.node.count_publishers("/control/state")
            home_publishers = self.node.count_publishers("/control/home_status")
            if control_publishers > 1 or home_publishers > 1:
                raise RuntimeError("more than one control bridge is active")
            ready = (
                self._control_state is not None
                and all(count == 1 for count in subscriber_counts)
                and control_publishers == 1
                and home_publishers == 1
            )
            if not ready:
                stable_since = None
                continue
            if stable_since is None:
                stable_since = time.monotonic()
            elif time.monotonic() - stable_since >= 0.5:
                return
        raise RuntimeError("control bridge topics were not discovered")

    def _request_authorization(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        from flexiv_inspire_control.ipc_client import RDKIPCClient

        client = RDKIPCClient(self._rdk_socket)
        try:
            response_kind, response = client.request(kind, payload)
        finally:
            client.close()
        if response_kind != f"{kind}_result" or not response.get("authorized", False):
            raise RuntimeError(str(response.get("reason", response_kind)))
        return response

    @staticmethod
    def _json_message(message_type: Any, payload: dict[str, Any]) -> Any:
        message = message_type()
        message.data = json.dumps(payload, separators=(",", ":"))
        return message

    def _publish_three(self, publisher: Any, message: Any) -> None:
        for _ in range(3):
            publisher.publish(message)
            self.spin(0.05)

    def home(self, *, timeout_s: float) -> None:
        response = self._request_authorization(
            "authorize_home",
            {
                "session_id": self._session_id,
                "operator_confirmation": "FLEXIV-HOME-MOVE",
                "clear_hold_latched": False,
            },
        )
        authorization = self._json_message(
            self._String,
            {
                "session_id": self._session_id,
                "one_time_token": response["one_time_token"],
                "expires_monotonic_ns": response["expires_monotonic_ns"],
            },
        )
        before = len(self._home_events)
        self._publish_three(self.home_authorization, authorization)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            self.spin()
            recent = self._home_events[before:]
            if any(str(item.get("state", "")) == "authorized" for item in recent):
                break
            rejection = next(
                (
                    item
                    for item in recent
                    if str(item.get("state", "")) == "authorization_rejected"
                ),
                None,
            )
            if rejection is not None:
                raise RuntimeError(str(rejection.get("reason", "Home rejected")))
        else:
            raise TimeoutError("control bridge did not acknowledge Home authorization")

        request_id = f"replay-{uuid.uuid4().hex}"
        request = self._json_message(
            self._String,
            {
                "request_id": request_id,
                "source": "replay_controller",
                "session_id": self._session_id,
            },
        )
        self.home_request.publish(request)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self.spin(0.05)
            matching = [
                item
                for item in self._home_events
                if str(item.get("request_id", "")) == request_id
            ]
            if any(str(item.get("state", "")) == "complete" for item in matching):
                return
            failed = next(
                (
                    item
                    for item in matching
                    if str(item.get("state", "")) in {"failed", "rejected"}
                ),
                None,
            )
            if failed is not None:
                raise RuntimeError(str(failed.get("reason", "Home failed")))
        raise TimeoutError("Home did not complete before the replay timeout")

    def authorize_replay(self, timeout_s: float = 3.0) -> None:
        response = self._request_authorization(
            "authorize_control",
            {
                "session_id": self._session_id,
                "source": "replay",
                "operator_confirmation": "FLEXIV-CONTROL-ARM",
                "clear_hold_latched": False,
            },
        )
        authorization = self._json_message(
            self._String,
            {
                "session_id": self._session_id,
                "source": "replay",
                "one_time_token": response["one_time_token"],
                "expires_monotonic_ns": response["expires_monotonic_ns"],
            },
        )
        self._publish_three(self.arm_authorization, authorization)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self.spin()
            state = self._state_name()
            if state == "REPLAY_ARMED":
                self._owns_replay_control = True
                return
            if state in _TERMINAL_CONTROL_STATES:
                raise RuntimeError(f"control bridge entered {state}: {self._hold_reason()}")
        raise TimeoutError("control bridge did not enter REPLAY_ARMED")

    def _state_name(self) -> str:
        return "" if self._control_state is None else str(self._control_state.state_name)

    def _hold_reason(self) -> str:
        return "" if self._control_state is None else str(self._control_state.hold_reason)

    def require_ready_gates(self) -> None:
        state = self._control_state
        if state is None:
            raise RuntimeError("control state is unavailable")
        failures = []
        if str(state.session_id) != self._session_id:
            failures.append("session mismatch")
        if not bool(state.local_permission):
            failures.append("local permission is false")
        if not bool(state.physical_pedal):
            failures.append("physical pedal is not held")
        if not bool(state.ft_zeroed_for_session):
            failures.append("F/T is not zeroed for this session")
        if not bool(state.arms_online):
            failures.append("arms are offline")
        if not bool(state.hands_online):
            failures.append("hands are offline")
        if failures:
            raise RuntimeError("replay gates not ready: " + ", ".join(failures))

    def publish_heartbeat(self) -> None:
        message = self._UInt64()
        message.data = time.monotonic_ns()
        self.heartbeat.publish(message)

    def wait_until(self, deadline_ns: int, *, active_required: bool = False) -> None:
        while time.monotonic_ns() < deadline_ns:
            self.spin(0.01)
            self.publish_heartbeat()
            if active_required and self._state_name() in _TERMINAL_CONTROL_STATES:
                raise RuntimeError(
                    f"replay stopped in {self._state_name()}: {self._hold_reason()}"
                )

    def publish_command(
        self,
        command: RecordedCommand,
        *,
        sequence: int,
        ttl_s: float,
        episode_uuid: str,
    ) -> None:
        from flexiv_inspire_interfaces.msg import BimanualCommandPoint

        message = self._BimanualCommand()
        message.header.stamp = self.node.get_clock().now().to_msg()
        point = BimanualCommandPoint()
        _fill_command_message(
            message,
            point,
            command,
            session_id=self._session_id,
            sequence=sequence,
            ttl_s=ttl_s,
            episode_uuid=episode_uuid,
        )
        self.publish_heartbeat()
        self.command.publish(message)

    def wait_active(self, timeout_s: float = 0.5) -> None:
        deadline = time.monotonic_ns() + int(timeout_s * 1e9)
        while time.monotonic_ns() < deadline:
            self.spin(0.01)
            self.publish_heartbeat()
            state = self._state_name()
            if state == "ACTIVE":
                return
            if state in _TERMINAL_CONTROL_STATES:
                raise RuntimeError(f"first replay command failed: {state}:{self._hold_reason()}")
        raise TimeoutError("control bridge did not accept the first replay command")

    def request_stop(self) -> None:
        if self._stopped or not self._owns_replay_control:
            return
        self._stopped = True
        self.stop.publish(self._Empty())
        self.spin(0.10)

    def close(self) -> None:
        self.node.destroy_node()


def run_replay(config_path: str | Path) -> int:
    spec, site, episode, commands = _validate_replay_request(config_path)
    session_id = str(site.document["session"]["id"])
    runtime_root = Path(site.document["session"]["runtime_root"]).expanduser()
    rdk_socket = runtime_root / "rdk.sock"
    if not rdk_socket.is_socket():
        raise RuntimeError(f"RDK endpoint is not an active Unix socket: {rdk_socket}")

    import rclpy

    rclpy.init()
    replay: ReplayNode | None = None
    try:
        replay = ReplayNode(session_id=session_id, rdk_socket=rdk_socket)
        replay.wait_for_bridge()
        home_timeout = float(site.document["flexiv"]["home"]["timeout_s"]) + 10.0
        print(f"replay selected {episode.directory}; commands={len(commands)}", flush=True)
        print("moving both arms to configured Home before replay", flush=True)
        replay.home(timeout_s=home_timeout)
        replay.authorize_replay()
        replay.require_ready_gates()
        print(
            f"replay armed; keep the physical pedal held; starts in "
            f"{spec.replay.start_delay_s:g}s",
            flush=True,
        )
        replay.wait_until(
            time.monotonic_ns() + int(spec.replay.start_delay_s * 1e9)
        )

        first_timestamp = commands[0].timestamp_ns
        base_sequence = min(time.monotonic_ns(), (((1 << 64) - 1) >> 6) - len(commands))
        start_ns = time.monotonic_ns()
        episode_uuid = str(episode.manifest.get("episode_uuid", ""))
        replay.publish_command(
            commands[0],
            sequence=base_sequence,
            ttl_s=spec.replay.ttl_s,
            episode_uuid=episode_uuid,
        )
        replay.wait_active()
        for index, command in enumerate(commands[1:], start=1):
            offset_ns = int(
                (command.timestamp_ns - first_timestamp) / spec.replay.speed
            )
            deadline_ns = start_ns + offset_ns
            replay.wait_until(deadline_ns, active_required=True)
            lateness_ns = time.monotonic_ns() - deadline_ns
            if lateness_ns > int(spec.replay.max_schedule_lateness_s * 1e9):
                raise RuntimeError(
                    f"replay scheduler missed sample {index} by "
                    f"{lateness_ns / 1e6:.3f} ms"
                )
            replay.publish_command(
                command,
                sequence=base_sequence + index,
                ttl_s=spec.replay.ttl_s,
                episode_uuid=episode_uuid,
            )
        replay.wait_until(
            time.monotonic_ns() + min(int(spec.replay.ttl_s * 5e8), 100_000_000),
            active_required=True,
        )
        print(
            f"replay complete at {spec.replay.speed:g}x; requesting guarded hold",
            flush=True,
        )
        return 0
    finally:
        try:
            if replay is not None:
                replay.request_stop()
        finally:
            if replay is not None:
                replay.close()
            if rclpy.ok():
                rclpy.shutdown()


def main(argv: list[str] | None = None) -> int:
    selected = sys.argv[1:] if argv is None else argv
    if selected:
        raise SystemExit("flexiv-inspire-replay takes no arguments; edit config/playback.yaml")
    try:
        return run_replay(_default_config())
    except KeyboardInterrupt:
        print("replay interrupted", file=sys.stderr)
        return 130
    except (OSError, ValueError, SystemConfigError, RuntimeError) as exc:
        raise SystemExit(f"replay refused: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
