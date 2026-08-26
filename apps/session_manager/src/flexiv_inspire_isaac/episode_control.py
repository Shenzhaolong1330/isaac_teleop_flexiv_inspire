"""Config-driven multi-episode collection controller.

The controller owns the recorder subprocess lifecycle.  Left/right pedal and
Quest-A events are serialized through one guarded Home transaction so robot
reset motion is never part of a demonstration.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import String
from flexiv_inspire_interfaces.msg import ControlState

from flexiv_inspire_control.ipc_client import RDKIPCClient
from flexiv_inspire_isaac.data_pipeline.manifest import local_minute_timestamp


_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
_RAW_EPISODE_DIRECTORY = "raw"
_ROUTINE_CONTROL_HOLD_REASONS = {
    "physical_pedal_released",
    "source_deadman_released",
    "control_authorization_expired",
    "command_stale",
    "source_heartbeat_stale",
    "arm_offline",
    "hand_offline",
    "safety_limit",
}

_CAMERA_LABELS = {
    "head": "头部相机",
    "left_wrist": "左腕相机",
    "right_wrist": "右腕相机",
}

_STREAM_LABELS = {
    "camera/head/color/image_raw/compressed": "头部 RGB",
    "camera/left_wrist/color/image_raw/compressed": "左腕 RGB",
    "camera/right_wrist/color/image_raw/compressed": "右腕 RGB",
    "robot/left_arm/state": "左臂状态",
    "robot/right_arm/state": "右臂状态",
    "robot/left_hand/state": "左手状态",
    "robot/right_hand/state": "右手状态",
    "robot/left_hand/tactile_raw": "左手触觉",
    "robot/right_hand/tactile_raw": "右手触觉",
    "control/sent_command": "已下发动作",
}


def _terminal_banner(logger, title: str, *details: str, level: str = "info") -> None:
    """Print one unmistakable collection transition in the foreground terminal."""

    border = "═" * 72
    lines = [f"\n╔{border}╗", f"  {title}"]
    lines.extend(f"  {detail}" for detail in details if detail)
    lines.append(f"╚{border}╝")
    getattr(logger, level)("\n".join(lines))


def _human_completion_reason(reason: str) -> str:
    marker = "required streams have no valid samples: "
    if marker not in reason:
        return reason
    missing = reason.split(marker, 1)[1].split(";", 1)[0]
    labels = [
        _STREAM_LABELS.get(item.strip(), item.strip())
        for item in missing.split(",")
        if item.strip()
    ]
    return "缺少有效数据流：" + "、".join(labels)


def _next_task_episode_index(storage_root: Path, task_name: str) -> int:
    """Continue one task's numbering across separate collection processes."""

    if not storage_root.is_dir():
        return 1
    pattern = re.compile(
        rf"{re.escape(task_name)}_episode_(\d{{3,}})_\d{{8}}_\d{{4}}"
        rf"(?:_\d{{2}})?"
    )
    maximum = 0
    for child in storage_root.iterdir():
        if not child.is_dir():
            continue
        match = pattern.fullmatch(child.name)
        if match is not None:
            maximum = max(maximum, int(match.group(1)))
    return maximum + 1


class EpisodeController(Node):
    def __init__(self) -> None:
        super().__init__("flexiv_inspire_episode_controller")
        defaults = {
            "sessions_root": "sessions",
            "dataset_name": "dataset",
            "task_name": "task",
            "episode_count": 1,
            "task_description": "",
            "session_id": "",
            "tool_config": "",
            "ft_zero_record": "",
            "camera_config": "",
            "camera_head_extrinsics": "",
            "camera_left_wrist_extrinsics": "",
            "camera_right_wrist_extrinsics": "",
            "manus_calibration": "",
            "deviceio_socket": "",
            "runtime_dir": "",
            "rdk_socket": "",
            "camera_recording_mode": "jpeg",
            "deviceio_mode": "native",
            "ros_mcap_enabled": True,
            "deviceio_profile": "full",
            "record_only_while_pedal_pressed": False,
            "camera_hz": 15.0,
            "action_hz": 30.0,
            "home_result_timeout_s": 30.0,
            "recorder_state_timeout_s": 10.0,
            "control_authorization_refresh_s": 10.0,
            "auto_start": True,
            "auto_authorize_home": True,
            "auto_authorize_control": True,
            "control_source": "teleop",
        }
        for key, value in defaults.items():
            self.declare_parameter(key, value)
        self._validate_configuration()
        self._lock = threading.RLock()
        self._process: subprocess.Popen[str] | None = None
        self._sequence = 0
        sessions_root = Path(
            str(self.get_parameter("sessions_root").value)
        ).expanduser().resolve()
        dataset_name = str(self.get_parameter("dataset_name").value).strip()
        task_name = str(self.get_parameter("task_name").value).strip()
        self._episode_index = _next_task_episode_index(
            sessions_root / dataset_name / _RAW_EPISODE_DIRECTORY,
            task_name,
        )
        self._attempt = 1
        self._completed_episodes = 0
        self._current_manifest: Path | None = None
        self._pending_home_request_id: str | None = None
        self._pending_home_deadline_ns = 0
        self._pending_home_action: str | None = None
        self._awaiting_home_authorization = False
        self._home_recovery_attempted = False
        self._routine_control_rearm_pending = False
        self._routine_control_rearm_pressed_attempted = False
        self._next_control_authorization_refresh_ns = 0
        self._control_state_name = ""
        self._control_physical_pedal = False
        self._last_announced_control_state: tuple[str, str, bool] | None = None
        self._control_ready_for_recording = False
        self._next_ready_wait_log_ns = 0
        self._startup_done = False
        self._finished = False
        self._camera_status_by_name: dict[str, tuple[bool, bool, str]] = {}

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self._status = self.create_publisher(String, "/episode/status", qos)
        self._home_request = self.create_publisher(
            String, "/control/home_request_context", qos
        )
        self._home_authorization = self.create_publisher(
            String, "/control/local_home_authorization", qos
        )
        self._arm_authorization = self.create_publisher(
            String, "/control/local_arm_authorization", qos
        )
        self.create_subscription(String, "/episode/control", self._on_control, qos)
        self.create_subscription(
            String, "/control/home_status", self._on_home_status, qos
        )
        self.create_subscription(
            ControlState, "/control/state", self._on_control_state, qos
        )
        for camera in _CAMERA_LABELS:
            self.create_subscription(
                String,
                f"/camera/{camera}/color/status",
                lambda message, name=camera: self._on_camera_status(name, message),
                qos,
            )
        self.create_timer(0.25, self._tick)
        self.get_logger().info(
            "数采控制：左踏板=丢弃当前条并 Home 后重录；"
            "中踏板=按住使能双臂、松开立即停止；"
            "右踏板=保存当前条并 Home 后无缝开始下一条；"
            "Quest A=暂停记录、Home 后继续当前条；Ctrl+C=结束数采"
        )
        self._publish("STARTING")

    @property
    def finished(self) -> bool:
        return self._finished

    def _required(self, name: str) -> str:
        value = str(self.get_parameter(name).value).strip()
        if not value:
            raise RuntimeError(f"episode controller parameter {name} is required")
        return value

    def _validate_configuration(self) -> None:
        dataset_name = str(self.get_parameter("dataset_name").value).strip()
        if not _NAME.fullmatch(dataset_name):
            raise RuntimeError(
                "dataset_name must use 1..96 letters, digits, '.', '_' or '-'"
            )
        task_name = str(self.get_parameter("task_name").value).strip()
        if not _NAME.fullmatch(task_name):
            raise RuntimeError(
                "task_name must use 1..96 letters, digits, '.', '_' or '-'"
            )
        count = int(self.get_parameter("episode_count").value)
        if not 1 <= count <= 100_000:
            raise RuntimeError("episode_count must be in [1,100000]")
        if not str(self.get_parameter("task_description").value).strip():
            raise RuntimeError("task_description is required")
        if str(self.get_parameter("deviceio_mode").value) != "native":
            raise RuntimeError("automatic collection requires native DeviceIO")
        for key in ("home_result_timeout_s", "recorder_state_timeout_s"):
            if not 1.0 <= float(self.get_parameter(key).value) <= 120.0:
                raise RuntimeError(f"{key} must be in [1,120]")
        refresh = float(
            self.get_parameter("control_authorization_refresh_s").value
        )
        if not 2.0 <= refresh <= 25.0:
            raise RuntimeError(
                "control_authorization_refresh_s must be in [2,25]"
            )
        for key in ("camera_hz", "action_hz"):
            value = float(self.get_parameter(key).value)
            if not math.isfinite(value) or value <= 0.0:
                raise RuntimeError(f"{key} must be positive and finite")

    def _state_file(self) -> Path:
        return Path(self._required("runtime_dir")) / "episode-recorder-state.json"

    def _command(self) -> list[str]:
        self._sequence += 1
        collection_timestamp = local_minute_timestamp()
        task_name = str(self.get_parameter("task_name").value).strip()
        base_episode_name = (
            f"{task_name}_episode_{self._episode_index:03d}_"
            f"{collection_timestamp}"
        )
        root = Path(self._required("sessions_root")).expanduser().resolve()
        dataset = str(self.get_parameter("dataset_name").value).strip()
        # The collection timestamp intentionally has minute precision so it is
        # easy to read in a dataset browser.  Retrying `robot record` within
        # that minute must still get a fresh directory rather than killing the
        # recorder before it can acknowledge startup.
        episode_name = base_episode_name
        suffix = 1
        storage_root = root / dataset / _RAW_EPISODE_DIRECTORY
        while (storage_root / episode_name).exists():
            episode_name = f"{base_episode_name}_{suffix:02d}"
            suffix += 1
        self._current_manifest = storage_root / episode_name / "manifest.json"
        command = [
            sys.executable,
            "-m",
            "flexiv_inspire_isaac.data_pipeline.episode_manager",
            "--root",
            str(root),
            "--dataset-name",
            dataset,
            "--storage-subdirectory",
            _RAW_EPISODE_DIRECTORY,
            "--episode-index",
            str(self._episode_index),
            "--attempt",
            str(self._attempt),
            "--task-name",
            task_name,
            "--episode-directory-name",
            episode_name,
            "--collection-timestamp-local",
            collection_timestamp,
            "--task-description",
            str(self.get_parameter("task_description").value),
            "--session-id",
            self._required("session_id"),
            "--tool-config",
            self._required("tool_config"),
            "--ft-zero-record",
            self._required("ft_zero_record"),
            "--calibration",
            f"cameras={self._required('camera_config')}",
            "--camera-recording-mode",
            str(self.get_parameter("camera_recording_mode").value),
            "--deviceio-mode",
            str(self.get_parameter("deviceio_mode").value),
            "--deviceio-profile",
            str(self.get_parameter("deviceio_profile").value),
            "--expected-camera-hz",
            str(float(self.get_parameter("camera_hz").value)),
            "--expected-action-hz",
            str(float(self.get_parameter("action_hz").value)),
            "--deviceio-socket",
            self._required("deviceio_socket"),
            "--control-state-file",
            str(self._state_file()),
        ]
        if not bool(self.get_parameter("ros_mcap_enabled").value):
            command.append("--no-ros-mcap")
        if bool(
            self.get_parameter("record_only_while_pedal_pressed").value
        ):
            command.append("--record-only-while-pedal-pressed")
        manus = str(self.get_parameter("manus_calibration").value).strip()
        if manus:
            command.extend(("--calibration", f"manus={manus}"))
        for camera in ("head", "left_wrist", "right_wrist"):
            path = str(
                self.get_parameter(f"camera_{camera}_extrinsics").value
            ).strip()
            if path:
                command.extend(("--calibration", f"camera_{camera}={path}"))
        return command

    def _publish(self, value: str) -> None:
        message = String()
        message.data = value
        self._status.publish(message)

    def _publish_progress(self, state: str) -> None:
        progress = {
            "state": state,
            "dataset_name": str(self.get_parameter("dataset_name").value),
            "task_name": str(self.get_parameter("task_name").value),
            "episode_index": self._episode_index,
            "attempt": self._attempt,
            "completed_episodes": self._completed_episodes,
            "target_episodes": int(self.get_parameter("episode_count").value),
        }
        self._publish(json.dumps(progress, separators=(",", ":")))
        self.get_logger().info(
            f"{state}: episode={self._episode_index} "
            f"attempt={self._attempt} completed={self._completed_episodes}/"
            f"{progress['target_episodes']}"
        )

    def _episode_label(self) -> str:
        return (
            f"episode={self._episode_index:03d} attempt={self._attempt} "
            f"completed={self._completed_episodes}/"
            f"{int(self.get_parameter('episode_count').value)}"
        )

    def _on_camera_status(self, camera: str, message: String) -> None:
        """Surface camera disconnects hidden in supporting-service log files."""

        try:
            payload = json.loads(message.data)
            if not isinstance(payload, dict):
                raise ValueError("camera status must be a JSON object")
            status = (
                bool(payload.get("connected", False)),
                bool(payload.get("fault", False)),
                str(payload.get("reason", "unknown")),
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            self.get_logger().warning(
                f"ignored malformed {camera} camera status: {exc}"
            )
            return
        previous = self._camera_status_by_name.get(camera)
        self._camera_status_by_name[camera] = status
        if previous == status:
            return
        connected, fault, reason = status
        label = _CAMERA_LABELS.get(camera, camera)
        serial = str(payload.get("serial", "")).strip()
        identity = f"{label}" + (f"（{serial}）" if serial else "")
        if fault:
            _terminal_banner(
                self.get_logger(),
                f"⚠ 数据流中断：{identity}",
                f"原因：{reason}",
                "该相机恢复前请勿继续采集；没有有效帧的 episode 会自动作废重录。",
                level="error",
            )
        elif connected and (previous is None or not previous[0] or previous[1]):
            _terminal_banner(
                self.get_logger(),
                f"✓ 数据流恢复：{identity}",
                "后续踩下中踏板时会正常写入该相机画面。",
            )

    def _read_recorder_state(self) -> dict[str, object]:
        try:
            value = json.loads(self._state_file().read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}

    def _wait_recorder_state(self, expected: str) -> None:
        timeout = float(self.get_parameter("recorder_state_timeout_s").value)
        deadline = time.monotonic() + timeout
        process = self._process
        while time.monotonic() < deadline:
            if process is None or process.poll() is not None:
                raise RuntimeError("episode recorder exited before acknowledgement")
            if str(self._read_recorder_state().get("state", "")) == expected:
                return
            time.sleep(0.02)
        raise RuntimeError(f"episode recorder did not acknowledge {expected}")

    def _start(self) -> None:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                return
            state_file = self._state_file()
            try:
                state_file.unlink()
            except FileNotFoundError:
                pass
            self._publish_progress("STARTING_EPISODE")
            self._process = subprocess.Popen(self._command(), text=True)
        try:
            self._wait_recorder_state("RECORDING")
        except Exception:
            self._stop(rerecord=True)
            raise
        self._publish_progress("RECORDING")
        destination = "" if self._current_manifest is None else str(
            self._current_manifest.parent
        )
        mode = (
            "等待中踏板；仅中踏板按住期间写入训练数据"
            if bool(self.get_parameter("record_only_while_pedal_pressed").value)
            else "正在持续写入数据"
        )
        _terminal_banner(
            self.get_logger(),
            f"● EPISODE {self._episode_index:03d} 已开始",
            self._episode_label(),
            mode,
            f"目录：{destination}",
        )

    def _delete_discarded_episode(self) -> None:
        """Delete only the exact episode directory owned by this controller."""

        manifest_path = self._current_manifest
        if manifest_path is None:
            return
        root = Path(self._required("sessions_root")).expanduser().resolve()
        dataset = str(self.get_parameter("dataset_name").value).strip()
        dataset_root = (root / dataset / _RAW_EPISODE_DIRECTORY).resolve()
        episode_directory = manifest_path.parent.resolve()
        try:
            episode_directory.relative_to(dataset_root)
        except ValueError as exc:
            raise RuntimeError(
                f"refusing to delete episode outside dataset: {episode_directory}"
            ) from exc
        if (
            episode_directory.parent != dataset_root
            or re.fullmatch(
                rf"{re.escape(str(self.get_parameter('task_name').value).strip())}"
                rf"_episode_{self._episode_index:03d}_\d{{8}}_\d{{4}}"
                rf"(?:_\d{{2}})?",
                episode_directory.name,
            )
            is None
        ):
            raise RuntimeError(
                f"refusing unexpected episode deletion target: {episode_directory}"
            )
        if episode_directory.exists():
            shutil.rmtree(episode_directory)
            self.get_logger().info(f"已删除丢弃的数据: {episode_directory}")
        self._current_manifest = None

    def _stop(self, *, rerecord: bool) -> bool:
        with self._lock:
            process = self._process
            self._process = None
        if process is None:
            return False
        if process.poll() is None:
            self._publish_progress("DISCARDING" if rerecord else "FINALIZING")
            process.send_signal(signal.SIGUSR1 if rerecord else signal.SIGINT)
            try:
                process.wait(timeout=30.0)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5.0)
        manifest = {}
        episode_directory = (
            None if self._current_manifest is None else self._current_manifest.parent
        )
        if self._current_manifest is not None and self._current_manifest.is_file():
            manifest = json.loads(self._current_manifest.read_text(encoding="utf-8"))
        completed = bool(manifest.get("completed", False))
        if rerecord and completed:
            raise RuntimeError("discarded episode was incorrectly marked completed")
        if rerecord:
            _terminal_banner(
                self.get_logger(),
                f"↺ EPISODE {self._episode_index:03d} 已丢弃",
                "不会计入完成数量；Home 后自动重录当前编号。",
                level="warning",
            )
            self._delete_discarded_episode()
        if not rerecord and not completed:
            reason = str(
                manifest.get("completion_reason", "manifest missing")
            )
            _terminal_banner(
                self.get_logger(),
                f"✗ EPISODE {self._episode_index:03d} 无效，自动重录",
                _human_completion_reason(reason),
                "本条不计数，目录将删除。请先恢复缺失的数据流。",
                level="error",
            )
            self._delete_discarded_episode()
            self.get_logger().warning(
                f"episode 记录失败，已自动删除并将重录当前编号: {reason}"
            )
            return False
        if completed:
            _terminal_banner(
                self.get_logger(),
                f"✓ EPISODE {self._episode_index:03d} 保存完成",
                "必需数据流完整性校验通过。",
                f"目录：{episode_directory or ''}",
            )
        return completed

    def _pause(self) -> None:
        process = self._process
        if process is None or process.poll() is not None:
            raise RuntimeError("no active episode to pause")
        process.send_signal(signal.SIGUSR2)
        self._wait_recorder_state("PAUSED")
        self._publish_progress("PAUSED_FOR_HOME")

    def _resume(self) -> None:
        process = self._process
        if process is None or process.poll() is not None:
            raise RuntimeError("no active episode to resume")
        process.send_signal(signal.SIGHUP)
        self._wait_recorder_state("RECORDING")
        self._publish_progress("RECORDING")

    def _request_authorization(self, kind: str, payload: dict) -> dict:
        client = RDKIPCClient(Path(self._required("rdk_socket")))
        try:
            # Clearing a routine hold prepares both stopped arms back into NRT
            # Cartesian mode before returning the one-time token.  Flexiv mode
            # transitions can legitimately take several seconds.
            response_kind, response = client.request(
                kind, payload, timeout_s=15.0
            )
        finally:
            client.close()
        expected = f"{kind}_result"
        if response_kind != expected or not response.get("authorized", False):
            raise RuntimeError(response.get("reason", response_kind))
        return response

    def _authorize_control(self, *, clear_hold_latched: bool = False) -> None:
        if not bool(self.get_parameter("auto_authorize_control").value):
            return
        source = str(self.get_parameter("control_source").value)
        response = self._request_authorization(
            "authorize_control",
            {
                "session_id": self._required("session_id"),
                "source": source,
                "operator_confirmation": "FLEXIV-CONTROL-ARM",
                "clear_hold_latched": clear_hold_latched,
            },
        )
        outgoing = String()
        outgoing.data = json.dumps(
            {
                "session_id": self._required("session_id"),
                "source": source,
                "one_time_token": response["one_time_token"],
                "expires_monotonic_ns": response["expires_monotonic_ns"],
            },
            separators=(",", ":"),
        )
        for _ in range(3):
            self._arm_authorization.publish(outgoing)
        self._next_control_authorization_refresh_ns = (
            time.monotonic_ns()
            + int(
                float(
                    self.get_parameter("control_authorization_refresh_s").value
                )
                * 1e9
            )
        )

    def _on_control_state(self, message: ControlState) -> None:
        """Re-arm after an expected clutch or authorization-lifetime hold."""

        self._control_state_name = str(message.state_name)
        self._control_physical_pedal = bool(message.physical_pedal)
        # Pedal edges are useful status changes while running, but once the
        # bridge reports FAULT they do not change the fault.  Do not print the
        # same alarming line again on every press/release.
        announced_pedal = (
            False
            if self._control_state_name == "FAULT"
            else bool(message.physical_pedal)
        )
        announced = (
            self._control_state_name,
            str(message.hold_reason),
            announced_pedal,
        )
        if announced != self._last_announced_control_state:
            previous = self._last_announced_control_state
            self._last_announced_control_state = announced
            if self._control_state_name == "ACTIVE":
                _terminal_banner(
                    self.get_logger(),
                    f"▶ 正在记录动作 · EPISODE {self._episode_index:03d}",
                    "中踏板已按下：机械臂运动已使能，观测与动作正在写入。",
                )
            elif previous is not None and previous[0] == "ACTIVE":
                _terminal_banner(
                    self.get_logger(),
                    f"Ⅱ 动作记录已暂停 · EPISODE {self._episode_index:03d}",
                    "中踏板已松开：机械臂停止；本条尚未结束。再次踩下可继续。",
                )
            if self._control_state_name == "HOLD_LATCHED":
                detail = "机械臂进入 HOLD_LATCHED：" + (
                    message.hold_reason or "unknown"
                )
                if message.hold_reason in _ROUTINE_CONTROL_HOLD_REASONS:
                    self.get_logger().info(detail)
                else:
                    self.get_logger().error(detail)
            elif self._control_state_name == "FAULT":
                self.get_logger().error(
                    "机械臂控制进入 FAULT：" + (message.hold_reason or "unknown")
                )
        self._control_ready_for_recording = (
            self._control_state_name == "READY"
            and bool(getattr(message, "local_permission", False))
            and bool(getattr(message, "ft_zeroed_for_session", False))
            and bool(getattr(message, "arms_online", False))
            and bool(getattr(message, "hands_online", False))
        )

        if message.state_name != "HOLD_LATCHED":
            self._routine_control_rearm_pending = False
            self._routine_control_rearm_pressed_attempted = False
            return
        if (
            self._pending_home_action is not None
            or message.hold_reason not in _ROUTINE_CONTROL_HOLD_REASONS
        ):
            self._routine_control_rearm_pending = False
            return

        # Do not clear the daemon's routine HOLD as soon as the pedal is
        # released.  The release notification and the daemon's safe-stop can
        # cross in flight; clearing here used to let the bridge become ACTIVE
        # before the daemon had finished latching the old release.  The first
        # command after the next press was then rejected with the stale
        # ``hold_latched:physical_pedal_released`` reason and escalated to
        # FAULT.  Remember the routine hold while released, then clear it and
        # issue a fresh one-time authorization on the next physical press.
        if not bool(message.physical_pedal):
            self._routine_control_rearm_pending = True
            self._routine_control_rearm_pressed_attempted = False
            return
        if self._routine_control_rearm_pressed_attempted:
            return

        # A daemon-side release latch can arrive just after the bridge has
        # briefly reported TELEOP_ARMED. Treat a routine HOLD observed while
        # pressed as the same re-arm request even if that intermediate state
        # reset ``_routine_control_rearm_pending``.
        self._routine_control_rearm_pressed_attempted = True
        self._routine_control_rearm_pending = False
        try:
            self._authorize_control(clear_hold_latched=True)
        except Exception as exc:
            reason = str(exc)
            controller_fault = any(
                marker in reason.lower()
                for marker in ("minor fault", "not operational")
            )
            if controller_fault:
                # Repeating SwitchMode on every 200 Hz state message cannot
                # clear a Flexiv controller fault and only floods the terminal.
                # Keep this pedal press attempted; release/press can try once
                # more after the operator has run Reset.
                self._routine_control_rearm_pending = False
                self._routine_control_rearm_pressed_attempted = True
                self.get_logger().error(
                    "机械臂控制器故障，遥操作保持禁用；松开中踏板后执行 "
                    f"robot reset。详情：{reason}"
                )
            else:
                # Transient IPC/authorization races remain retryable while the
                # operator holds the pedal.
                self._routine_control_rearm_pending = True
                self._routine_control_rearm_pressed_attempted = False
                self.get_logger().error(f"teleop re-arm failed: {exc}")

    def _authorize_home(self, *, clear_hold_latched: bool = False) -> None:
        if not bool(self.get_parameter("auto_authorize_home").value):
            self._send_home_request()
            return
        response = self._request_authorization(
            "authorize_home",
            {
                "session_id": self._required("session_id"),
                "operator_confirmation": "FLEXIV-HOME-MOVE",
                "clear_hold_latched": clear_hold_latched,
            },
        )
        outgoing = String()
        outgoing.data = json.dumps(
            {
                "session_id": self._required("session_id"),
                "one_time_token": response["one_time_token"],
                "expires_monotonic_ns": response["expires_monotonic_ns"],
                "clear_hold_latched": clear_hold_latched,
            },
            separators=(",", ":"),
        )
        self._awaiting_home_authorization = True
        for _ in range(3):
            self._home_authorization.publish(outgoing)

    def _begin_home(self, action: str) -> None:
        if self._pending_home_action is not None:
            self.get_logger().warning("ignored episode input while Home is active")
            return
        if action not in {"pause", "discard", "next"}:
            raise ValueError(f"unknown Home action {action}")
        # Every transition first closes both recording gates.  Home can then
        # take control immediately while the current bag remains intact but
        # paused; finalization happens only after Home has completed.
        self._pause()
        self._pending_home_action = action
        self._home_recovery_attempted = False
        self._pending_home_deadline_ns = time.monotonic_ns() + int(
            float(self.get_parameter("home_result_timeout_s").value) * 1e9
        )
        self._publish_progress("AUTHORIZING_HOME")
        self._authorize_home()

    def _send_home_request(self) -> None:
        request_id = f"episode-{uuid.uuid4().hex}"
        self._pending_home_request_id = request_id
        self._awaiting_home_authorization = False
        outgoing = String()
        outgoing.data = json.dumps(
            {
                "request_id": request_id,
                "source": "episode_controller",
                "session_id": self._required("session_id"),
            },
            separators=(",", ":"),
        )
        self._publish_progress("HOMING")
        self._home_request.publish(outgoing)

    def _on_control(self, message: String) -> None:
        command = str(message.data).strip().lower()
        if self._control_state_name in {
            "FAULT",
            "MAINTENANCE",
            "DISABLED",
        }:
            self.get_logger().error(
                f"ignored episode input {command!r}: control is "
                f"{self._control_state_name}; recorder remains active"
            )
            self._publish_progress("WAITING_FOR_CONTROL_RECOVERY")
            return
        try:
            if command in {"stop", "toggle"}:
                _terminal_banner(
                    self.get_logger(),
                    f"■ 右踏板：结束并保存 EPISODE {self._episode_index:03d}",
                    "正在暂停写入并执行 Home；校验通过后自动开始下一条。",
                )
                self._begin_home("next")
            elif command == "rerecord":
                _terminal_banner(
                    self.get_logger(),
                    f"↺ 左踏板：放弃 EPISODE {self._episode_index:03d}",
                    "正在暂停写入并执行 Home；随后重新录制当前编号。",
                    level="warning",
                )
                self._begin_home("discard")
            elif command == "home":
                _terminal_banner(
                    self.get_logger(),
                    f"⌂ Quest A：暂停 EPISODE {self._episode_index:03d} 并 Home",
                    "Home 期间不记录；完成后继续当前条。",
                )
                self._begin_home("pause")
            else:
                self.get_logger().warning(
                    f"ignored unknown episode control: {command!r}"
                )
        except Exception as exc:
            self._fail(f"episode_control:{exc}")

    def _on_home_status(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
            request_id = str(payload.get("request_id", ""))
            state = str(payload.get("state", "")).strip().lower()
            reason = str(payload.get("reason", "")).strip()
        except Exception as exc:
            self.get_logger().warning(f"ignored malformed Home status: {exc}")
            return
        try:
            if state == "authorized" and self._awaiting_home_authorization:
                self._send_home_request()
                return
            if (
                state == "authorization_rejected"
                and self._awaiting_home_authorization
            ):
                raise RuntimeError(reason or state)
            if not request_id or request_id != self._pending_home_request_id:
                return
            if state in {"failed", "rejected"}:
                if (
                    reason == "non_routine_hold_preserved:invalid_command"
                    and not self._home_recovery_attempted
                ):
                    self._home_recovery_attempted = True
                    self._pending_home_request_id = None
                    self._pending_home_deadline_ns = time.monotonic_ns() + int(
                        float(
                            self.get_parameter("home_result_timeout_s").value
                        )
                        * 1e9
                    )
                    self._publish_progress("RECOVERING_HOME")
                    self._authorize_home(clear_hold_latched=True)
                    return
                raise RuntimeError(reason or state)
            if state != "complete":
                return
            self._pending_home_request_id = None
            self._home_recovery_attempted = False
            action = self._pending_home_action
            self._pending_home_action = None
            self._pending_home_deadline_ns = 0
            if action == "pause":
                self._resume()
                self._authorize_control()
            elif action == "discard":
                self._stop(rerecord=True)
                self._attempt += 1
                self._start()
                self._authorize_control()
            elif action == "next":
                completed = self._stop(rerecord=False)
                if not completed:
                    self.get_logger().warning(
                        "右踏板：当前 episode 记录不完整，"
                        "本条不计数；已自动删除并重新录制当前条"
                    )
                    self._attempt += 1
                    self._start()
                    self._authorize_control()
                    return
                self._completed_episodes += 1
                target = int(self.get_parameter("episode_count").value)
                if self._completed_episodes >= target:
                    self._publish_progress("COMPLETE")
                    self._finished = True
                else:
                    self._episode_index += 1
                    self._attempt = 1
                    self._start()
                    self._authorize_control()
        except Exception as exc:
            self._recover_from_home_failure(str(exc))

    def _recover_from_home_failure(self, reason: str) -> None:
        """Keep the foreground collection alive when Home cannot run.

        A Home failure is an operator-visible transition failure, not a
        recorder-process failure.  Left-pedal semantics still discard the
        attempt; right-pedal and Quest-A semantics resume the current attempt.
        """

        action = self._pending_home_action
        self._pending_home_action = None
        self._pending_home_request_id = None
        self._pending_home_deadline_ns = 0
        self._awaiting_home_authorization = False
        self._home_recovery_attempted = False
        self.get_logger().error(
            f"Home failed without stopping collection: {reason}"
        )
        try:
            if action == "discard":
                self._stop(rerecord=True)
                self._attempt += 1
                self._start()
            else:
                self._resume()
            self._publish_progress("RECORDING_HOME_FAILED")
        except Exception as exc:
            self._fail(f"home_recovery:{reason}; recorder:{exc}")

    def _tick(self) -> None:
        if not self._startup_done and bool(self.get_parameter("auto_start").value):
            if not self._control_ready_for_recording:
                now_ns = time.monotonic_ns()
                if now_ns >= self._next_ready_wait_log_ns:
                    self.get_logger().info(
                        "waiting for Reset/READY before recording; "
                        f"control_state={self._control_state_name or 'not-received'}"
                    )
                    self._publish_progress("WAITING_FOR_READY")
                    self._next_ready_wait_log_ns = now_ns + 5_000_000_000
                return
            self._startup_done = True
            try:
                self._start()
                # Native DeviceIO must be listening before control can receive
                # a motion-authorized command.  This also keeps Reset/Home and
                # inter-episode gaps outside the demonstration capture.
                self._authorize_control()
            except Exception as exc:
                self._fail(f"startup:{exc}")
            return
        if (
            self._pending_home_action is not None
            and time.monotonic_ns() >= self._pending_home_deadline_ns
        ):
            self._fail("home:result_timeout")
            return
        process = self._process
        if process is not None and process.poll() is not None:
            self._fail(f"recorder:unexpected_exit:{process.returncode}")
            return
        if (
            process is not None
            and self._pending_home_action is None
            and (
                self._control_state_name
                in {"TELEOP_ARMED", "POLICY_ARMED", "REPLAY_ARMED", "ACTIVE"}
                or (
                    self._control_state_name == "READY"
                    and self._control_physical_pedal
                )
            )
            and bool(self.get_parameter("auto_authorize_control").value)
            and time.monotonic_ns()
            >= self._next_control_authorization_refresh_ns
        ):
            try:
                self._authorize_control()
            except Exception as exc:
                # A daemon restart can race the refresh. Retrying one second
                # later keeps the collection session alive while never using
                # an expired token for robot motion.
                self._next_control_authorization_refresh_ns = (
                    time.monotonic_ns() + 1_000_000_000
                )
                self.get_logger().warning(
                    f"control authorization refresh failed: {exc}"
                )

    def _fail(self, reason: str) -> None:
        self.get_logger().error(reason)
        try:
            if self._process is not None:
                self._stop(rerecord=True)
        except Exception as stop_exc:
            reason = f"{reason}; discard_failed:{stop_exc}"
        self._publish(f"FAULT:{reason}")
        self._finished = True

    def destroy_node(self) -> bool:
        try:
            if self._process is not None and self._process.poll() is None:
                self._stop(rerecord=False)
        except Exception as exc:
            self.get_logger().error(f"episode shutdown failed: {exc}")
        return super().destroy_node()


def main(args=None) -> None:
    # Keep the context alive while SIGINT finalizes the recorder.  rclpy's
    # default handler shuts the context down first, making the final status
    # publication fail and encouraging a second Ctrl-C during bag flushing.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = EpisodeController()
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
