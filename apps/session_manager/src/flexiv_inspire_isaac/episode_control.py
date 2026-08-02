"""Config-driven multi-episode collection controller.

The controller owns the recorder subprocess lifecycle.  Left/right pedal and
Quest-A events are serialized through one guarded Home transaction so robot
reset motion is never part of a demonstration.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import uuid

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from flexiv_inspire_interfaces.msg import ControlState

from flexiv_inspire_control.ipc_client import RDKIPCClient
from flexiv_inspire_isaac.data_pipeline.manifest import local_minute_timestamp


_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")


class EpisodeController(Node):
    def __init__(self) -> None:
        super().__init__("flexiv_inspire_episode_controller")
        defaults = {
            "sessions_root": "sessions",
            "dataset_name": "dataset",
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
            "home_result_timeout_s": 30.0,
            "recorder_state_timeout_s": 10.0,
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
        self._episode_index = 1
        self._attempt = 1
        self._completed_episodes = 0
        self._current_manifest: Path | None = None
        self._pending_home_request_id: str | None = None
        self._pending_home_deadline_ns = 0
        self._pending_home_action: str | None = None
        self._awaiting_home_authorization = False
        self._routine_control_rearm_pending = False
        self._startup_done = False
        self._finished = False

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
        self.create_timer(0.25, self._tick)
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

    def _state_file(self) -> Path:
        return Path(self._required("runtime_dir")) / "episode-recorder-state.json"

    def _command(self) -> list[str]:
        self._sequence += 1
        collection_timestamp = local_minute_timestamp()
        episode_name = (
            f"episode_{self._episode_index:06d}_attempt_{self._attempt:02d}_"
            f"{collection_timestamp}"
        )
        root = Path(self._required("sessions_root")).expanduser().resolve()
        dataset = str(self.get_parameter("dataset_name").value).strip()
        self._current_manifest = root / dataset / episode_name / "manifest.json"
        command = [
            sys.executable,
            "-m",
            "flexiv_inspire_isaac.data_pipeline.episode_manager",
            "--root",
            str(root),
            "--dataset-name",
            dataset,
            "--episode-index",
            str(self._episode_index),
            "--attempt",
            str(self._attempt),
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
            "--deviceio-socket",
            self._required("deviceio_socket"),
            "--control-state-file",
            str(self._state_file()),
        ]
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

    def _stop(self, *, rerecord: bool) -> bool:
        with self._lock:
            process = self._process
            self._process = None
        if process is None or process.poll() is not None:
            return False
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
        if self._current_manifest is not None and self._current_manifest.is_file():
            manifest = json.loads(self._current_manifest.read_text(encoding="utf-8"))
        completed = bool(manifest.get("completed", False))
        if rerecord and completed:
            raise RuntimeError("discarded episode was incorrectly marked completed")
        if not rerecord and not completed:
            raise RuntimeError(
                f"episode finalization failed: {manifest.get('completion_reason', 'manifest missing')}"
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
            response_kind, response = client.request(kind, payload)
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

    def _on_control_state(self, message: ControlState) -> None:
        """Make the physical pedal behave like a reusable teleop clutch.

        Releasing the pedal still executes the normal measured hardware hold.
        Once that routine hold is visible and the pedal is up, obtain a fresh
        one-time authorization and re-arm teleop.  Fault, tracking, collision,
        limit and malformed-command holds remain latched for manual handling.
        """

        if message.state_name != "HOLD_LATCHED":
            self._routine_control_rearm_pending = False
            return
        if (
            self._routine_control_rearm_pending
            or self._pending_home_action is not None
            or bool(message.physical_pedal)
            or message.hold_reason
            not in {"physical_pedal_released", "source_deadman_released"}
        ):
            return
        self._routine_control_rearm_pending = True
        try:
            self._authorize_control(clear_hold_latched=True)
        except Exception as exc:
            self._routine_control_rearm_pending = False
            self.get_logger().error(f"pedal clutch re-arm failed: {exc}")

    def _authorize_home(self) -> None:
        if not bool(self.get_parameter("auto_authorize_home").value):
            self._send_home_request()
            return
        response = self._request_authorization(
            "authorize_home",
            {
                "session_id": self._required("session_id"),
                "operator_confirmation": "FLEXIV-HOME-MOVE",
                "clear_hold_latched": False,
            },
        )
        outgoing = String()
        outgoing.data = json.dumps(
            {
                "session_id": self._required("session_id"),
                "one_time_token": response["one_time_token"],
                "expires_monotonic_ns": response["expires_monotonic_ns"],
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
        try:
            if command in {"stop", "toggle"}:
                self._begin_home("next")
            elif command == "rerecord":
                self._begin_home("discard")
            elif command == "home":
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
                raise RuntimeError(reason or state)
            if state != "complete":
                return
            action = self._pending_home_action
            self._pending_home_action = None
            self._pending_home_request_id = None
            self._pending_home_deadline_ns = 0
            self._authorize_control()
            if action == "pause":
                self._resume()
            elif action == "discard":
                self._stop(rerecord=True)
                self._attempt += 1
                self._start()
            elif action == "next":
                if self._stop(rerecord=False):
                    self._completed_episodes += 1
                target = int(self.get_parameter("episode_count").value)
                if self._completed_episodes >= target:
                    self._publish_progress("COMPLETE")
                    self._finished = True
                else:
                    self._episode_index += 1
                    self._attempt = 1
                    self._start()
        except Exception as exc:
            self._fail(f"home:{exc}")

    def _tick(self) -> None:
        if not self._startup_done and bool(self.get_parameter("auto_start").value):
            self._startup_done = True
            try:
                self._start()
                self._authorize_control()
            except Exception as exc:
                self._fail(f"startup:{exc}")
            return
        if (
            self._pending_home_action is not None
            and time.monotonic_ns() >= self._pending_home_deadline_ns
        ):
            self._fail("home:result_timeout")
        process = self._process
        if process is not None and process.poll() is not None:
            self._process = None
            self._fail(f"recorder:unexpected_exit:{process.returncode}")

    def _fail(self, reason: str) -> None:
        self.get_logger().error(reason)
        try:
            if self._process is not None and self._process.poll() is None:
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
    rclpy.init(args=args)
    node = EpisodeController()
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
