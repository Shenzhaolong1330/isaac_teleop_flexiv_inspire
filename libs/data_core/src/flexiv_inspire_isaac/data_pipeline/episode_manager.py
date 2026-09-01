"""Runnable episode manager for dual MCAP capture and atomic manifests."""

from __future__ import annotations

import argparse
import base64
import importlib.metadata
import json
import os
import platform
import re
import signal
import subprocess
import sys
import threading
import time
import tomllib
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from isaac_teleop_core.deviceio import default_deviceio_socket

from .manifest import (
    EpisodeManifest,
    StreamStats,
    canonical_yaml_sha256,
    local_minute_timestamp,
    sha256_file,
)
from .native_deviceio import NativeDeviceIOIngress
from .recorder import AsyncMcapRecorder, McapJsonSink, RecordEnvelope

ROS_BAG_TOPICS = (
    "/robot/left_arm/state",
    "/robot/right_arm/state",
    "/robot/left_arm/joint_states",
    "/robot/right_arm/joint_states",
    "/robot/left_arm/tcp_pose",
    "/robot/right_arm/tcp_pose",
    "/robot/left_arm/tcp_twist",
    "/robot/right_arm/tcp_twist",
    "/robot/left_arm/raw_ft",
    "/robot/right_arm/raw_ft",
    "/robot/left_arm/tcp_wrench",
    "/robot/right_arm/tcp_wrench",
    "/robot/left_hand/state",
    "/robot/right_hand/state",
    "/robot/left_hand/joint_states",
    "/robot/right_hand/joint_states",
    "/robot/left_hand/dynamic_joint_states",
    "/robot/right_hand/dynamic_joint_states",
    "/robot/left_hand/tactile_raw",
    "/robot/right_hand/tactile_raw",
    "/camera/head/color/frame",
    "/camera/left_wrist/color/frame",
    "/camera/right_wrist/color/frame",
    "/camera/head/color/image_raw/compressed",
    "/camera/left_wrist/color/image_raw/compressed",
    "/camera/right_wrist/color/image_raw/compressed",
    "/camera/head/depth/image_rect_raw",
    "/camera/left_wrist/depth/image_rect_raw",
    "/camera/right_wrist/depth/image_rect_raw",
    "/camera/head/depth/points",
    "/camera/left_wrist/depth/points",
    "/camera/right_wrist/depth/points",
    "/control/state",
    "/control/requested_command",
    "/control/safe_command",
    "/control/sent_command",
    "/control/command_trace",
    "/control/stop",
    "/xr_teleop/ee_poses",
    "/xr_teleop/controller_data",
    "/xr_teleop/hand",
    "/manus/left/ergonomics",
    "/manus/right/ergonomics",
    "/tf",
    "/tf_static",
    "/teleop/deadman",
    "/command_sources/teleop/command",
    "/command_sources/teleop/heartbeat",
    "/command_sources/policy/command",
    "/command_sources/policy/heartbeat",
    "/command_sources/replay/command",
    "/command_sources/replay/heartbeat",
    "/maintenance/events",
    "/episode/events",
)

EXPECTED_HZ = {
    "robot/left_arm/state": 300.0,
    "robot/right_arm/state": 300.0,
    "robot/left_hand/state": 200.0,
    "robot/right_hand/state": 200.0,
    "robot/left_hand/tactile_raw": 15.0,
    "robot/right_hand/tactile_raw": 15.0,
    "camera/head/color/image_raw/compressed": 15.0,
    "camera/left_wrist/color/image_raw/compressed": 15.0,
    "camera/right_wrist/color/image_raw/compressed": 15.0,
}

TRAINING_DEVICEIO_TOPICS = {
    "/robot/left_arm/state",
    "/robot/right_arm/state",
    "/robot/left_hand/state",
    "/robot/right_hand/state",
    "/robot/left_hand/tactile_raw",
    "/robot/right_hand/tactile_raw",
    "/camera/head/color/image_raw/compressed",
    "/camera/left_wrist/color/image_raw/compressed",
    "/camera/right_wrist/color/image_raw/compressed",
    "/control/sent_command",
}

RIGHT_TRAINING_DEVICEIO_TOPICS = {
    "/robot/right_arm/state",
    "/robot/right_hand/state",
    "/robot/right_hand/tactile_raw",
    "/camera/head/color/image_raw/compressed",
    "/camera/right_wrist/color/image_raw/compressed",
    "/control/sent_command",
}
DEVICEIO_PROFILE_TOPICS = {
    "training": TRAINING_DEVICEIO_TOPICS,
    "right_training": RIGHT_TRAINING_DEVICEIO_TOPICS,
}


def _time_ns(value: Any) -> int:
    return int(value.sec) * 1_000_000_000 + int(value.nanosec)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"base64": base64.b64encode(bytes(value)).decode("ascii")}
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _message_dict(message: Any) -> dict:
    from rosidl_runtime_py.convert import message_to_ordereddict

    return _json_safe(message_to_ordereddict(message))


def _envelope(topic: str, message: Any, payload: Any | None = None) -> RecordEnvelope:
    receive_ns = time.monotonic_ns()
    acquisition = getattr(message, "acquisition", None)
    if acquisition is None:
        source_ns = receive_ns
        mapped_ns = receive_ns
        sequence = int(getattr(message, "sequence", 0))
        valid = True
        reason = ""
        source_domain = host_domain = "host_monotonic"
    else:
        source_ns = _time_ns(acquisition.source_time)
        mapped_raw = _time_ns(acquisition.mapped_host_time)
        mapped_ns = mapped_raw if acquisition.timing_valid and mapped_raw > 0 else None
        sequence = int(acquisition.source_sequence)
        valid = bool(acquisition.valid) and mapped_ns is not None
        reason = str(acquisition.invalid_reason)
        if mapped_ns is None:
            reason = ";".join(filter(None, (reason, "source-clock-unmapped")))
        source_domain = str(acquisition.source_clock_domain)
        host_domain = str(acquisition.host_clock_domain)
    return RecordEnvelope(
        topic=topic,
        source_time_ns=source_ns,
        host_receive_time_ns=receive_ns,
        sequence=sequence,
        valid=valid,
        invalid_reason=reason,
        source_clock_domain=source_domain,
        host_clock_domain=host_domain,
        mapped_host_time_ns=mapped_ns,
        payload=_message_dict(message) if payload is None else payload,
    )


def _runtime_versions() -> dict[str, str]:
    def package(name: str, fallback: str) -> str:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return fallback

    try:
        project_file = Path(__file__).resolve().parents[3] / "pyproject.toml"
        project_version = str(
            tomllib.loads(project_file.read_text())["project"]["version"]
        )
    except Exception:
        project_version = package("flexiv-inspire-isaac", "unknown")
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=project_file.parent,
        ).stdout.strip()
    except Exception:
        git_commit = "unknown"
    return {
        "project": project_version,
        "project_git_commit": git_commit,
        "python": platform.python_version(),
        "ros_distro": os.environ.get("ROS_DISTRO", "unknown"),
        "rmw_implementation": os.environ.get("RMW_IMPLEMENTATION", "unknown"),
        "flexiv_rdk": "1.9.0-pinned",
        "lerobot": "0.6.0-pinned",
        "rerun": package("rerun-sdk", "not-importable-in-recorder-env"),
        "grpcio": package("grpcio", "unknown"),
        "mcap": package("mcap", "unknown"),
        "librealsense": "2.57.7-pinned",
        "cuda_toolkit": "12.8-local-selection",
        "schema": "flexiv-inspire-episode-v1",
    }


def _load_ft_zero_event(path: Path, session_id: str, tool_hash: str) -> dict:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError("F/T zero record is empty")
    candidates = []
    try:
        parsed = json.loads(text)
        candidates = parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        for line in text.splitlines():
            line = line.strip()
            if line:
                candidates.append(json.loads(line))

    selected: dict | None = None
    selected_generation: int | None = None
    invalidating_types = {
        "ft_zero_started",
        "ft_zero_failed",
        "ft_zero_invalidated",
        "rdk_reconnected",
        "rdk_connection_changed",
        "tool_payload_changed",
        "tool_payload_configuration_changed",
        "hardware_session_invalidated",
    }
    for raw in candidates:
        if not isinstance(raw, dict):
            continue
        event = raw
        payload = raw.get("payload")
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                payload = None
        if isinstance(payload, dict):
            event = {**raw, **payload}
        if str(event.get("session_id", "")) != session_id:
            continue
        event_type = str(event.get("event_type", ""))
        event_hash = str(
            event.get(
                "tool_payload_config_hash",
                event.get("tool_configuration_hash", ""),
            )
        )
        if event_type == "ft_zero_completed":
            if bool(event.get("success")) and event_hash == tool_hash:
                selected = event
                raw_generation = event.get("connection_generation")
                selected_generation = (
                    None if raw_generation is None else int(raw_generation)
                )
            else:
                selected = None
                selected_generation = None
            continue
        raw_generation = event.get("connection_generation")
        event_generation = None if raw_generation is None else int(raw_generation)
        if (
            selected is not None
            and selected_generation is not None
            and event_generation is not None
            and event_generation > selected_generation
        ):
            selected = None
            selected_generation = None
        lowered = event_type.lower()
        invalidates = (
            event_type in invalidating_types
            or "reconnect" in lowered
            or ("tool" in lowered and "chang" in lowered)
        )
        if selected is not None and invalidates:
            if (
                event_generation is None
                or selected_generation is None
                or event_generation == selected_generation
                or "reconnect" in lowered
                or "tool" in lowered
            ):
                selected = None
                selected_generation = None
    if selected is None:
        raise ValueError(
            "no currently-valid ft_zero_completed event matches session and tool hash"
        )
    return selected


def _critical_envelope(topic: str, sequence: int, payload: dict) -> RecordEnvelope:
    now = time.monotonic_ns()
    return RecordEnvelope(
        topic=topic,
        source_time_ns=now,
        host_receive_time_ns=now,
        sequence=sequence,
        valid=True,
        payload=payload,
        source_clock_domain="host_monotonic",
        host_clock_domain="host_monotonic",
        mapped_host_time_ns=now,
    )


class RosbagProcess:
    def __init__(
        self, output: Path, extra_topics: tuple[str, ...] = (), *, enabled: bool = True
    ) -> None:
        self.output = output
        self.extra_topics = extra_topics
        self.enabled = enabled
        self.node_name = f"flexiv_inspire_rosbag_{uuid.uuid4().hex[:12]}"
        self.process: subprocess.Popen | None = None
        self._started = False
        self._paused = False
        self._early_exit_code: int | None = None

    def start(self) -> None:
        if not self.enabled:
            return
        command = [
            "ros2",
            "bag",
            "record",
            "--disable-keyboard-controls",
            "--node-name",
            self.node_name,
            "-s",
            "mcap",
            "-o",
            str(self.output),
            "--topics",
            *ROS_BAG_TOPICS,
            *self.extra_topics,
        ]
        self.process = subprocess.Popen(command, start_new_session=True)
        self._started = False
        self._paused = False
        self._early_exit_code = None
        time.sleep(0.5)
        if self.process.poll() is not None:
            return_code = self.process.wait()
            self._early_exit_code = return_code
            self.process = None
            raise RuntimeError(f"ros2 bag record exited early with {return_code}")
        self._started = True

    def _set_paused(self, node: Any, *, paused: bool, timeout_s: float = 5.0) -> None:
        if not self.started:
            raise RuntimeError("cannot pause/resume a stopped rosbag recorder")
        if self._paused == paused:
            return
        import rclpy
        from rosbag2_interfaces.srv import Pause, Resume

        service_type = Pause if paused else Resume
        operation = "pause" if paused else "resume"
        client = node.create_client(service_type, f"/{self.node_name}/{operation}")
        try:
            if not client.wait_for_service(timeout_sec=timeout_s):
                raise TimeoutError(f"rosbag {operation} service was not available")
            future = client.call_async(service_type.Request())
            rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_s)
            if not future.done():
                raise TimeoutError(f"rosbag {operation} service timed out")
            exception = future.exception()
            if exception is not None:
                raise RuntimeError(f"rosbag {operation} failed: {exception}")
            if future.result() is None:
                raise RuntimeError(f"rosbag {operation} returned no response")
            self._paused = paused
        finally:
            node.destroy_client(client)

    def pause(self, node: Any, timeout_s: float = 5.0) -> None:
        self._set_paused(node, paused=True, timeout_s=timeout_s)

    def resume(self, node: Any, timeout_s: float = 5.0) -> None:
        self._set_paused(node, paused=False, timeout_s=timeout_s)

    @property
    def started(self) -> bool:
        return (
            self._started and self.process is not None and self.process.poll() is None
        )

    def stop_and_validate(self, timeout_s: float = 15.0) -> None:
        if self.process is None:
            detail = (
                ""
                if self._early_exit_code is None
                else f"; early exit {self._early_exit_code}"
            )
            raise RuntimeError(f"rosbag was not started{detail}")
        if self.process.poll() is not None:
            return_code = self.process.returncode
            raise RuntimeError(f"rosbag already exited with {return_code}")
        os.killpg(self.process.pid, signal.SIGINT)
        try:
            return_code = self.process.wait(timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(3.0)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(3.0)
            raise TimeoutError("rosbag did not finish cleanly after SIGINT")
        self._started = False
        self._paused = False
        if return_code != 0:
            raise RuntimeError(f"rosbag exited with {return_code}")
        metadata = self.output / "metadata.yaml"
        storage = list(self.output.glob("*.mcap"))
        if (
            not metadata.is_file()
            or not storage
            or any(path.stat().st_size == 0 for path in storage)
        ):
            raise RuntimeError("rosbag MCAP metadata/storage validation failed")


class EpisodeSession:
    def __init__(
        self,
        root: Path,
        session_id: str,
        calibration_paths: dict[str, Path],
        tool_config: Path,
        ft_zero_record: Path,
        camera_recording_mode: str = "jpeg",
        deviceio_capture_layer: str = "native-pre-dds",
        *,
        dataset_name: str = "",
        storage_subdirectory: str = "",
        episode_index: int = 0,
        attempt: int = 1,
        task_name: str = "",
        task_description: str = "",
        episode_directory_name: str = "",
        collection_timestamp_local: str = "",
        recording_gate: threading.Event | None = None,
        record_ros_mcap: bool = True,
        deviceio_profile: str = "full",
        record_only_while_pedal_pressed: bool = False,
        expected_hz: dict[str, float] | None = None,
    ) -> None:
        if camera_recording_mode != "jpeg":
            raise ValueError(
                "episode manager records the atomic ROS JPEG mirror only; raw_rgb must be recorded by the camera DeviceIO sink"
            )
        if deviceio_profile not in {"training", "right_training", "full"}:
            raise ValueError(
                "deviceio_profile must be training, right_training or full"
            )
        # Complete every read-only authorization/hash check before creating an
        # episode directory, so a rejected F/T record cannot leave an orphan.
        tool_hash = canonical_yaml_sha256(tool_config)
        tool_file_hash = sha256_file(tool_config)
        ft_event = _load_ft_zero_event(ft_zero_record, session_id, tool_hash)
        calibration_hashes = {
            name: sha256_file(path) for name, path in calibration_paths.items()
        }
        versions = _runtime_versions()
        self.episode_uuid = str(uuid.uuid4())
        collection_timestamp_local = (
            collection_timestamp_local or local_minute_timestamp()
        )
        if not re.fullmatch(r"\d{8}_\d{4}", collection_timestamp_local):
            raise ValueError("collection_timestamp_local must use YYYYMMDD_HHMM")
        task_name = task_name.strip()
        if task_name and not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", task_name
        ):
            raise ValueError("task_name must be one safe directory name")
        dataset_root = root / dataset_name if dataset_name else root
        # Keep the storage layer separate from the dataset identity written to
        # the manifest. The collector uses ``raw``; the empty default keeps
        # direct EpisodeSession callers and historical layouts compatible.
        storage_subdirectory = storage_subdirectory.strip()
        if storage_subdirectory and not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", storage_subdirectory
        ):
            raise ValueError("storage_subdirectory must be one safe directory name")
        storage_root = (
            dataset_root / storage_subdirectory
            if storage_subdirectory
            else dataset_root
        )
        default_directory_name = (
            f"{task_name}_episode_{episode_index:03d}_{collection_timestamp_local}"
            if task_name and episode_index > 0
            else f"episode_{collection_timestamp_local}"
        )
        self.directory = storage_root / (
            episode_directory_name or default_directory_name
        )
        self.directory.mkdir(parents=True, exist_ok=False)
        self.manifest_path = self.directory / "manifest.json"
        self.device_path = self.directory / "deviceio.mcap"
        self.ros_path = self.directory / "ros_mcap"
        versions["camera_recording_mode_declared"] = camera_recording_mode
        versions["deviceio_capture_layer"] = deviceio_capture_layer
        self.manifest = EpisodeManifest(
            schema_version="1",
            episode_uuid=self.episode_uuid,
            session_id=session_id,
            deviceio_mcap=self.device_path.name,
            ros_mcap=self.ros_path.name if record_ros_mcap else "",
            software_versions=versions,
            calibration_hashes=calibration_hashes,
            tool_configuration_hash=tool_hash,
            tool_configuration_file_hash=tool_file_hash,
            ft_zero_event=ft_event,
            deviceio_capture_layer=deviceio_capture_layer,
            dataset_name=dataset_name,
            episode_index=episode_index,
            attempt=attempt,
            collection_timestamp_local=collection_timestamp_local,
            task_name=task_name,
            task_description=task_description,
        )
        self.manifest.write_atomic(self.manifest_path)
        self._critical_sequence = 0
        self._deviceio_profile = deviceio_profile
        self._record_ros_mcap = record_ros_mcap
        self._record_only_while_pedal_pressed = bool(record_only_while_pedal_pressed)
        self._expected_hz = {**EXPECTED_HZ, **(expected_hz or {})}
        # Start closed in motion-only mode. The first physical middle-pedal
        # press opens this gate through /teleop/deadman.
        self._motion_recording_enabled = not self._record_only_while_pedal_pressed
        self._last_sequence_by_topic: dict[str, int] = {}
        self._stats_lock = threading.Lock()
        self._recording_gate = recording_gate or threading.Event()
        self._recording_gate.set()
        self._pause_started_ns = 0
        self.recorder = None
        try:
            self.recorder = AsyncMcapRecorder(McapJsonSink(self.device_path))
            self.recorder.start()
            self._submit_unchecked(
                _critical_envelope(
                    "/maintenance/ft_zero_event",
                    self._next_critical_sequence(),
                    ft_event,
                ),
                critical=True,
            )
            self._submit_unchecked(
                _critical_envelope(
                    "/episode/events",
                    self._next_critical_sequence(),
                    {
                        "event_type": "episode_started",
                        "episode_uuid": self.episode_uuid,
                        "session_id": session_id,
                        "camera_recording_mode_declared": camera_recording_mode,
                        "capture_layer": deviceio_capture_layer,
                    },
                ),
                critical=True,
            )
        except Exception as exc:
            close_error = ""
            try:
                if self.recorder is not None:
                    self.recorder.close()
            except Exception as close_exc:
                close_error = f"; recorder-close: {close_exc}"
            self.manifest.completed = False
            self.manifest.completion_reason = f"startup-fault:{exc}{close_error}"
            self.manifest.write_atomic(self.manifest_path)
            raise

    def _next_critical_sequence(self) -> int:
        self._critical_sequence += 1
        return self._critical_sequence

    def submit(self, envelope: RecordEnvelope, *, critical: bool = False) -> None:
        profile_topics = DEVICEIO_PROFILE_TOPICS.get(self._deviceio_profile)
        if profile_topics is not None and envelope.topic not in profile_topics:
            return
        if self._record_only_while_pedal_pressed and not self._motion_recording_enabled:
            with self._stats_lock:
                self.manifest.suppressed_samples += 1
            return
        if not self._recording_gate.is_set():
            with self._stats_lock:
                self.manifest.suppressed_samples += 1
            return
        self._submit_unchecked(envelope, critical=critical)

    def set_motion_recording(self, enabled: bool) -> None:
        """Open or close the data gate from the physical middle pedal."""
        self._motion_recording_enabled = bool(enabled)

    def _submit_unchecked(
        self, envelope: RecordEnvelope, *, critical: bool = True
    ) -> None:
        self.recorder.submit(envelope, critical=critical)
        name = envelope.topic.lstrip("/")
        with self._stats_lock:
            stats = self.manifest.streams.setdefault(
                name, StreamStats(self._expected_hz.get(name, 0.0))
            )
            stats.samples += 1
            stats.invalid += int(not envelope.valid)
            previous_sequence = self._last_sequence_by_topic.get(name)
            if (
                previous_sequence is not None
                and envelope.sequence > previous_sequence + 1
            ):
                stats.drops += envelope.sequence - previous_sequence - 1
            if previous_sequence is None or envelope.sequence > previous_sequence:
                self._last_sequence_by_topic[name] = envelope.sequence
            if stats.first_source_time_ns is None:
                stats.first_source_time_ns = envelope.source_time_ns
            stats.last_source_time_ns = envelope.source_time_ns

    def pause(self, *, reason: str) -> None:
        if self._pause_started_ns:
            return
        self._recording_gate.clear()
        self._pause_started_ns = time.monotonic_ns()
        self.manifest.pause_count += 1
        self._submit_unchecked(
            _critical_envelope(
                "/episode/events",
                self._next_critical_sequence(),
                {
                    "event_type": "episode_paused",
                    "episode_uuid": self.episode_uuid,
                    "session_id": self.manifest.session_id,
                    "reason": reason,
                },
            )
        )

    def resume(self, *, reason: str) -> None:
        if self._recording_gate.is_set():
            return
        now = time.monotonic_ns()
        if self._pause_started_ns:
            self.manifest.paused_duration_ns += now - self._pause_started_ns
        self._pause_started_ns = 0
        self._submit_unchecked(
            _critical_envelope(
                "/episode/events",
                self._next_critical_sequence(),
                {
                    "event_type": "episode_resumed",
                    "episode_uuid": self.episode_uuid,
                    "session_id": self.manifest.session_id,
                    "reason": reason,
                },
            )
        )
        self._recording_gate.set()

    def _finish_pause_interval(self) -> None:
        if not self._recording_gate.is_set() and self._pause_started_ns:
            self.manifest.paused_duration_ns += (
                time.monotonic_ns() - self._pause_started_ns
            )
            self._pause_started_ns = 0

    def submit_native(self, document: Any) -> None:
        """Accept one validated producer-native envelope from the local ingress."""
        topic = str(document["topic"])
        if topic.startswith("/_deviceio/source_stats/"):
            producer = str(document["producer"])
            payload = document.get("payload")
            if isinstance(payload, dict):
                with self._stats_lock:
                    self.manifest.native_source_stats[producer] = dict(payload)
            return
        profile_topics = DEVICEIO_PROFILE_TOPICS.get(self._deviceio_profile)
        if profile_topics is not None and topic not in profile_topics:
            return
        mapped = document.get("mapped_host_time_ns")
        if not bool(document.get("timing_valid", False)):
            mapped = None
        envelope = RecordEnvelope(
            topic=topic,
            source_time_ns=int(document["source_time_ns"]),
            host_receive_time_ns=int(document["host_receive_time_ns"]),
            sequence=int(document["sequence"]),
            valid=bool(document["valid"]),
            invalid_reason=str(document.get("invalid_reason", "")),
            source_clock_domain=str(document["source_clock_domain"]),
            host_clock_domain=str(document["host_clock_domain"]),
            mapped_host_time_ns=None if mapped is None else int(mapped),
            payload=document["payload"],
        )
        critical = topic.startswith(("/control/", "/episode/", "/maintenance/"))
        self.submit(envelope, critical=critical)

    def _update_stream_stats(self) -> None:
        recorder_stats = self.recorder.stats()
        for topic, drops in recorder_stats["dropped_by_topic"].items():
            name = topic.lstrip("/")
            stats = self.manifest.streams.setdefault(
                name, StreamStats(self._expected_hz.get(name, 0.0))
            )
            stats.drops += int(drops)
        for stats in self.manifest.streams.values():
            if (
                stats.samples > 1
                and stats.first_source_time_ns is not None
                and stats.last_source_time_ns is not None
                and stats.last_source_time_ns > stats.first_source_time_ns
            ):
                stats.observed_hz = (
                    (stats.samples - 1)
                    * 1e9
                    / (stats.last_source_time_ns - stats.first_source_time_ns)
                )

    def _required_stream_errors(self) -> list[str]:
        profile = getattr(self, "_deviceio_profile", "training")
        required = tuple(
            topic.lstrip("/")
            for topic in DEVICEIO_PROFILE_TOPICS.get(
                profile, TRAINING_DEVICEIO_TOPICS
            )
        )
        errors = []
        for name in required:
            stats = self.manifest.streams.get(name)
            valid_samples = 0 if stats is None else stats.samples - stats.invalid
            if valid_samples <= 0:
                errors.append(name)
        return errors

    def abort(self, *, reason: str) -> None:
        errors: list[str] = []
        self._finish_pause_interval()
        try:
            self._submit_unchecked(
                _critical_envelope(
                    "/episode/events",
                    self._next_critical_sequence(),
                    {
                        "event_type": "episode_failed",
                        "episode_uuid": self.episode_uuid,
                        "session_id": self.manifest.session_id,
                        "reason": reason,
                    },
                ),
                critical=True,
            )
        except Exception as exc:
            errors.append(f"episode-failure-event: {exc}")
        try:
            self.recorder.close()
        except Exception as exc:
            errors.append(f"deviceio: {exc}")
        self._update_stream_stats()
        self.manifest.completed = False
        self.manifest.completion_reason = "; ".join([reason, *errors])
        self.manifest.write_atomic(self.manifest_path)

    def finish(self, rosbag: RosbagProcess, *, reason: str) -> None:
        errors: list[str] = []
        self._finish_pause_interval()
        try:
            self._submit_unchecked(
                _critical_envelope(
                    "/episode/events",
                    self._next_critical_sequence(),
                    {
                        "event_type": (
                            "episode_rerecord_requested"
                            if reason == "rerecord-requested"
                            else "episode_stopped"
                        ),
                        "episode_uuid": self.episode_uuid,
                        "session_id": self.manifest.session_id,
                        "reason": reason,
                    },
                ),
                critical=True,
            )
        except Exception as exc:
            errors.append(f"episode-stop-event: {exc}")
        if not getattr(rosbag, "enabled", True):
            pass
        elif rosbag.started:
            try:
                rosbag.stop_and_validate()
            except Exception as exc:
                errors.append(f"rosbag: {exc}")
        elif not reason.startswith("fault:"):
            errors.append("rosbag: recorder was not started")
        try:
            self.recorder.close()
        except Exception as exc:
            errors.append(f"deviceio: {exc}")
        self._update_stream_stats()
        rerecord_requested = reason == "rerecord-requested"
        missing_streams = self._required_stream_errors()
        # An operator-discarded attempt is deliberately not a demonstration.
        # It can legitimately have no sent command (for example, left pedal
        # was pressed before the first teleop clutch), so retain it for audit
        # without failing the controller that must immediately start attempt 2.
        if (
            missing_streams
            and not reason.startswith("fault:")
            and not rerecord_requested
        ):
            errors.append(
                "required streams have no valid samples: " + ",".join(missing_streams)
            )
        # Preserve the raw capture for audit but never label a re-record as a
        # completed demonstration eligible for training or replay.
        terminal_fault = reason.startswith("fault:")
        self.manifest.completed = not errors and not terminal_fault
        if rerecord_requested:
            self.manifest.completed = False
        self.manifest.completion_reason = (
            reason if not errors else "; ".join([reason, *errors])
        )
        self.manifest.write_atomic(self.manifest_path)
        if errors or terminal_fault:
            raise RuntimeError(self.manifest.completion_reason)


def build_node(session: EpisodeSession, *, record_ros_mirror: bool = True):
    from flexiv_inspire_interfaces.msg import (
        ArmState,
        BimanualCommand,
        CameraFrame,
        CommandTrace,
        ControlState,
        EpisodeEvent,
        HandState,
        TactileFrame,
    )
    from geometry_msgs.msg import PoseArray
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Bool, ByteMultiArray

    class EpisodeRecorderNode(Node):
        def __init__(self) -> None:
            super().__init__("flexiv_inspire_episode_manager")
            # DeviceIO captures producer-side robot observations in native mode.
            # XR/raw command inputs have no DeviceIO producer, so always record
            # them here into the same MCAP truth file.
            self.create_subscription(
                PoseArray,
                "/xr_teleop/ee_poses",
                lambda msg: self._record("/xr_teleop/ee_poses", msg),
                qos_profile_sensor_data,
            )
            self.create_subscription(
                ByteMultiArray,
                "/xr_teleop/controller_data",
                lambda msg: self._record("/xr_teleop/controller_data", msg),
                qos_profile_sensor_data,
            )
            self.create_subscription(
                PoseArray,
                "/xr_teleop/hand",
                lambda msg: self._record("/xr_teleop/hand", msg),
                qos_profile_sensor_data,
            )
            for side in ("left", "right"):
                topic = f"/manus/{side}/ergonomics"
                self.create_subscription(
                    JointState,
                    topic,
                    lambda msg, selected=topic: self._record(selected, msg),
                    qos_profile_sensor_data,
                )
            self.create_subscription(
                Bool,
                "/teleop/deadman",
                self._deadman,
                qos_profile_sensor_data,
            )
            for source in ("teleop", "policy", "replay"):
                self.create_subscription(
                    BimanualCommand,
                    f"/command_sources/{source}/command",
                    lambda msg, selected=source: self._record(
                        f"/command_sources/{selected}/command", msg, True
                    ),
                    1,
                )
            if not record_ros_mirror:
                return
            for side in ("left", "right"):
                self.create_subscription(
                    ArmState,
                    f"/robot/{side}_arm/state",
                    lambda msg, selected=side: self._arm(selected, msg),
                    qos_profile_sensor_data,
                )
                self.create_subscription(
                    HandState,
                    f"/robot/{side}_hand/state",
                    lambda msg, selected=side: self._record(
                        f"/robot/{selected}_hand/state", msg
                    ),
                    qos_profile_sensor_data,
                )
                self.create_subscription(
                    TactileFrame,
                    f"/robot/{side}_hand/tactile_raw",
                    lambda msg, selected=side: self._record(
                        f"/robot/{selected}_hand/tactile_raw", msg
                    ),
                    qos_profile_sensor_data,
                )
            for camera in ("head", "left_wrist", "right_wrist"):
                self.create_subscription(
                    CameraFrame,
                    f"/camera/{camera}/color/frame",
                    lambda msg, selected=camera: self._camera(selected, msg),
                    qos_profile_sensor_data,
                )
            self.create_subscription(
                ControlState,
                "/control/state",
                lambda msg: self._record("/control/state", msg, True),
                1,
            )
            for topic in ("requested_command", "safe_command", "sent_command"):
                self.create_subscription(
                    BimanualCommand,
                    f"/control/{topic}",
                    lambda msg, selected=topic: self._record(
                        f"/control/{selected}", msg, True
                    ),
                    1,
                )
            self.create_subscription(
                CommandTrace,
                "/control/command_trace",
                lambda msg: self._record("/control/command_trace", msg, True),
                1,
            )
            self.create_subscription(
                EpisodeEvent,
                "/episode/events",
                lambda msg: self._record("/episode/events", msg, True),
                10,
            )
            self.create_subscription(
                EpisodeEvent,
                "/maintenance/events",
                lambda msg: self._record("/maintenance/events", msg, True),
                10,
            )

        def _record(self, topic: str, message: Any, critical: bool = False) -> None:
            session.submit(_envelope(topic, message), critical=critical)

        def _deadman(self, message: Any) -> None:
            # Update before submitting so a press starts capture immediately
            # and a release stops capture immediately.
            session.set_motion_recording(bool(message.data))
            self._record("/teleop/deadman", message)

        def _arm(self, side: str, message: Any) -> None:
            common = _envelope(f"/robot/{side}_arm/state", message)
            session.submit(common)
            pose = message.tcp_pose
            pose_payload = {
                "xyz": [pose.position.x, pose.position.y, pose.position.z],
                "quaternion_xyzw": [
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                ],
            }
            session.submit(
                _envelope(f"/robot/{side}_arm/tcp_pose", message, pose_payload)
            )
            twist = message.tcp_twist
            session.submit(
                _envelope(
                    f"/robot/{side}_arm/tcp_twist",
                    message,
                    {
                        "values": [
                            twist.linear.x,
                            twist.linear.y,
                            twist.linear.z,
                            twist.angular.x,
                            twist.angular.y,
                            twist.angular.z,
                        ]
                    },
                )
            )
            for name, wrench in (
                ("raw_ft", message.raw_ft),
                ("tcp_wrench", message.tcp_wrench),
            ):
                session.submit(
                    _envelope(
                        f"/robot/{side}_arm/{name}",
                        message,
                        {
                            "values": [
                                wrench.force.x,
                                wrench.force.y,
                                wrench.force.z,
                                wrench.torque.x,
                                wrench.torque.y,
                                wrench.torque.z,
                            ]
                        },
                    )
                )

        def _camera(self, camera: str, message: Any) -> None:
            valid_name = str(message.camera) == camera
            payload = {
                "jpeg_b64": base64.b64encode(bytes(message.image.data)).decode("ascii"),
                "encoding": "jpeg",
                "width": int(message.width),
                "height": int(message.height),
            }
            envelope = _envelope(
                f"/camera/{camera}/color/image_raw/compressed", message, payload
            )
            if not valid_name or not str(message.image.format).lower().startswith(
                "jpeg"
            ):
                envelope = RecordEnvelope(
                    **{
                        **asdict(envelope),
                        "valid": False,
                        "invalid_reason": ";".join(
                            filter(
                                None,
                                (
                                    envelope.invalid_reason,
                                    "camera-name-or-encoding-mismatch",
                                ),
                            )
                        ),
                    }
                )
            session.submit(envelope)

    return EpisodeRecorderNode()


def _parse_mapping(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("calibration must be NAME=PATH")
        name, raw_path = value.split("=", 1)
        path = Path(raw_path).resolve(strict=True)
        result[name] = path
    if not result:
        raise ValueError("at least one --calibration NAME=PATH is required")
    return result


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Atomic DeviceIO+ROS MCAP episode recorder"
    )
    parser.add_argument("--root", type=Path, default=Path("episodes"))
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--tool-config", type=Path, required=True)
    parser.add_argument("--ft-zero-record", type=Path, required=True)
    parser.add_argument(
        "--calibration", action="append", default=[], metavar="NAME=PATH"
    )
    parser.add_argument(
        "--camera-recording-mode", choices=("jpeg", "raw_rgb"), default="jpeg"
    )
    parser.add_argument(
        "--deviceio-mode",
        choices=("native", "post-dds-mirror"),
        default="native",
        help="native records producer-side observations before DDS; mirror is compatibility mode",
    )
    parser.add_argument(
        "--deviceio-socket", type=Path, default=default_deviceio_socket()
    )
    parser.add_argument(
        "--deviceio-profile",
        choices=("training", "right_training", "full"),
        default="full",
    )
    parser.add_argument("--no-ros-mcap", action="store_true")
    parser.add_argument("--record-only-while-pedal-pressed", action="store_true")
    parser.add_argument("--expected-camera-hz", type=float, default=15.0)
    parser.add_argument("--expected-action-hz", type=float, default=0.0)
    parser.add_argument("--extra-topic", action="append", default=[])
    parser.add_argument("--duration-s", type=float, default=0.0)
    parser.add_argument("--dataset-name", default="")
    parser.add_argument(
        "--storage-subdirectory",
        default="",
        help="optional child directory below the dataset root (for example: raw)",
    )
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--task-name", default="")
    parser.add_argument("--task-description", default="")
    parser.add_argument("--episode-directory-name", default="")
    parser.add_argument("--collection-timestamp-local", default="")
    parser.add_argument("--control-state-file", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    options = _parse_args(argv)
    calibrations = _parse_mapping(options.calibration)
    tool_config = options.tool_config.resolve(strict=True)
    ft_record = options.ft_zero_record.resolve(strict=True)
    episode: EpisodeSession | None = None
    rosbag: RosbagProcess | None = None
    node = None
    ingress: NativeDeviceIOIngress | None = None
    rclpy_module = None
    reason = "operator-stop"
    failure: Exception | None = None
    rerecord_requested = threading.Event()
    pause_requested = threading.Event()
    resume_requested = threading.Event()
    recording_gate = threading.Event()
    recording_gate.set()
    previous_usr1_handler = None
    previous_usr2_handler = None
    previous_hup_handler = None

    def _request_rerecord(_signum, _frame) -> None:
        rerecord_requested.set()

    def _request_pause(_signum, _frame) -> None:
        # Close the producer gate in the signal callback. The parent waits for
        # the PAUSED state file before it is allowed to request robot motion.
        recording_gate.clear()
        pause_requested.set()

    def _request_resume(_signum, _frame) -> None:
        resume_requested.set()

    def _write_control_state(state: str, reason_text: str = "") -> None:
        if options.control_state_file is None:
            return
        path = options.control_state_file
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "state": state,
                    "reason": reason_text,
                    "episode_uuid": "" if episode is None else episode.episode_uuid,
                    "episode_directory": ""
                    if episode is None
                    else str(episode.directory),
                },
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        os.replace(temporary, path)

    try:
        # SIGUSR1 is reserved for an explicit left-pedal re-record request.
        # Install it before recorder startup to preserve an atomic manifest.
        if hasattr(signal, "SIGUSR1"):
            previous_usr1_handler = signal.signal(signal.SIGUSR1, _request_rerecord)
        if hasattr(signal, "SIGUSR2"):
            previous_usr2_handler = signal.signal(signal.SIGUSR2, _request_pause)
        if hasattr(signal, "SIGHUP"):
            previous_hup_handler = signal.signal(signal.SIGHUP, _request_resume)
        episode = EpisodeSession(
            options.root.resolve(),
            options.session_id,
            calibrations,
            tool_config,
            ft_record,
            options.camera_recording_mode,
            "native-pre-dds"
            if options.deviceio_mode == "native"
            else "post-dds-typed-mirror",
            dataset_name=options.dataset_name,
            storage_subdirectory=options.storage_subdirectory,
            episode_index=options.episode_index,
            attempt=options.attempt,
            task_name=options.task_name,
            task_description=options.task_description,
            episode_directory_name=options.episode_directory_name,
            collection_timestamp_local=options.collection_timestamp_local,
            recording_gate=recording_gate,
            record_ros_mcap=not options.no_ros_mcap,
            deviceio_profile=options.deviceio_profile,
            record_only_while_pedal_pressed=(options.record_only_while_pedal_pressed),
            expected_hz={
                "camera/head/color/image_raw/compressed": options.expected_camera_hz,
                "camera/left_wrist/color/image_raw/compressed": (
                    options.expected_camera_hz
                ),
                "camera/right_wrist/color/image_raw/compressed": (
                    options.expected_camera_hz
                ),
                "control/sent_command": options.expected_action_hz,
            },
        )
        if options.deviceio_mode == "native":
            ingress = NativeDeviceIOIngress(
                options.deviceio_socket, episode.submit_native
            )
            ingress.start()
        rosbag = RosbagProcess(
            episode.ros_path,
            tuple(options.extra_topic),
            enabled=not options.no_ros_mcap,
        )
        import rclpy as rclpy_module

        rclpy_module.init(args=None)
        node = build_node(episode, record_ros_mirror=options.deviceio_mode != "native")
        rosbag.start()
        _write_control_state("RECORDING")
        start = time.monotonic()
        while rclpy_module.ok():
            rclpy_module.spin_once(node, timeout_sec=0.1)
            if pause_requested.is_set():
                pause_requested.clear()
                if rosbag.enabled:
                    rosbag.pause(node)
                episode.pause(reason="guarded-home")
                _write_control_state("PAUSED")
            if resume_requested.is_set():
                resume_requested.clear()
                if rosbag.enabled:
                    rosbag.resume(node)
                episode.resume(reason="guarded-home-complete")
                _write_control_state("RECORDING")
            if rerecord_requested.is_set():
                reason = "rerecord-requested"
                break
            if ingress is not None:
                ingress.check_health()
            if (
                options.duration_s > 0
                and time.monotonic() - start >= options.duration_s
            ):
                reason = "duration-complete"
                break
    except KeyboardInterrupt:
        reason = "operator-stop"
    except Exception as exc:
        # rclpy's SIGINT handler can shut down the context before
        # ``spin_once`` returns.  Jazzy then raises ExternalShutdownException
        # instead of KeyboardInterrupt.  The operator explicitly requested a
        # normal stop, so finalize this attempt rather than printing a
        # traceback and labelling the recording a fault.
        if type(exc).__name__ == "ExternalShutdownException":
            reason = "operator-stop"
        else:
            reason = f"fault:{exc}"
            failure = exc
    finally:
        try:
            _write_control_state("STOPPING", reason)
        except Exception:
            pass
        cleanup_errors: list[str] = []
        if ingress is not None:
            try:
                ingress_stats = asdict(ingress.stats())
                if episode is not None:
                    episode.manifest.native_source_stats["ingress"] = ingress_stats
                ingress.close()
            except Exception as exc:
                cleanup_errors.append(f"native-deviceio: {exc}")
        if node is not None:
            try:
                node.destroy_node()
            except Exception as exc:
                cleanup_errors.append(f"node-destroy: {exc}")
        if rclpy_module is not None:
            try:
                if rclpy_module.ok():
                    rclpy_module.shutdown()
            except Exception as exc:
                cleanup_errors.append(f"rclpy-shutdown: {exc}")
        if cleanup_errors and not reason.startswith("fault:"):
            reason = "fault:" + "; ".join(cleanup_errors)
        if episode is not None:
            try:
                if rosbag is not None and (rosbag.started or not rosbag.enabled):
                    episode.finish(rosbag, reason=reason)
                else:
                    episode.abort(reason=reason)
            except Exception as exc:
                cleanup_errors.append(f"episode-finalize: {exc}")
            if cleanup_errors and failure is None:
                failure = RuntimeError("; ".join(cleanup_errors))
        if previous_usr1_handler is not None:
            signal.signal(signal.SIGUSR1, previous_usr1_handler)
        if previous_usr2_handler is not None:
            signal.signal(signal.SIGUSR2, previous_usr2_handler)
        if previous_hup_handler is not None:
            signal.signal(signal.SIGHUP, previous_hup_handler)
        if options.control_state_file is not None:
            try:
                _write_control_state("STOPPED", reason)
            except Exception:
                pass
    if episode is not None:
        print(str(episode.directory))
    if failure is not None:
        raise failure
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
