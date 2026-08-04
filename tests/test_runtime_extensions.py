import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from isaac_teleop_core.rotation6d import RotationError, rotation6d_to_matrix
from flexiv_inspire_isaac.cameras.capture import CameraFrame, RealSenseClockMapper, RealSenseRgbCapture
from flexiv_inspire_isaac.cameras.config import CameraConfig
from flexiv_inspire_isaac.data_pipeline.episode_manager import EpisodeSession, RosbagProcess, _load_ft_zero_event
from flexiv_inspire_isaac.data_pipeline.lerobot_v3 import _validated_action
from flexiv_inspire_isaac.data_pipeline.manifest import canonical_yaml_sha256, sha256_file
from flexiv_inspire_isaac.data_pipeline.mcap_input import _flatten_safe_command, _flatten_safe_command_checked
from flexiv_inspire_isaac.data_pipeline.recorder import AsyncMcapRecorder, RecordEnvelope
from flexiv_inspire_isaac.dftp.ros_node import command_source_matches
from flexiv_inspire_isaac.policy_api.models import ActionChunk, ActionPoint, validate_action_chunk
from flexiv_inspire_isaac.policy_api.ros_adapter import Snapshot, _hand_observation, remaining_action_timing


IDENTITY = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]


def action_values(rot6d=IDENTITY):
    return [0.0, 0.0, 0.0, *rot6d, 0.0, 0.0, 0.0, *rot6d, *([100.0] * 12)]


def action_chunk(values=None, receive_ns=1_000_000_000, ttl_ns=500_000_000):
    return ActionChunk(1, "lease", "session", 1, 0, receive_ns, ttl_ns, "world", True,
                       (ActionPoint(0.1, tuple(action_values() if values is None else values)),))


def test_rotation_validation_is_shared_by_policy_and_lerobot():
    valid = action_values([1e-7, 0, 0, 0, 1e-7, 0])
    validate_action_chunk(action_chunk(valid), now_ns=1_000_000_001)
    _validated_action(valid)
    invalid = action_values([1.0, 0, 0, 1e9, 1.0, 0])
    with pytest.raises(RotationError):
        rotation6d_to_matrix(invalid[3:9])
    with pytest.raises(RotationError):
        validate_action_chunk(action_chunk(invalid), now_ns=1_000_000_001)
    with pytest.raises(RotationError):
        _validated_action(invalid)


def test_policy_adapter_import_constructs_snapshot():
    snapshot = Snapshot()
    assert snapshot.arms == {} and snapshot.camera_frames == {}


def test_policy_hand_field_timing_preserves_async_reads():
    def stamp(ns):
        return SimpleNamespace(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)

    def acquisition(mapped_ns):
        return SimpleNamespace(
            source_time=stamp(mapped_ns),
            host_receive_time=stamp(mapped_ns + 10),
            mapped_host_time=stamp(mapped_ns),
            acquisition_start=stamp(mapped_ns - 5),
            acquisition_end=stamp(mapped_ns + 5),
            source_sequence=1,
            valid=True,
            invalid_reason="",
            source_clock_domain="host_monotonic",
            host_clock_domain="host_monotonic",
            timing_valid=True,
        )

    message = SimpleNamespace(
        angle=[1] * 6, position=[2] * 6, actual_force=[3] * 6,
        current=[4] * 6, temperature=[5] * 6, error=[0] * 6, status=[1] * 6,
        acquisition=acquisition(1_000),
    )
    fields = ("angle", "position", "actual_force", "current", "temperature", "error", "status")
    for index, field in enumerate(fields):
        setattr(message, f"{field}_timing", acquisition(1_000 + index * 100))
    result = _hand_observation(message, now_ns=3_000)
    assert [result.field_timing[field].age_ns for field in fields] == [
        2_000 - index * 100 for index in range(7)
    ]


def test_policy_adapter_subtracts_transport_delay_from_ttl_and_offsets():
    remaining, offsets = remaining_action_timing(action_chunk(), 1_050_000_000)
    assert remaining == 450_000_000
    assert offsets == (50_000_000,)


def test_dftp_command_source_must_equal_active_source():
    assert command_source_matches("policy", "policy")
    assert not command_source_matches("policy", "teleop")
    assert not command_source_matches("", "policy")
    assert not command_source_matches("attacker", "attacker")


def test_realsense_duplicate_timestamp_keeps_fit_window():
    mapper = RealSenseClockMapper(minimum_samples=2, window=10)
    assert mapper.observe(100, 1100, 10_000, "realsense_hardware_clock") is None
    assert mapper.observe(200, 1200, 10_100, "realsense_hardware_clock") == 1200
    before = tuple(mapper._pairs)
    assert mapper.observe(200, 1210, 10_110, "realsense_hardware_clock") == 1200
    assert tuple(mapper._pairs) == before


def test_camera_recording_modes_are_explicit_and_raw_is_guarded():
    common = dict(camera_name="head", serial="s", sequence=1, device_sequence=2,
                  source_time_ns=10, source_clock_domain="realsense_hardware_clock",
                  host_receive_time_ns=20, acquisition_start_ns=11, acquisition_end_ns=19,
                  mapped_host_time_ns=15, width=424, height=240, pixel_format="rgb8",
                  jpeg_quality=90, jpeg=b"jpeg")
    jpeg = CameraFrame(**common).to_record_envelope()
    assert jpeg.topic.endswith("/image_raw/compressed")
    assert jpeg.payload["encoding"] == "jpeg" and "jpeg_b64" in jpeg.payload
    raw = CameraFrame(**common, recording_mode="raw_rgb", raw_rgb=b"raw").to_record_envelope()
    assert raw.topic.endswith("/image_raw")
    assert raw.payload["encoding"] == "raw_rgb" and "raw_rgb_b64" in raw.payload
    assert "jpeg_b64" not in raw.payload
    config = CameraConfig("head", "s", 424, 240, 30, "rgb8", 90, False)
    with pytest.raises(PermissionError):
        RealSenseRgbCapture(config, rs_module=object(), recording_mode="raw_rgb")
    RealSenseRgbCapture(config, rs_module=object(), recording_mode="raw_rgb",
                        raw_rgb_confirmation="CAMERA-RAW-RGB-CALIBRATION")


def test_safe_command_ros_dictionary_flattens_to_canonical_30d():
    point = {
        "left_delta_xyz": [0, 0, 0], "left_delta_rotation6d": IDENTITY,
        "right_delta_xyz": [0, 0, 0], "right_delta_rotation6d": IDENTITY,
        "left_hand_targets": [1] * 6, "right_hand_targets": [2] * 6,
    }
    payload = {
        "schema_version": 1, "frame_id": "world", "valid_mask": 15,
        "deadman": True, "representation": 1,
        "rotation_order": "R00,R10,R20,R01,R11,R21",
        "trajectory": [point],
    }
    assert len(_flatten_safe_command(payload)) == 30
    payload["rotation_order"] = "wrong"
    assert _flatten_safe_command(payload) is None


def test_safe_command_quaternion_converts_and_joint_mode_is_explicitly_invalid():
    point = {
        "left_delta_xyz": [0, 0, 0],
        "right_delta_xyz": [0, 0, 0],
        "left_delta_quaternion_xyzw": [0, 0, 0, 1],
        "right_delta_quaternion_xyzw": [0, 0, 0, -1],
        "left_hand_targets": [1] * 6,
        "right_hand_targets": [2] * 6,
    }
    payload = {
        "schema_version": 1, "frame_id": "world", "valid_mask": 15,
        "deadman": True, "representation": 2,
        "rotation_order": "qx,qy,qz,qw", "trajectory": [point],
    }
    values, reason = _flatten_safe_command_checked(payload)
    assert reason == "" and values[3:9] == IDENTITY and values[12:18] == IDENTITY
    payload["representation"] = 3
    values, reason = _flatten_safe_command_checked(payload)
    assert values is None and "joint-position" in reason
    payload["representation"] = 2
    payload["valid_mask"] = 7
    values, reason = _flatten_safe_command_checked(payload)
    assert values is None and "mask" in reason


def test_ft_zero_jsonl_selects_last_matching_success_and_tool_hash(tmp_path):
    tool = tmp_path / "tool.yaml"
    tool.write_text("tool: test\n")
    digest = canonical_yaml_sha256(tool)
    path = tmp_path / "events.jsonl"
    events = [
        {"event_type": "ft_zero_completed", "session_id": "other", "success": True,
         "tool_payload_config_hash": digest},
        {"event_type": "ft_zero_failed", "session_id": "s", "success": False,
         "tool_payload_config_hash": digest},
        {"event_type": "ft_zero_completed", "session_id": "s", "success": True,
         "tool_payload_config_hash": digest, "event_id": "wanted",
         "connection_generation": 4},
    ]
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n")
    assert _load_ft_zero_event(path, "s", digest)["event_id"] == "wanted"
    events.append({
        "event_type": "ft_zero_invalidated", "session_id": "s",
        "connection_generation": 4, "reason": "rdk_connection_generation_changed",
    })
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n")
    with pytest.raises(ValueError, match="currently-valid"):
        _load_ft_zero_event(path, "s", digest)
    events.append({
        "event_type": "ft_zero_completed", "session_id": "s", "success": True,
        "tool_payload_config_hash": digest, "event_id": "new",
        "connection_generation": 5,
    })
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n")
    assert _load_ft_zero_event(path, "s", digest)["event_id"] == "new"
    events.append({
        "event_type": "ft_zero_completed", "session_id": "s", "success": True,
        "tool_payload_config_hash": "different-tool", "connection_generation": 6,
    })
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n")
    with pytest.raises(ValueError, match="currently-valid"):
        _load_ft_zero_event(path, "s", digest)
    events.append({
        "event_type": "ft_zero_completed", "session_id": "s", "success": True,
        "tool_payload_config_hash": digest, "connection_generation": 7,
    })
    events.append({
        "event_type": "connection_notice", "session_id": "s",
        "connection_generation": 8,
    })
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n")
    with pytest.raises(ValueError, match="currently-valid"):
        _load_ft_zero_event(path, "s", digest)

def test_canonical_yaml_hash_ignores_comments_and_layout(tmp_path):
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("schema_version: 1\narms: {left: {mass: 1}, right: {mass: 2}}\n")
    second.write_text("# operator note\narms:\n  right: {mass: 2}\n  left:\n    mass: 1\nschema_version: 1\n")
    assert canonical_yaml_sha256(first) == canonical_yaml_sha256(second)
    assert sha256_file(first) != sha256_file(second)


def _episode_session(tmp_path, **kwargs):
    tool = tmp_path / "tool.yaml"
    tool.write_text("schema_version: 1\ntool: test\n")
    digest = canonical_yaml_sha256(tool)
    event = tmp_path / "ft.jsonl"
    event.write_text(json.dumps({
        "event_type": "ft_zero_completed", "session_id": "s", "success": True,
        "tool_payload_config_hash": digest, "connection_generation": 1,
    }) + "\n")
    return EpisodeSession(
        tmp_path / "episodes",
        "s",
        {"calibration": tool},
        tool,
        event,
        **kwargs,
    )


def test_invalid_ft_record_leaves_no_orphan_episode_directory(tmp_path):
    tool = tmp_path / "tool.yaml"
    tool.write_text("schema_version: 1\ntool: synthetic\n")
    record = tmp_path / "invalid.jsonl"
    record.write_text(json.dumps({
        "event_type": "ft_zero_completed", "session_id": "wrong",
        "success": True, "tool_payload_config_hash": canonical_yaml_sha256(tool),
    }) + "\n")
    root = tmp_path / "episodes"
    with pytest.raises(ValueError, match="currently-valid"):
        EpisodeSession(root, "s", {"calibration": tool}, tool, record)
    assert not root.exists() or not list(root.iterdir())


def test_rosbag_early_exit_is_reaped_and_not_started(tmp_path, monkeypatch):
    class Process:
        pid = 123
        returncode = 7
        waited = False
        def poll(self): return 7
        def wait(self, *_args, **_kwargs):
            self.waited = True
            return 7
    process = Process()
    monkeypatch.setattr(
        "flexiv_inspire_isaac.data_pipeline.episode_manager.subprocess.Popen",
        lambda *_args, **_kwargs: process,
    )
    monkeypatch.setattr(
        "flexiv_inspire_isaac.data_pipeline.episode_manager.time.sleep",
        lambda *_args: None,
    )
    bag = RosbagProcess(tmp_path / "bag")
    with pytest.raises(RuntimeError, match="early with 7"):
        bag.start()
    assert process.waited and bag.process is None and not bag.started


def test_rosbag_pause_and_resume_use_its_unique_recorder_services(
    tmp_path, monkeypatch
):
    class Process:
        def poll(self):
            return None

    class Future:
        def done(self):
            return True

        def exception(self):
            return None

        def result(self):
            return object()

    class Client:
        def wait_for_service(self, timeout_sec):
            return timeout_sec == 5.0

        def call_async(self, _request):
            return Future()

    class Node:
        def __init__(self):
            self.services = []

        def create_client(self, _service_type, name):
            self.services.append(name)
            return Client()

        def destroy_client(self, _client):
            return None

    monkeypatch.setattr(
        "rclpy.spin_until_future_complete", lambda *_args, **_kwargs: None
    )
    bag = RosbagProcess(tmp_path / "bag")
    bag.process = Process()
    bag._started = True
    node = Node()

    bag.pause(node)
    bag.resume(node)

    assert node.services == [
        f"/{bag.node_name}/pause",
        f"/{bag.node_name}/resume",
    ]


def test_fault_reason_never_marks_episode_complete(tmp_path):
    episode = _episode_session(tmp_path)
    bag = SimpleNamespace(started=True, stop_and_validate=lambda: None)
    with pytest.raises(RuntimeError, match="fault:driver-disconnected"):
        episode.finish(bag, reason="fault:driver-disconnected")
    manifest = json.loads(episode.manifest_path.read_text())
    assert manifest["completed"] is False
    assert manifest["completion_reason"].startswith("fault:")


def test_rerecord_without_sent_command_is_retained_but_not_an_error(tmp_path):
    episode = _episode_session(tmp_path)
    bag = SimpleNamespace(started=True, stop_and_validate=lambda: None)

    episode.finish(bag, reason="rerecord-requested")

    manifest = json.loads(episode.manifest_path.read_text())
    assert manifest["completed"] is False
    assert manifest["completion_reason"] == "rerecord-requested"


def test_training_deviceio_profile_drops_diagnostics_but_keeps_training_data(tmp_path):
    episode = _episode_session(tmp_path)
    episode._deviceio_profile = "training"

    def envelope(topic: str, sequence: int) -> RecordEnvelope:
        return RecordEnvelope(
            topic=topic,
            source_time_ns=sequence,
            host_receive_time_ns=sequence,
            sequence=sequence,
            valid=True,
            mapped_host_time_ns=sequence,
            payload={"value": sequence},
        )

    episode.submit(envelope("/control/requested_command", 1))
    episode.submit(envelope("/control/safe_command", 2))
    episode.submit(envelope("/control/sent_command", 3))
    episode.submit(envelope("/robot/left_arm/state", 4))

    assert "control/requested_command" not in episode.manifest.streams
    assert "control/safe_command" not in episode.manifest.streams
    assert episode.manifest.streams["control/sent_command"].samples == 1
    assert episode.manifest.streams["robot/left_arm/state"].samples == 1
    episode.abort(reason="test-complete")


def test_motion_only_recording_uses_middle_pedal_gate(tmp_path):
    episode = _episode_session(
        tmp_path,
        deviceio_profile="training",
        record_only_while_pedal_pressed=True,
    )

    def envelope(sequence: int) -> RecordEnvelope:
        return RecordEnvelope(
            topic="/robot/left_arm/state",
            source_time_ns=sequence,
            host_receive_time_ns=sequence,
            sequence=sequence,
            valid=True,
            mapped_host_time_ns=sequence,
            payload={"value": sequence},
        )

    episode.submit(envelope(1))
    assert "robot/left_arm/state" not in episode.manifest.streams

    episode.set_motion_recording(True)
    episode.submit(envelope(2))
    assert episode.manifest.streams["robot/left_arm/state"].samples == 1

    episode.set_motion_recording(False)
    episode.submit(envelope(3))
    assert episode.manifest.streams["robot/left_arm/state"].samples == 1
    assert episode.manifest.suppressed_samples == 2
    episode.abort(reason="test-complete")


def test_disabled_ros_mcap_is_absent_from_manifest_and_finish(tmp_path):
    tool = tmp_path / "tool.yaml"
    tool.write_text("schema_version: 1\ntool: test\n")
    digest = canonical_yaml_sha256(tool)
    event = tmp_path / "ft.jsonl"
    event.write_text(json.dumps({
        "event_type": "ft_zero_completed",
        "session_id": "s",
        "success": True,
        "tool_payload_config_hash": digest,
        "connection_generation": 1,
    }) + "\n")
    episode = EpisodeSession(
        tmp_path / "episodes",
        "s",
        {"calibration": tool},
        tool,
        event,
        record_ros_mcap=False,
    )
    bag = RosbagProcess(episode.ros_path, enabled=False)
    bag.start()
    episode.finish(bag, reason="rerecord-requested")

    assert episode.manifest.ros_mcap == ""
    assert not episode.ros_path.exists()


def test_startup_abort_closes_recorder_and_writes_failed_manifest(tmp_path):
    episode = _episode_session(tmp_path)
    episode.abort(reason="fault:rosbag-start")
    manifest = json.loads(episode.manifest_path.read_text())
    assert manifest["completed"] is False
    assert "rosbag-start" in manifest["completion_reason"]
    assert episode.recorder._thread is None


class FailingSink:
    def __init__(self): self.closed = False
    def write(self, _envelope): raise OSError("disk failed")
    def close(self): self.closed = True


def test_critical_write_failure_is_fail_stop_and_retained():
    sink = FailingSink()
    recorder = AsyncMcapRecorder(sink)
    recorder.start()
    recorder.submit(RecordEnvelope("event", 1, 1, 1, True, {"x": 1}), critical=True)
    with pytest.raises(RuntimeError, match="disk failed"):
        recorder.close()
    stats = recorder.stats()
    assert stats["fatal_error"] and stats["queued_critical"] == 1
    assert sink.closed


class BlockingSink:
    def __init__(self):
        self.entered = threading.Event(); self.release = threading.Event(); self.closed = False
    def write(self, _envelope):
        self.entered.set(); self.release.wait(2)
    def close(self): self.closed = True


def test_close_timeout_does_not_close_sink_under_live_writer():
    sink = BlockingSink()
    recorder = AsyncMcapRecorder(sink)
    recorder.start()
    recorder.submit(RecordEnvelope("sensor", 1, 1, 1, True, {"x": 1}))
    assert sink.entered.wait(1)
    with pytest.raises(TimeoutError):
        recorder.close(timeout_s=0.01)
    assert not sink.closed and recorder._thread is not None and recorder._thread.is_alive()
    sink.release.set()
    recorder.close(timeout_s=1)
    assert sink.closed
