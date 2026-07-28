from __future__ import annotations

import json
from pathlib import Path
import socket
import stat
import time

from mcap.reader import make_reader

from isaac_teleop_core.deviceio import AsyncDeviceIOEmitter, record_envelope
from flexiv_inspire_isaac.data_pipeline.episode_manager import EpisodeSession
from flexiv_inspire_isaac.data_pipeline.manifest import canonical_yaml_sha256
from flexiv_inspire_isaac.data_pipeline.native_deviceio import NativeDeviceIOIngress


def _wait(predicate, timeout_s: float = 2.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition did not become true")


def _envelope(sequence: int, topic: str = "/robot/left_arm/state"):
    now = time.monotonic_ns()
    return record_envelope(
        producer="test",
        topic=topic,
        source_time_ns=now,
        host_receive_time_ns=now,
        sequence=sequence,
        payload={"value": sequence},
    )


def test_native_ingress_receives_envelope_and_uses_private_socket(tmp_path):
    received = []
    endpoint = tmp_path / "runtime" / "deviceio.sock"
    ingress = NativeDeviceIOIngress(endpoint, received.append)
    ingress.start()
    emitter = AsyncDeviceIOEmitter("test", endpoint)
    emitter.emit(_envelope(1))
    _wait(lambda: len(received) >= 1)
    emitter.close()
    assert stat.S_IMODE(endpoint.stat().st_mode) == 0o600
    assert received[0]["topic"] == "/robot/left_arm/state"
    assert received[0]["timing_valid"] is True
    ingress.close()
    assert not endpoint.exists()


def test_critical_record_is_retained_until_collector_starts(tmp_path):
    endpoint = tmp_path / "late.sock"
    emitter = AsyncDeviceIOEmitter("test", endpoint, reconnect_period_s=0.005)
    emitter.emit(_envelope(7, "/control/sent_command"), critical=True)
    time.sleep(0.03)
    assert emitter.stats().reconnect_failures > 0
    received = []
    ingress = NativeDeviceIOIngress(endpoint, received.append)
    ingress.start()
    _wait(lambda: any(item["sequence"] == 7 for item in received))
    emitter.close()
    ingress.close()


def test_invalid_datagram_is_counted_without_stopping_ingress(tmp_path):
    endpoint = tmp_path / "invalid.sock"
    received = []
    ingress = NativeDeviceIOIngress(endpoint, received.append)
    ingress.start()
    sender = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    sender.sendto(b"{}", str(endpoint))
    sender.sendto(json.dumps(_envelope(2)).encode(), str(endpoint))
    _wait(lambda: ingress.stats().invalid == 1 and len(received) == 1)
    sender.close()
    ingress.close()


def _episode(tmp_path: Path) -> EpisodeSession:
    tool = tmp_path / "tool.yaml"
    tool.write_text("schema_version: 1\ntool: test\n")
    digest = canonical_yaml_sha256(tool)
    event = tmp_path / "ft.jsonl"
    event.write_text(json.dumps({
        "event_type": "ft_zero_completed",
        "session_id": "session",
        "success": True,
        "tool_payload_config_hash": digest,
        "connection_generation": 1,
    }) + "\n")
    return EpisodeSession(
        tmp_path / "episodes",
        "session",
        {"calibration": tool},
        tool,
        event,
        deviceio_capture_layer="native-pre-dds",
    )


def test_native_ingress_writes_actual_deviceio_mcap_and_source_stats(tmp_path):
    episode = _episode(tmp_path)
    endpoint = tmp_path / "episode.sock"
    ingress = NativeDeviceIOIngress(endpoint, episode.submit_native)
    ingress.start()
    emitter = AsyncDeviceIOEmitter("test", endpoint)
    emitter.emit(_envelope(3))
    _wait(lambda: ingress.stats().received >= 1)
    emitter.close()
    ingress.close()
    episode.abort(reason="test-complete")

    topics = []
    with episode.device_path.open("rb") as stream:
        reader = make_reader(stream)
        for _schema, channel, _message in reader.iter_messages():
            topics.append(channel.topic)
    assert "/robot/left_arm/state" in topics
    manifest = json.loads(episode.manifest_path.read_text())
    assert manifest["deviceio_capture_layer"] == "native-pre-dds"
    assert manifest["streams"]["robot/left_arm/state"]["samples"] == 1
