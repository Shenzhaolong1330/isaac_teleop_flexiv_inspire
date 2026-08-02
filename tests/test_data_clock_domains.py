import json
from types import SimpleNamespace

from flexiv_inspire_isaac.data_pipeline.recorder import RecordEnvelope
from flexiv_inspire_isaac.data_pipeline.alignment import TimedSample, causal_nearest
from flexiv_inspire_isaac.data_pipeline.mcap_input import (
    _camera_sample_from_ros_message,
)


def test_unmapped_device_clock_has_null_age_and_explicit_invalid_timing():
    envelope = RecordEnvelope(
        topic="camera/head",
        source_time_ns=123,
        host_receive_time_ns=9_000_000,
        sequence=1,
        valid=True,
        payload={"jpeg": "omitted"},
        source_clock_domain="realsense_device",
        host_clock_domain="host_monotonic",
    )
    document = json.loads(envelope.json_bytes())
    assert document["source_time_ns"] == 123
    assert document["mapped_host_time_ns"] is None
    assert document["age_ns"] is None
    assert not document["timing_valid"]
    assert document["timing_invalid_reason"] == "source-clock-unmapped"


def test_explicit_clock_mapping_produces_nonnegative_age():
    envelope = RecordEnvelope(
        topic="camera/head",
        source_time_ns=123,
        host_receive_time_ns=1_050,
        sequence=1,
        valid=True,
        payload={},
        source_clock_domain="realsense_device",
        host_clock_domain="host_monotonic",
        mapped_host_time_ns=1_000,
    )
    document = json.loads(envelope.json_bytes())
    assert document["age_ns"] == 50
    assert document["timing_valid"]


def test_alignment_uses_mapped_host_time_not_raw_cross_domain_time():
    samples = [
        TimedSample(
            "older",
            source_time_ns=9_000_000_000_000,
            host_receive_time_ns=1_010,
            sequence=1,
            source_clock_domain="controller",
            host_clock_domain="host_monotonic",
            mapped_host_time_ns=1_000,
            require_explicit_mapping=True,
        ),
        TimedSample(
            "newer",
            source_time_ns=5,
            host_receive_time_ns=2_010,
            sequence=2,
            source_clock_domain="realsense_hardware",
            host_clock_domain="host_monotonic",
            mapped_host_time_ns=2_000,
            require_explicit_mapping=True,
        ),
    ]
    aligned = causal_nearest(samples, 2_050, 100)
    assert aligned.valid
    assert aligned.value == "newer"
    assert aligned.source_time_ns == 2_000
    assert aligned.age_ns == 50


def test_unmapped_explicit_clock_sample_is_not_aligned():
    sample = TimedSample(
        "never-use-raw-time",
        source_time_ns=123,
        host_receive_time_ns=5_000,
        sequence=1,
        source_clock_domain="device",
        host_clock_domain="host_monotonic",
        mapped_host_time_ns=None,
        require_explicit_mapping=True,
    )
    aligned = causal_nearest([sample], 5_000, 100)
    assert not aligned.valid
    assert aligned.reason == "timing-unmapped"


def test_ros_camera_recovery_preserves_mapped_monotonic_time():
    def stamp(value):
        return SimpleNamespace(sec=value // 1_000_000_000, nanosec=value % 1_000_000_000)

    acquisition = SimpleNamespace(
        source_time=stamp(123),
        host_receive_time=stamp(1_050),
        mapped_host_time=stamp(1_000),
        source_sequence=7,
        valid=True,
        invalid_reason="",
        timing_valid=True,
        source_clock_domain="realsense_hardware_clock",
        host_clock_domain="host_monotonic",
    )
    message = SimpleNamespace(
        camera="head",
        acquisition=acquisition,
        image=SimpleNamespace(data=b"jpeg"),
    )
    name, sample = _camera_sample_from_ros_message(message)
    assert name == "camera/head/jpeg"
    assert sample.value == b"jpeg"
    assert sample.alignment_time_ns == 1_000
    assert sample.valid


def test_mcap_loader_requires_mapping_and_normalizes_camera_topic(tmp_path):
    import pytest
    pytest.importorskip("mcap")
    from flexiv_inspire_isaac.data_pipeline.mcap_input import load_json_mcap_streams
