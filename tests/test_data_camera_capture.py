from __future__ import annotations

import base64

import numpy as np
import pytest

from flexiv_inspire_isaac.cameras.capture import (
    RealSenseClockMapper,
    RealSenseRgbCapture,
    verify_runtime_linkage,
)
from flexiv_inspire_isaac.cameras.config import CameraConfig


def camera_config() -> CameraConfig:
    return CameraConfig(
        name="head",
        serial="serial-1",
        width=424,
        height=240,
        fps=30,
        pixel_format="rgb8",
        jpeg_quality=90,
        depth_enabled=False,
    )


def test_hardware_clock_mapping_is_explicitly_invalid_during_warmup():
    mapper = RealSenseClockMapper(minimum_samples=4, window=20)
    mapped = []
    for index in range(8):
        source = index * 33_333_333
        host = 8_000_000_000 + source + (index % 2) * 50_000
        mapped.append(
            mapper.observe(
                source,
                host,
                1_700_000_000_000_000_000,
                "realsense_hardware_clock",
            )
        )
    assert mapped[:3] == [None, None, None]
    assert all(value is not None for value in mapped[3:])
    assert all(
        value <= 8_000_000_000 + index * 33_333_333 + (index % 2) * 50_000
        for index, value in enumerate(mapped)
        if value is not None
    )


def test_hardware_timestamp_reset_restarts_clock_warmup():
    mapper = RealSenseClockMapper(minimum_samples=2)
    assert (
        mapper.observe(100, 10_100, 999, "realsense_hardware_clock")
        is None
    )
    assert (
        mapper.observe(200, 10_200, 999, "realsense_hardware_clock")
        is not None
    )
    assert (
        mapper.observe(50, 20_050, 999, "realsense_hardware_clock")
        is None
    )


class FakeColor:
    def get_data(self):
        return np.zeros((240, 424, 3), dtype=np.uint8)

    def get_timestamp(self):
        return 123.5

    def get_frame_timestamp_domain(self):
        return "timestamp_domain.hardware_clock"

    def get_frame_number(self):
        return 11


class FakeFrames:
    def get_color_frame(self):
        return FakeColor()


class FakePipeline:
    def wait_for_frames(self, timeout_ms):
        assert timeout_ms == 1000
        return FakeFrames()


class FakeRs:
    __file__ = __file__


def test_capture_once_produces_jpeg_record_without_zero_filling():
    monotonic_values = iter([10_000, 20_000])
    capture = RealSenseRgbCapture(
        camera_config(),
        rs_module=FakeRs(),
        jpeg_encoder=lambda image, quality, pixel_format: b"jpeg-payload",
        monotonic_ns=lambda: next(monotonic_values),
        wall_ns=lambda: 1_700_000_000_000_000_000,
    )
    frame = capture.capture_once(FakePipeline())
    assert frame.sequence == 1
    assert frame.device_sequence == 11
    assert frame.source_time_ns == 123_500_000
    assert frame.mapped_host_time_ns is None
    envelope = frame.to_record_envelope()
    assert envelope.valid is True
    assert envelope.source_clock_domain == "realsense_hardware_clock"
    assert (
        base64.b64decode(envelope.payload["jpeg_b64"])
        == b"jpeg-payload"
    )


def test_linkage_preflight_rejects_pkg_config_mismatch(monkeypatch):
    monkeypatch.setattr(
        "flexiv_inspire_isaac.cameras.capture._run_read_only",
        lambda command: "2.58.1",
    )
    with pytest.raises(RuntimeError, match="expected 2.57.7"):
        verify_runtime_linkage(FakeRs())
