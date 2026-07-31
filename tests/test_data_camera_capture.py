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


def depth_camera_config() -> CameraConfig:
    return CameraConfig(
        name="head", serial="serial-1", width=424, height=240, fps=30,
        pixel_format="rgb8", jpeg_quality=90, depth_enabled=True,
        depth_width=424, depth_height=240, depth_fps=30,
        pointcloud_enabled=True, pointcloud_stride=2,
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


class FakeIntrinsics:
    fx, fy, ppx, ppy = 200.0, 200.0, 211.5, 119.5


class FakeProfile:
    def as_video_stream_profile(self):
        return self

    def get_intrinsics(self):
        return FakeIntrinsics()


class FakeDepth:
    profile = FakeProfile()

    def get_data(self):
        return np.full((240, 424), 1000, dtype=np.uint16)

    def get_units(self):
        return 0.001


class FakeDepthFrames(FakeFrames):
    def get_depth_frame(self):
        return FakeDepth()


class FakeAlign:
    def process(self, frames):
        return frames


class FakeDepthRs:
    __file__ = __file__
    class stream:
        color = object()

    @staticmethod
    def align(_stream):
        return FakeAlign()


class FakePipeline:
    def wait_for_frames(self, timeout_ms):
        assert timeout_ms == 1000
        return FakeFrames()


class FakeDepthPipeline(FakePipeline):
    def wait_for_frames(self, timeout_ms):
        assert timeout_ms == 1000
        return FakeDepthFrames()


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
    assert base64.b64decode(envelope.payload["jpeg_b64"]) == b"jpeg-payload"


def test_capture_once_records_aligned_z16_and_organized_pointcloud():
    capture = RealSenseRgbCapture(
        depth_camera_config(), rs_module=FakeDepthRs(),
        jpeg_encoder=lambda image, quality, pixel_format: b"jpeg-payload",
        monotonic_ns=iter([10_000, 20_000]).__next__,
        wall_ns=lambda: 1_700_000_000_000_000_000,
    )
    frame = capture.capture_once(FakeDepthPipeline())
    assert len(frame.depth_z16 or b"") == 424 * 240 * 2
    assert (frame.pointcloud_width, frame.pointcloud_height) == (212, 120)
    assert len(frame.pointcloud_xyz_f32 or b"") == 212 * 120 * 3 * 4
    envelope = frame.to_record_envelope()
    assert envelope.payload["depth_frame_id"] == "head_color_optical_frame"
    assert envelope.payload["pointcloud_encoding"] == "xyz_f32_le"


def test_linkage_preflight_rejects_pkg_config_mismatch(monkeypatch):
    monkeypatch.setattr(
        "flexiv_inspire_isaac.cameras.capture._run_read_only",
        lambda command: "2.58.1",
    )
    with pytest.raises(RuntimeError, match="expected 2.57.7"):
        verify_runtime_linkage(FakeRs())
