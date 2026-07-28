"""Independent RGB-only Intel RealSense acquisition.

This module never changes camera firmware or persistent device settings.  Each
camera owns its own pipeline and thread, so a slow or disconnected wrist camera
cannot stall the other two streams.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import base64
import math
from pathlib import Path
import re
import subprocess
import threading
import time
from typing import Callable, Mapping

import numpy as np

from .config import CameraConfig, PINNED_LIBREALSENSE


def _run_read_only(command: list[str]) -> str:
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=5.0,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"{' '.join(command)} failed: {detail}")
    return result.stdout.strip()


def verify_runtime_linkage(rs_module) -> dict[str, object]:
    """Fail closed on a 2.57/2.58 Python/native-library mixture.

    pyrealsense2 wheels do not consistently expose ``__version__``.  Therefore
    the authoritative checks are pkg-config plus the native dependencies of the
    loaded extension.  All operations here are read-only.
    """

    pkg_version = _run_read_only(["pkg-config", "--modversion", "realsense2"])
    if pkg_version != PINNED_LIBREALSENSE:
        raise RuntimeError(
            f"pkg-config reports librealsense {pkg_version}, expected "
            f"{PINNED_LIBREALSENSE}"
        )

    module_file = Path(str(getattr(rs_module, "__file__", ""))).resolve()
    if not module_file.exists():
        raise RuntimeError("cannot locate the loaded pyrealsense2 module")
    if module_file.suffix == ".so":
        extensions = [module_file]
    else:
        extensions = sorted(module_file.parent.glob("pyrealsense2*.so"))
    if not extensions:
        raise RuntimeError("cannot locate the pyrealsense2 extension .so")

    extension = extensions[0].resolve()
    linkage = _run_read_only(["ldd", str(extension)])
    if "2.58" in linkage:
        raise RuntimeError("pyrealsense2 is linked to forbidden librealsense 2.58")

    linked_paths: list[str] = []
    for line in linkage.splitlines():
        if "librealsense2" not in line:
            continue
        match = re.search(r"=>\s+(\S+)", line)
        if match:
            linked_paths.append(str(Path(match.group(1)).resolve()))
        else:
            token = line.strip().split()[0]
            if token.startswith("/"):
                linked_paths.append(str(Path(token).resolve()))
    if not linked_paths:
        raise RuntimeError("ldd did not report the loaded librealsense2 library")
    if any("2.58" in path for path in linked_paths):
        raise RuntimeError("resolved linkage contains forbidden librealsense 2.58")
    if not all(
        ("2.57.7" in path) or re.search(r"\.so\.2\.57(?:$|\.)", path)
        for path in linked_paths
    ):
        raise RuntimeError(
            "loaded librealsense path is not from the pinned 2.57.7 family: "
            + ", ".join(linked_paths)
        )

    reported = getattr(rs_module, "__version__", None)
    if reported is not None and not str(reported).startswith(PINNED_LIBREALSENSE):
        raise RuntimeError(
            f"pyrealsense2 reports {reported}, expected {PINNED_LIBREALSENSE}"
        )
    return {
        "pkg_config_version": pkg_version,
        "python_module": str(module_file),
        "extension": str(extension),
        "linked_libraries": linked_paths,
        "reported_python_version": (
            str(reported) if reported is not None else None
        ),
        "mixed_2_58": False,
    }


def _domain_name(domain: object) -> str:
    value = str(domain).lower()
    if "hardware" in value:
        return "realsense_hardware_clock"
    if "global" in value:
        return "realsense_global_time"
    if "system" in value:
        return "realsense_system_time"
    return f"realsense_unknown:{value}"


class RealSenseClockMapper:
    """Map a device timestamp to host monotonic time without hiding uncertainty.

    For hardware-clock timestamps, an affine slope is estimated and the lower
    envelope of receive times supplies the offset.  Mapping stays invalid until
    enough observations exist.  A timestamp reset clears the estimator.
    """

    def __init__(self, *, minimum_samples: int = 8, window: int = 300) -> None:
        if minimum_samples < 2 or window < minimum_samples:
            raise ValueError("invalid clock-mapper sample limits")
        self.minimum_samples = minimum_samples
        self._pairs: deque[tuple[int, int]] = deque(maxlen=window)
        self._last_source_ns: int | None = None

    def reset(self) -> None:
        self._pairs.clear()
        self._last_source_ns = None

    def observe(
        self,
        source_ns: int,
        host_monotonic_ns: int,
        host_wall_ns: int,
        source_clock_domain: str,
    ) -> int | None:
        if source_ns < 0:
            return None
        if source_clock_domain in {
            "realsense_global_time",
            "realsense_system_time",
        }:
            mapped = source_ns + (host_monotonic_ns - host_wall_ns)
            return mapped if mapped <= host_monotonic_ns else None
        if source_clock_domain != "realsense_hardware_clock":
            return None
        if self._last_source_ns is not None and source_ns < self._last_source_ns:
            self.reset()
        duplicate = self._last_source_ns is not None and source_ns == self._last_source_ns
        if not duplicate:
            self._last_source_ns = source_ns
            self._pairs.append((source_ns, host_monotonic_ns))
        if len(self._pairs) < self.minimum_samples:
            return None

        x0, y0 = self._pairs[0]
        xs = [float(x - x0) for x, _ in self._pairs]
        ys = [float(y - y0) for _, y in self._pairs]
        x_mean = sum(xs) / len(xs)
        y_mean = sum(ys) / len(ys)
        denominator = sum((x - x_mean) ** 2 for x in xs)
        if denominator <= 0.0:
            return None
        slope = sum(
            (x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)
        ) / denominator
        if not math.isfinite(slope) or not 0.995 <= slope <= 1.005:
            return None
        offsets = [
            host - slope * source for source, host in self._pairs
        ]
        offset = min(offsets)
        mapped = int(round(slope * source_ns + offset))
        return mapped if mapped <= host_monotonic_ns else None


@dataclass(frozen=True)
class CameraFrame:
    camera_name: str
    serial: str
    sequence: int
    device_sequence: int
    source_time_ns: int
    source_clock_domain: str
    host_receive_time_ns: int
    acquisition_start_ns: int
    acquisition_end_ns: int
    mapped_host_time_ns: int | None
    width: int
    height: int
    pixel_format: str
    jpeg_quality: int
    jpeg: bytes
    recording_mode: str = "jpeg"
    raw_rgb: bytes | None = None
    valid: bool = True
    invalid_reason: str = ""

    def to_record_envelope(self):
        from ..data_pipeline.recorder import RecordEnvelope

        raw_mode = self.recording_mode == "raw_rgb"
        if raw_mode and self.raw_rgb is None:
            raise ValueError("raw_rgb recording mode requires raw frame bytes")
        topic = f"/camera/{self.camera_name}/color/image_raw" if raw_mode else f"/camera/{self.camera_name}/color/image_raw/compressed"
        payload = {
            "camera_name": self.camera_name,
            "serial": self.serial,
            "device_sequence": self.device_sequence,
            "acquisition_start_ns": self.acquisition_start_ns,
            "acquisition_end_ns": self.acquisition_end_ns,
            "width": self.width,
            "height": self.height,
            "pixel_format": self.pixel_format,
            "encoding": self.recording_mode,
            "jpeg_quality": self.jpeg_quality,
        }
        if raw_mode:
            payload["raw_rgb_b64"] = base64.b64encode(self.raw_rgb).decode("ascii")
        else:
            payload["jpeg_b64"] = base64.b64encode(self.jpeg).decode("ascii")
        return RecordEnvelope(
            topic=topic,
            source_time_ns=self.source_time_ns,
            host_receive_time_ns=self.host_receive_time_ns,
            sequence=self.sequence,
            valid=self.valid,
            invalid_reason=self.invalid_reason,
            source_clock_domain=self.source_clock_domain,
            host_clock_domain="host_monotonic",
            mapped_host_time_ns=self.mapped_host_time_ns,
            payload=payload,
        )


@dataclass(frozen=True)
class CameraStatus:
    camera_name: str
    serial: str
    connected: bool
    fault: bool
    reason: str
    host_monotonic_ns: int
    captured_frames: int
    device_dropped_frames: int
    reconnects: int
    encode_failures: int


def encode_jpeg(
    image: np.ndarray, quality: int, pixel_format: str
) -> bytes:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for JPEG encoding") from exc
    encoded_image = image
    if pixel_format.lower() == "rgb8":
        encoded_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, payload = cv2.imencode(
        ".jpg",
        encoded_image,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    if not ok:
        raise RuntimeError("OpenCV failed to encode a RealSense frame")
    return bytes(payload)


class RealSenseRgbCapture:
    """One serial-number-bound RGB pipeline and acquisition thread."""

    def __init__(
        self,
        config: CameraConfig,
        *,
        on_frame: Callable[[CameraFrame], None] | None = None,
        on_status: Callable[[CameraStatus], None] | None = None,
        rs_module=None,
        jpeg_encoder: Callable[[np.ndarray, int, str], bytes] = encode_jpeg,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        wall_ns: Callable[[], int] = time.time_ns,
        reconnect_delay_s: float = 0.5,
        recording_mode: str = "jpeg",
        raw_rgb_confirmation: str = "",
    ) -> None:
        self.config = config
        self.on_frame = on_frame or (lambda _frame: None)
        self.on_status = on_status or (lambda _status: None)
        if rs_module is None:
            try:
                import pyrealsense2 as rs_module
            except ImportError as exc:
                raise RuntimeError(
                    "pyrealsense2 compatible with librealsense 2.57.7 is required"
                ) from exc
        self.rs = rs_module
        self.jpeg_encoder = jpeg_encoder
        self.monotonic_ns = monotonic_ns
        self.wall_ns = wall_ns
        self.reconnect_delay_s = reconnect_delay_s
        if recording_mode not in {"jpeg", "raw_rgb"}:
            raise ValueError("recording_mode must be jpeg or raw_rgb")
        if recording_mode == "raw_rgb" and raw_rgb_confirmation != "CAMERA-RAW-RGB-CALIBRATION":
            raise PermissionError("raw_rgb mode requires local calibration confirmation")
        self.recording_mode = recording_mode
        self.clock_mapper = RealSenseClockMapper()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sequence = 0
        self._last_device_sequence: int | None = None
        self._captured_frames = 0
        self._device_dropped_frames = 0
        self._reconnects = 0
        self._encode_failures = 0
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError(f"{self.config.name} camera already started")
        verify_runtime_linkage(self.rs)
        self._stop.clear()
        self._thread = threading.Thread(
            target=self.run,
            name=f"realsense-{self.config.name}",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout_s: float = 3.0) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout_s)
            if thread.is_alive():
                raise TimeoutError(
                    f"{self.config.name} RealSense worker did not stop"
                )

    def snapshot_status(
        self, *, connected: bool, fault: bool, reason: str
    ) -> CameraStatus:
        with self._lock:
            return CameraStatus(
                camera_name=self.config.name,
                serial=self.config.serial,
                connected=connected,
                fault=fault,
                reason=reason,
                host_monotonic_ns=self.monotonic_ns(),
                captured_frames=self._captured_frames,
                device_dropped_frames=self._device_dropped_frames,
                reconnects=self._reconnects,
                encode_failures=self._encode_failures,
            )

    def _emit_status(
        self, *, connected: bool, fault: bool, reason: str
    ) -> None:
        self.on_status(
            self.snapshot_status(
                connected=connected, fault=fault, reason=reason
            )
        )

    def _new_pipeline(self):
        pipeline = self.rs.pipeline()
        rs_config = self.rs.config()
        rs_config.enable_device(self.config.serial)
        rs_format = getattr(
            self.rs.format,
            "rgb8" if self.config.pixel_format.lower() == "rgb8" else "bgr8",
        )
        rs_config.enable_stream(
            self.rs.stream.color,
            self.config.width,
            self.config.height,
            rs_format,
            self.config.fps,
        )
        return pipeline, rs_config

    def run(self) -> None:
        first_connection = True
        while not self._stop.is_set():
            pipeline = None
            try:
                self._emit_status(
                    connected=False, fault=False, reason="connecting"
                )
                pipeline, rs_config = self._new_pipeline()
                pipeline.start(rs_config)
                if not first_connection:
                    with self._lock:
                        self._reconnects += 1
                first_connection = False
                self.clock_mapper.reset()
                self._last_device_sequence = None
                self._emit_status(
                    connected=True, fault=False, reason="streaming"
                )
                while not self._stop.is_set():
                    self.on_frame(self.capture_once(pipeline))
            except Exception as exc:
                self._emit_status(
                    connected=False,
                    fault=True,
                    reason=f"capture-failed:{type(exc).__name__}:{exc}",
                )
            finally:
                if pipeline is not None:
                    try:
                        pipeline.stop()
                    except Exception:
                        pass
            if not self._stop.is_set():
                self._stop.wait(self.reconnect_delay_s)
        self._emit_status(connected=False, fault=False, reason="stopped")

    def capture_once(self, pipeline) -> CameraFrame:
        acquisition_start = self.monotonic_ns()
        frames = pipeline.wait_for_frames(timeout_ms=1000)
        color = frames.get_color_frame()
        if not color:
            raise RuntimeError("frameset did not contain a color frame")
        image = np.asanyarray(color.get_data())
        acquisition_end = self.monotonic_ns()
        host_wall = self.wall_ns()
        if image.shape[:2] != (self.config.height, self.config.width):
            raise RuntimeError(
                f"unexpected RGB shape {image.shape}, expected "
                f"({self.config.height}, {self.config.width}, 3)"
            )
        try:
            jpeg = self.jpeg_encoder(
                image, self.config.jpeg_quality, self.config.pixel_format
            )
        except Exception:
            with self._lock:
                self._encode_failures += 1
            raise
        source_time_ns = int(round(float(color.get_timestamp()) * 1_000_000.0))
        source_domain = _domain_name(color.get_frame_timestamp_domain())
        mapped = self.clock_mapper.observe(
            source_time_ns,
            acquisition_end,
            host_wall,
            source_domain,
        )
        device_sequence = int(color.get_frame_number())
        with self._lock:
            if (
                self._last_device_sequence is not None
                and device_sequence > self._last_device_sequence + 1
            ):
                self._device_dropped_frames += (
                    device_sequence - self._last_device_sequence - 1
                )
            self._last_device_sequence = device_sequence
            self._sequence += 1
            self._captured_frames += 1
            sequence = self._sequence
        return CameraFrame(
            camera_name=self.config.name,
            serial=self.config.serial,
            sequence=sequence,
            device_sequence=device_sequence,
            source_time_ns=source_time_ns,
            source_clock_domain=source_domain,
            host_receive_time_ns=acquisition_end,
            acquisition_start_ns=acquisition_start,
            acquisition_end_ns=acquisition_end,
            mapped_host_time_ns=mapped,
            width=self.config.width,
            height=self.config.height,
            pixel_format=self.config.pixel_format,
            jpeg_quality=self.config.jpeg_quality,
            jpeg=jpeg,
            recording_mode=self.recording_mode,
            raw_rgb=image.tobytes(order="C") if self.recording_mode == "raw_rgb" else None,
        )


class TripleRealSenseCapture:
    """Lifecycle wrapper for exactly head, left-wrist and right-wrist cameras."""

    def __init__(
        self,
        configs: Mapping[str, CameraConfig],
        **capture_kwargs,
    ) -> None:
        if set(configs) != {"head", "left_wrist", "right_wrist"}:
            raise ValueError("three configured cameras are required")
        self.captures = {
            name: RealSenseRgbCapture(config, **capture_kwargs)
            for name, config in configs.items()
        }

    def start(self) -> None:
        started: list[RealSenseRgbCapture] = []
        try:
            for capture in self.captures.values():
                capture.start()
                started.append(capture)
        except Exception:
            for capture in reversed(started):
                capture.stop()
            raise

    def stop(self) -> None:
        errors: list[str] = []
        for capture in self.captures.values():
            try:
                capture.stop()
            except Exception as exc:
                errors.append(str(exc))
        if errors:
            raise RuntimeError("; ".join(errors))

    def status_documents(self) -> dict[str, dict[str, object]]:
        return {
            name: asdict(
                capture.snapshot_status(
                    connected=False,
                    fault=False,
                    reason="snapshot-does-not-imply-connectivity",
                )
            )
            for name, capture in self.captures.items()
        }
