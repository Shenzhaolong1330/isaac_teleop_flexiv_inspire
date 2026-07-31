"""ROS 2 publisher and optional DeviceIO MCAP tap for three RGB cameras."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from .capture import CameraFrame as CapturedCameraFrame, CameraStatus, TripleRealSenseCapture
from .config import load_camera_configs
from isaac_teleop_core.deviceio import AsyncDeviceIOEmitter, record_envelope


def _assign_time(message: Any, time_ns: int) -> None:
    message.sec = int(time_ns // 1_000_000_000)
    message.nanosec = int(time_ns % 1_000_000_000)


def main(args=None) -> int:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage
    from std_msgs.msg import String
    from flexiv_inspire_interfaces.msg import AcquisitionInfo, CameraFrame as CameraFrameMsg

    class TripleRgbNode(Node):
        def __init__(self) -> None:
            super().__init__("flexiv_inspire_triple_realsense_rgb")
            default_config = str(Path(__file__).with_name("realsense_rgb.yaml"))
            self.declare_parameter("config", default_config)
            self.declare_parameter("device_mcap_path", "")
            self.declare_parameter("recording_mode", "jpeg")
            self.declare_parameter("raw_rgb_confirmation", "")
            config_path = str(self.get_parameter("config").value)
            recording_mode = str(self.get_parameter("recording_mode").value)
            raw_rgb_confirmation = str(self.get_parameter("raw_rgb_confirmation").value)
            self._configs = load_camera_configs(config_path)
            self._frame_queues = {
                name: deque(maxlen=2) for name in self._configs
            }
            self._status_queue: deque[CameraStatus] = deque(maxlen=100)
            self._image_publishers = {}
            self._acquisition_publishers = {}
            self._frame_publishers = {}
            self._status_publishers = {}
            for name in self._configs:
                root = f"/camera/{name}/color"
                self._image_publishers[name] = self.create_publisher(
                    CompressedImage,
                    f"{root}/image_raw/compressed",
                    qos_profile_sensor_data,
                )
                self._frame_publishers[name] = self.create_publisher(
                    CameraFrameMsg, f"{root}/frame", qos_profile_sensor_data
                )
                self._acquisition_publishers[name] = self.create_publisher(
                    AcquisitionInfo,
                    f"{root}/acquisition",
                    qos_profile_sensor_data,
                )
                self._status_publishers[name] = self.create_publisher(
                    String, f"{root}/status", 10
                )

            self._recorder = None
            self._deviceio = AsyncDeviceIOEmitter("camera", sensor_capacity=128)
            mcap_path = str(self.get_parameter("device_mcap_path").value)
            if mcap_path:
                from ..data_pipeline.recorder import (
                    AsyncMcapRecorder,
                    McapJsonSink,
                )

                self._recorder = AsyncMcapRecorder(McapJsonSink(mcap_path))
                self._recorder.start()

            self._capture = TripleRealSenseCapture(
                self._configs,
                on_frame=self._on_frame,
                on_status=self._status_queue.append,
                recording_mode=recording_mode,
                raw_rgb_confirmation=raw_rgb_confirmation,
            )
            self.create_timer(0.002, self._drain)
            self._capture.start()
            self.get_logger().info(
                "three independent RealSense RGB-only pipelines started; "
                f"424x240@30, ROS JPEG90, DeviceIO mode={recording_mode}, depth disabled"
            )

        def destroy_node(self):
            try:
                self._capture.stop()
            finally:
                try:
                    self._deviceio.close()
                finally:
                    if self._recorder is not None:
                        self._recorder.close()
            return super().destroy_node()

        def _on_frame(self, frame: CapturedCameraFrame) -> None:
            native = frame.to_record_envelope()
            try:
                self._deviceio.emit(record_envelope(
                    producer="camera",
                    topic=native.topic,
                    source_time_ns=native.source_time_ns,
                    host_receive_time_ns=native.host_receive_time_ns,
                    sequence=native.sequence,
                    valid=native.valid,
                    invalid_reason=native.invalid_reason,
                    source_clock_domain=native.source_clock_domain,
                    host_clock_domain=native.host_clock_domain,
                    mapped_host_time_ns=native.mapped_host_time_ns,
                    timing_valid=native.effective_mapped_host_time_ns is not None,
                    payload=native.payload,
                ))
            except (BufferError, RuntimeError, ValueError):
                # Capture must continue if the recorder is absent or congested.
                pass
            if self._recorder is not None:
                self._recorder.submit(native)
            self._frame_queues[frame.camera_name].append(frame)

        def _drain(self) -> None:
            while self._status_queue:
                status = self._status_queue.popleft()
                message = String()
                message.data = json.dumps(
                    asdict(status),
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                self._status_publishers[status.camera_name].publish(message)
            for name, queue in self._frame_queues.items():
                if not queue:
                    continue
                frame = queue.pop()
                queue.clear()
                self._publish_frame(name, frame)

        def _publish_frame(self, name: str, frame: CapturedCameraFrame) -> None:
            image = CompressedImage()
            image.header.stamp = self.get_clock().now().to_msg()
            image.header.frame_id = f"{name}_color_optical_frame"
            image.format = f"jpeg; {frame.pixel_format}"
            image.data = frame.jpeg
            self._image_publishers[name].publish(image)

            acquisition = AcquisitionInfo()
            _assign_time(acquisition.source_time, frame.source_time_ns)
            _assign_time(
                acquisition.host_receive_time, frame.host_receive_time_ns
            )
            _assign_time(
                acquisition.acquisition_start, frame.acquisition_start_ns
            )
            _assign_time(
                acquisition.acquisition_end, frame.acquisition_end_ns
            )
            acquisition.source_sequence = frame.device_sequence
            acquisition.source_clock_domain = frame.source_clock_domain
            acquisition.host_clock_domain = "host_monotonic"
            acquisition.valid = frame.valid
            acquisition.invalid_reason = frame.invalid_reason
            if frame.mapped_host_time_ns is None:
                _assign_time(acquisition.mapped_host_time, 0)
                acquisition.timing_valid = False
                timing_reason = "timing:source-clock-unmapped"
                acquisition.invalid_reason = ";".join(
                    filter(None, (acquisition.invalid_reason, timing_reason))
                )
                age_ns = 0
            else:
                _assign_time(
                    acquisition.mapped_host_time,
                    frame.mapped_host_time_ns,
                )
                age_ns = (
                    frame.host_receive_time_ns - frame.mapped_host_time_ns
                )
                acquisition.timing_valid = age_ns >= 0
                if age_ns < 0:
                    timing_reason = "timing:mapped-source-after-receive"
                    acquisition.valid = False
                    acquisition.invalid_reason = ";".join(
                        filter(None, (acquisition.invalid_reason, timing_reason))
                    )
                    age_ns = 0
            acquisition.age.sec = int(age_ns // 1_000_000_000)
            acquisition.age.nanosec = int(age_ns % 1_000_000_000)
            self._acquisition_publishers[name].publish(acquisition)
            combined = CameraFrameMsg()
            combined.header = image.header
            combined.camera = name
            combined.width = frame.width
            combined.height = frame.height
            combined.image = image
            combined.acquisition = acquisition
            self._frame_publishers[name].publish(combined)

    rclpy.init(args=args)
    node = TripleRgbNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
