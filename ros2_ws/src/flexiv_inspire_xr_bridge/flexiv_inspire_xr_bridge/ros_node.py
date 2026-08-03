"""ROS 2 bridge from recorded CameraFrame JPEGs to Isaac Teleop RTP inputs."""
from __future__ import annotations

import json

from .streamer import EncoderSettings, LatestFrameEncoder

CAMERAS = ("head", "left_wrist", "right_wrist")


def main(args=None) -> int:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from std_msgs.msg import String
    from flexiv_inspire_interfaces.msg import CameraFrame

    class XrVideoBridge(Node):
        def __init__(self) -> None:
            super().__init__("flexiv_inspire_xr_bridge")
            defaults = {
                "enabled": True,
                "ffmpeg": "/usr/bin/ffmpeg",
                "encoder": "auto",
                "receiver_host": "127.0.0.1",
                "bitrate_mbps": 4.0,
                "keyframe_interval_frames": 15,
                "packet_size": 1200,
                "payload_type": 96,
            }
            for key, value in defaults.items():
                self.declare_parameter(key, value)
            default_ports = {"head": 5000, "left_wrist": 5002, "right_wrist": 5003}
            self._workers: dict[str, LatestFrameEncoder] = {}
            self._ready_streams: set[str] = set()
            self._subscriptions = []
            for name in CAMERAS:
                self.declare_parameter(f"streams.{name}.enabled", True)
                self.declare_parameter(f"streams.{name}.topic", f"/camera/{name}/color/frame")
                self.declare_parameter(f"streams.{name}.port", default_ports[name])
                self.declare_parameter(f"streams.{name}.fps", 30.0)
            self._status_publisher = self.create_publisher(String, "/xr_video/status", 10)
            if bool(self.get_parameter("enabled").value):
                self._start(CameraFrame, qos_profile_sensor_data)
            self.create_timer(1.0, self._publish_status)

        def _start(self, message_type, qos) -> None:
            for name in CAMERAS:
                if not bool(self.get_parameter(f"streams.{name}.enabled").value):
                    continue
                settings = EncoderSettings(
                    ffmpeg=str(self.get_parameter("ffmpeg").value),
                    host=str(self.get_parameter("receiver_host").value),
                    port=int(self.get_parameter(f"streams.{name}.port").value),
                    fps=float(self.get_parameter(f"streams.{name}.fps").value),
                    bitrate_mbps=float(self.get_parameter("bitrate_mbps").value),
                    gop=int(self.get_parameter("keyframe_interval_frames").value),
                    packet_size=int(self.get_parameter("packet_size").value),
                    payload_type=int(self.get_parameter("payload_type").value),
                    encoder=str(self.get_parameter("encoder").value),
                )
                worker = LatestFrameEncoder(settings)
                worker.start()
                self._workers[name] = worker
                topic = str(self.get_parameter(f"streams.{name}.topic").value)
                self._subscriptions.append(self.create_subscription(
                    message_type, topic,
                    lambda message, camera=name: self._on_frame(camera, message), qos,
                ))
                self.get_logger().info(f"{name}: {topic} -> rtp://{settings.host}:{settings.port}")

        def _on_frame(self, camera: str, message) -> None:
            if message.camera and message.camera != camera:
                return
            if not message.acquisition.valid:
                return
            self._workers[camera].submit(bytes(message.image.data))

        def _publish_status(self) -> None:
            snapshots = {k: v.snapshot() for k, v in self._workers.items()}
            for name, snapshot in snapshots.items():
                if int(snapshot["sent"]) > 0 and name not in self._ready_streams:
                    self._ready_streams.add(name)
                    self.get_logger().info(
                        f"XR_VIDEO_STREAM_READY: {name} "
                        f"encoder={snapshot['encoder']} sent={snapshot['sent']}"
                    )
            message = String()
            message.data = json.dumps(
                {"enabled": bool(self._workers), "streams": snapshots},
                separators=(",", ":"), sort_keys=True,
            )
            self._status_publisher.publish(message)

        def destroy_node(self):
            for worker in self._workers.values():
                worker.stop()
            return super().destroy_node()

    rclpy.init(args=args)
    node = XrVideoBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
