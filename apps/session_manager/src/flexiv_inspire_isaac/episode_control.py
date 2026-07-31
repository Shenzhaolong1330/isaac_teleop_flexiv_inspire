"""Pedal-driven episode process controller.

The controller never moves hardware. It only creates, stops, or invalidates
recording processes after the normal episode manager has verified the current
F/T-zero event and configuration hashes.
"""

from __future__ import annotations

from pathlib import Path
import signal
import subprocess
import sys
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


class EpisodeController(Node):
    def __init__(self) -> None:
        super().__init__("flexiv_inspire_episode_controller")
        defaults = {
            "sessions_root": "sessions",
            "session_id": "",
            "tool_config": "",
            "ft_zero_record": "",
            "camera_config": "",
            "manus_calibration": "",
            "deviceio_socket": "",
            "camera_recording_mode": "jpeg",
        }
        for key, value in defaults.items():
            self.declare_parameter(key, value)
        self._lock = threading.RLock()
        self._process: subprocess.Popen[str] | None = None
        self._sequence = 0
        self._status = self.create_publisher(
            String, "/episode/status", QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        )
        self.create_subscription(
            String, "/episode/control", self._on_control,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE),
        )
        self._publish("STOPPED")

    def _required(self, name: str) -> str:
        value = str(self.get_parameter(name).value).strip()
        if not value:
            raise RuntimeError(f"episode controller parameter {name} is required")
        return value

    def _command(self) -> list[str]:
        self._sequence += 1
        return [
            sys.executable, "-m", "flexiv_inspire_isaac.data_pipeline.episode_manager",
            "--root", self._required("sessions_root"),
            "--session-id", self._required("session_id"),
            "--tool-config", self._required("tool_config"),
            "--ft-zero-record", self._required("ft_zero_record"),
            "--calibration", f"cameras={self._required('camera_config')}",
            "--calibration", f"manus={self._required('manus_calibration')}",
            "--camera-recording-mode", str(self.get_parameter("camera_recording_mode").value),
            "--deviceio-mode", "native",
            "--deviceio-socket", self._required("deviceio_socket"),
        ]

    def _publish(self, value: str) -> None:
        message = String()
        message.data = value
        self._status.publish(message)

    def _start(self) -> None:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                self._publish("RECORDING")
                return
            self._publish("STARTING")
            try:
                self._process = subprocess.Popen(self._command(), text=True)
            except Exception as exc:
                self._process = None
                self._publish(f"FAULT:start:{type(exc).__name__}:{exc}")
                return
            self._publish("RECORDING")

    def _stop(self, *, rerecord: bool) -> None:
        with self._lock:
            process = self._process
            self._process = None
        if process is None or process.poll() is not None:
            self._publish("STOPPED")
            return
        self._publish("RERECORDING" if rerecord else "STOPPING")
        # SIGUSR1 asks the recorder to close atomically and mark this raw
        # capture invalid/re-recorded. SIGINT remains an ordinary stop.
        process.send_signal(signal.SIGUSR1 if rerecord else signal.SIGINT)
        try:
            process.wait(timeout=30.0)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
        self._publish("STOPPED")

    def _on_control(self, message: String) -> None:
        command = str(message.data).strip().lower()
        if command == "toggle":
            active = self._process is not None and self._process.poll() is None
            if active:
                self._stop(rerecord=False)
            else:
                self._start()
        elif command == "rerecord":
            self._stop(rerecord=True)
            self._start()
        else:
            self.get_logger().warning(f"ignored unknown episode control: {command!r}")

    def destroy_node(self) -> bool:
        self._stop(rerecord=False)
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = EpisodeController()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
