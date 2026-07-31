"""Pedal-driven episode process and guarded Home controller.

The controller never talks to hardware directly. A left-pedal re-record closes
the current capture as ineligible, requests guarded Home from the control
bridge, and starts a fresh capture only after the matching Home completes.
"""

from __future__ import annotations

import json
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import uuid

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
            "home_result_timeout_s": 30.0,
        }
        for key, value in defaults.items():
            self.declare_parameter(key, value)
        self._lock = threading.RLock()
        self._process: subprocess.Popen[str] | None = None
        self._sequence = 0
        self._pending_home_request_id: str | None = None
        self._pending_home_deadline_ns = 0
        self._status = self.create_publisher(
            String, "/episode/status", QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        )
        self._home_request = self.create_publisher(
            String,
            "/control/home_request_context",
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE),
        )
        self.create_subscription(
            String, "/episode/control", self._on_control,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE),
        )
        self.create_subscription(
            String,
            "/control/home_status",
            self._on_home_status,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE),
        )
        self.create_timer(0.25, self._check_home_timeout)
        self._publish("STOPPED")

    def _required(self, name: str) -> str:
        value = str(self.get_parameter(name).value).strip()
        if not value:
            raise RuntimeError(f"episode controller parameter {name} is required")
        return value

    def _command(self) -> list[str]:
        self._sequence += 1
        command = [
            sys.executable, "-m", "flexiv_inspire_isaac.data_pipeline.episode_manager",
            "--root", self._required("sessions_root"),
            "--session-id", self._required("session_id"),
            "--tool-config", self._required("tool_config"),
            "--ft-zero-record", self._required("ft_zero_record"),
            "--calibration", f"cameras={self._required('camera_config')}",
            "--camera-recording-mode", str(self.get_parameter("camera_recording_mode").value),
            "--deviceio-mode", "native",
            "--deviceio-socket", self._required("deviceio_socket"),
        ]
        manus = str(self.get_parameter("manus_calibration").value).strip()
        if manus:
            command.extend(("--calibration", f"manus={manus}"))
        return command

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
        if command in {"stop", "toggle"}:
            # ``toggle`` is accepted as a stop-only compatibility alias. It
            # must never start a capture after the right-pedal semantics
            # changed from toggle to stop.
            with self._lock:
                self._pending_home_request_id = None
                self._pending_home_deadline_ns = 0
            self._stop(rerecord=False)
        elif command == "rerecord":
            with self._lock:
                if self._pending_home_request_id is not None:
                    self._publish("WAITING_FOR_HOME")
                    return
            self._stop(rerecord=True)
            self._request_home_then_start()
        else:
            self.get_logger().warning(f"ignored unknown episode control: {command!r}")

    def _request_home_then_start(self) -> None:
        timeout_s = float(self.get_parameter("home_result_timeout_s").value)
        if not 1.0 <= timeout_s <= 120.0:
            self._publish("FAULT:home:invalid_home_result_timeout")
            return
        request_id = f"rerecord-{uuid.uuid4().hex}"
        with self._lock:
            self._pending_home_request_id = request_id
            self._pending_home_deadline_ns = (
                time.monotonic_ns() + int(timeout_s * 1e9)
            )
        outgoing = String()
        outgoing.data = json.dumps(
            {
                "request_id": request_id,
                "source": "episode_rerecord",
                "session_id": self._required("session_id"),
            },
            separators=(",", ":"),
        )
        self._publish("WAITING_FOR_HOME")
        self._home_request.publish(outgoing)

    def _on_home_status(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
            request_id = str(payload.get("request_id", ""))
            state = str(payload.get("state", "")).strip().lower()
            reason = str(payload.get("reason", "")).strip()
        except Exception as exc:
            self.get_logger().warning(f"ignored malformed Home status: {exc}")
            return
        with self._lock:
            if not request_id or request_id != self._pending_home_request_id:
                return
            if state in {"complete", "failed", "rejected"}:
                self._pending_home_request_id = None
                self._pending_home_deadline_ns = 0
        if state == "complete":
            self._start()
        elif state in {"moving", "accepted"}:
            self._publish("HOMING")
        elif state in {"failed", "rejected"}:
            self._publish(f"FAULT:home:{reason or state}")

    def _check_home_timeout(self) -> None:
        with self._lock:
            if (
                self._pending_home_request_id is None
                or time.monotonic_ns() < self._pending_home_deadline_ns
            ):
                return
            self._pending_home_request_id = None
            self._pending_home_deadline_ns = 0
        self._publish("FAULT:home:result_timeout")

    def destroy_node(self) -> bool:
        with self._lock:
            self._pending_home_request_id = None
            self._pending_home_deadline_ns = 0
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
