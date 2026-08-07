"""Publish OculusReader Quest controllers using the existing ROS contract.

The Quest APK computes ``head^-1 * controller`` before writing a frame to
logcat, matching the head-relative pose semantics of ``xr_raw_ros_source``.
MANUS remains an independent Ergonomics input and is intentionally absent
from this node.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
import math
from pathlib import Path
import shutil
import subprocess
import threading
import time
from typing import Any, Callable

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseArray, TransformStamped
from rclpy.node import Node
from std_msgs.msg import ByteMultiArray
from tf2_ros import TransformBroadcaster

from .quest_input_contract import controller_payload, encode_controller_payload


_LOG_TAG = "wE9ryARX"


@dataclass(frozen=True)
class OculusFrame:
    left: np.ndarray
    right: np.ndarray
    buttons: dict[str, Any]
    unix_ns: int
    monotonic_ns: int
    sequence: int


def _adb_devices(output: str) -> list[str]:
    devices: list[str] = []
    for raw in output.splitlines()[1:]:
        fields = raw.strip().split()
        if len(fields) >= 2 and fields[1] == "device":
            devices.append(fields[0])
    return devices


def _select_adb_device(output: str, configured: str = "") -> str:
    devices = _adb_devices(output)
    requested = configured.strip()
    if requested:
        if requested not in devices:
            raise RuntimeError(
                f"configured Quest adb_serial {requested!r} is not connected; "
                f"available={devices}"
            )
        return requested
    if len(devices) != 1:
        raise RuntimeError(
            "OculusReader requires exactly one authorized ADB device when "
            f"teleop.quest_input.oculus_reader.adb_serial is empty; found {devices}"
        )
    return devices[0]


def _extract_oculus_payload(line: str) -> str:
    marker = f"{_LOG_TAG}: "
    return line.split(marker, 1)[1].strip() if marker in line else ""


def _button_scalar(buttons: dict[str, Any], key: str) -> float:
    raw = buttons.get(key, 0.0)
    if isinstance(raw, (tuple, list)):
        raw = raw[0] if raw else 0.0
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"OculusReader button {key} is not finite")
    return min(1.0, max(0.0, value))


def _matrix_pose(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    transform = np.asarray(matrix, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("OculusReader transform must be a finite 4x4 matrix")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-4):
        raise ValueError("OculusReader transform has an invalid homogeneous row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2.0e-3):
        raise ValueError("OculusReader transform rotation is not orthonormal")
    if np.linalg.det(rotation) < 0.99:
        raise ValueError("OculusReader transform rotation is not right-handed")

    # Stable rotation-matrix to quaternion conversion, returned as ROS xyzw.
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (rotation[2, 1] - rotation[1, 2]) / scale
        qy = (rotation[0, 2] - rotation[2, 0]) / scale
        qz = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        diagonal = int(np.argmax(np.diag(rotation)))
        if diagonal == 0:
            scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            qw = (rotation[2, 1] - rotation[1, 2]) / scale
            qx = 0.25 * scale
            qy = (rotation[0, 1] + rotation[1, 0]) / scale
            qz = (rotation[0, 2] + rotation[2, 0]) / scale
        elif diagonal == 1:
            scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            qw = (rotation[0, 2] - rotation[2, 0]) / scale
            qx = (rotation[0, 1] + rotation[1, 0]) / scale
            qy = 0.25 * scale
            qz = (rotation[1, 2] + rotation[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            qw = (rotation[1, 0] - rotation[0, 1]) / scale
            qx = (rotation[0, 2] + rotation[2, 0]) / scale
            qy = (rotation[1, 2] + rotation[2, 1]) / scale
            qz = 0.25 * scale
    quaternion = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    return transform[:3, 3].copy(), quaternion


def _pose(matrix: np.ndarray) -> Pose:
    position, quaternion = _matrix_pose(matrix)
    result = Pose()
    result.position.x, result.position.y, result.position.z = position.tolist()
    (
        result.orientation.x,
        result.orientation.y,
        result.orientation.z,
        result.orientation.w,
    ) = quaternion.tolist()
    return result


def _load_payload_parser() -> Callable[[str], tuple[Any, Any]]:
    try:
        from oculus_reader.reader import OculusReader
    except ImportError as exc:
        raise RuntimeError(
            "oculus_reader is not installed in ros-py312; run "
            "scripts/env/create_envs.sh or sync requirements/ros-py312.lock"
        ) from exc
    return OculusReader.process_data


class OculusReaderRosSource(Node):
    def __init__(self) -> None:
        super().__init__("oculus_reader_ros_source")
        defaults = {
            "rate_hz": 60.0,
            "stale_timeout_s": 0.12,
            "adb_serial": "",
            "package_name": "com.rail.oculus.teleop",
            "auto_install_apk": True,
            "world_frame": "world",
            "left_wrist_frame": "left_wrist",
            "right_wrist_frame": "right_wrist",
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self._rate_hz = float(self.get_parameter("rate_hz").value)
        self._stale_ns = int(
            float(self.get_parameter("stale_timeout_s").value) * 1.0e9
        )
        if not 1.0 <= self._rate_hz <= 240.0 or self._stale_ns <= 0:
            raise ValueError("invalid OculusReader rate or stale timeout")
        self._world = str(self.get_parameter("world_frame").value).strip()
        self._left_wrist = str(self.get_parameter("left_wrist_frame").value).strip()
        self._right_wrist = str(self.get_parameter("right_wrist_frame").value).strip()
        if len({self._world, self._left_wrist, self._right_wrist}) != 3:
            raise ValueError("world and wrist frame names must be non-empty/distinct")

        self._parser = _load_payload_parser()
        self._lock = threading.Lock()
        self._latest: OculusFrame | None = None
        self._published_sequence = 0
        self._received_sequence = 0
        self._reader_failure = ""
        self._stop = threading.Event()
        self._ready_reported = False
        self._logcat: subprocess.Popen[str] | None = None

        self._ee_pub = self.create_publisher(PoseArray, "xr_teleop/ee_poses", 10)
        self._controller_pub = self.create_publisher(
            ByteMultiArray, "xr_teleop/controller_data", 10
        )
        self._tf = TransformBroadcaster(self)

        self._serial = self._connect_quest()
        self._start_logcat()
        self.create_timer(1.0 / self._rate_hz, self._publish_latest)
        self.get_logger().info(
            f"OculusReader Quest input starting via USB ADB device {self._serial}"
        )

    @staticmethod
    def _run(command: list[str], *, timeout: float = 10.0) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

    def _connect_quest(self) -> str:
        adb = shutil.which("adb")
        if adb is None:
            raise RuntimeError("OculusReader requires adb")
        devices = self._run([adb, "devices", "-l"])
        if devices.returncode != 0:
            raise RuntimeError(f"adb devices failed: {devices.stderr.strip()}")
        serial = _select_adb_device(
            devices.stdout, str(self.get_parameter("adb_serial").value)
        )
        package = str(self.get_parameter("package_name").value).strip()
        installed = self._run([adb, "-s", serial, "shell", "pm", "path", package])
        if installed.returncode != 0 or not installed.stdout.strip().startswith("package:"):
            if not bool(self.get_parameter("auto_install_apk").value):
                raise RuntimeError(
                    f"Quest package {package} is not installed and auto_install_apk=false"
                )
            apk = Path(resources.files("oculus_reader").joinpath("APK/teleop-debug.apk"))
            if not apk.is_file() or apk.stat().st_size < 1_000_000:
                raise RuntimeError(f"OculusReader APK is missing or is a Git LFS pointer: {apk}")
            result = self._run([adb, "-s", serial, "install", "-t", str(apk)], timeout=90.0)
            if result.returncode != 0:
                raise RuntimeError(f"OculusReader APK install failed: {result.stdout} {result.stderr}")
        activity = f"{package}/{package}.MainActivity"
        started = self._run(
            [
                adb,
                "-s",
                serial,
                "shell",
                "am",
                "start",
                "-n",
                activity,
                "-a",
                "android.intent.action.MAIN",
                "-c",
                "android.intent.category.LAUNCHER",
            ]
        )
        if started.returncode != 0:
            raise RuntimeError(f"OculusReader Quest app failed to start: {started.stderr}")
        return serial

    def _start_logcat(self) -> None:
        adb = shutil.which("adb")
        assert adb is not None
        self._logcat = subprocess.Popen(
            [adb, "-s", self._serial, "logcat", "-T", "0"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        threading.Thread(
            target=self._read_logcat, name="oculus-reader-logcat", daemon=True
        ).start()

    def _read_logcat(self) -> None:
        assert self._logcat is not None and self._logcat.stdout is not None
        try:
            for line in self._logcat.stdout:
                if self._stop.is_set():
                    break
                payload = _extract_oculus_payload(line)
                if not payload:
                    continue
                transforms, buttons = self._parser(payload)
                if not isinstance(transforms, dict) or not isinstance(buttons, dict):
                    continue
                # The upstream APK swaps the tracking condition used when only
                # one controller is present.  Requiring both also matches the
                # bimanual teleop contract and avoids publishing an uninitialized
                # single-hand matrix.
                if "l" not in transforms or "r" not in transforms:
                    continue
                now_unix = time.time_ns()
                now_monotonic = time.monotonic_ns()
                with self._lock:
                    self._received_sequence += 1
                    self._latest = OculusFrame(
                        left=np.asarray(transforms["l"], dtype=np.float64).copy(),
                        right=np.asarray(transforms["r"], dtype=np.float64).copy(),
                        buttons=dict(buttons),
                        unix_ns=now_unix,
                        monotonic_ns=now_monotonic,
                        sequence=self._received_sequence,
                    )
        except Exception as exc:  # pragma: no cover - hardware stream failure
            self._reader_failure = str(exc)
        finally:
            if not self._stop.is_set() and not self._reader_failure:
                status = self._logcat.poll()
                self._reader_failure = f"adb logcat exited with status {status}"

    def _publish_latest(self) -> None:
        if self._reader_failure:
            raise RuntimeError(f"OculusReader input failed: {self._reader_failure}")
        with self._lock:
            frame = self._latest
        if frame is None or frame.sequence == self._published_sequence:
            return
        if time.monotonic_ns() - frame.monotonic_ns > self._stale_ns:
            return
        left_pose = _pose(frame.left)
        right_pose = _pose(frame.right)
        message = PoseArray()
        message.header.stamp.sec = frame.unix_ns // 1_000_000_000
        message.header.stamp.nanosec = frame.unix_ns % 1_000_000_000
        message.header.frame_id = self._world
        message.poses = [left_pose, right_pose]
        self._ee_pub.publish(message)

        transforms = []
        for pose, child in (
            (left_pose, self._left_wrist),
            (right_pose, self._right_wrist),
        ):
            transform = TransformStamped()
            transform.header = message.header
            transform.child_frame_id = child
            transform.transform.translation.x = pose.position.x
            transform.transform.translation.y = pose.position.y
            transform.transform.translation.z = pose.position.z
            transform.transform.rotation = pose.orientation
            transforms.append(transform)
        self._tf.sendTransform(transforms)

        left_active = "X" in frame.buttons
        right_active = "A" in frame.buttons
        payload = controller_payload(
            timestamp_ns=frame.unix_ns,
            left_squeeze_value=_button_scalar(frame.buttons, "leftGrip"),
            right_squeeze_value=_button_scalar(frame.buttons, "rightGrip"),
            left_primary_click=bool(frame.buttons.get("X", False)),
            right_primary_click=bool(frame.buttons.get("A", False)),
            left_is_active=left_active,
            right_is_active=right_active,
        )
        controller_message = ByteMultiArray()
        controller_message.data = encode_controller_payload(payload)
        self._controller_pub.publish(controller_message)
        self._published_sequence = frame.sequence
        if not self._ready_reported:
            self._ready_reported = True
            self.get_logger().info(
                "QUEST_INPUT_READY: OculusReader published valid left/right wrist poses"
            )

    def destroy_node(self) -> bool:
        self._stop.set()
        if self._logcat is not None and self._logcat.poll() is None:
            self._logcat.terminate()
            try:
                self._logcat.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._logcat.kill()
                self._logcat.wait(timeout=1.0)
        return super().destroy_node()


def main(args=None) -> int:
    rclpy.init(args=args)
    node: OculusReaderRosSource | None = None
    try:
        node = OculusReaderRosSource()
        rclpy.spin(node)
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
