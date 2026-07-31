"""Hand-eye calibration with a printed ArUco marker and existing safe teleop.

The collector never publishes a robot command.  Move the arm using the normal
locally-authorized teleoperation path while it observes camera images and the
already world-aligned TCP pose.  This keeps calibration outside the robot
control boundary while still collecting synchronized robot/camera samples.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any

import numpy as np
import yaml


def _matrix(xyz: Any, quaternion_xyzw: Any) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion_xyzw, dtype=float)
    norm = np.linalg.norm([x, y, z, w])
    if not np.isfinite(norm) or norm < 1e-9:
        raise ValueError("invalid quaternion")
    x, y, z, w = np.asarray([x, y, z, w]) / norm
    rotation = np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = rotation, np.asarray(xyz, dtype=float)
    return result


def _quaternion(rotation: np.ndarray) -> list[float]:
    values, vectors = np.linalg.eigh(np.array([
        [rotation[0, 0]-rotation[1, 1]-rotation[2, 2], rotation[0, 1]+rotation[1, 0], rotation[0, 2]+rotation[2, 0], rotation[2, 1]-rotation[1, 2]],
        [rotation[0, 1]+rotation[1, 0], rotation[1, 1]-rotation[0, 0]-rotation[2, 2], rotation[1, 2]+rotation[2, 1], rotation[0, 2]-rotation[2, 0]],
        [rotation[0, 2]+rotation[2, 0], rotation[1, 2]+rotation[2, 1], rotation[2, 2]-rotation[0, 0]-rotation[1, 1], rotation[1, 0]-rotation[0, 1]],
        [rotation[2, 1]-rotation[1, 2], rotation[0, 2]-rotation[2, 0], rotation[1, 0]-rotation[0, 1], np.trace(rotation)],
    ]))
    quat = vectors[:, np.argmax(values)]
    if quat[3] < 0:
        quat = -quat
    return [float(value) for value in quat]


def _pose(transform: np.ndarray) -> dict[str, list[float]]:
    return {"xyz": [float(v) for v in transform[:3, 3]], "quaternion_xyzw": _quaternion(transform[:3, :3])}


def _inverse(transform: np.ndarray) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ transform[:3, 3]
    return result


def _left_quaternion(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = quaternion
    # Convert the standard wxyz multiplication matrix into our xyzw layout.
    permutation = np.eye(4)[[3, 0, 1, 2]]
    standard = np.array([[w, -x, -y, -z], [x, w, -z, y], [y, z, w, -x], [z, -y, x, w]])
    return permutation.T @ standard @ permutation


def _right_quaternion(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = quaternion
    permutation = np.eye(4)[[3, 0, 1, 2]]
    standard = np.array([[w, -x, -y, -z], [x, w, z, -y], [y, -z, w, x], [z, y, -x, w]])
    return permutation.T @ standard @ permutation


def solve_ax_xb(pairs: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Solve rigid transforms ``A X = X B`` from diverse relative motions."""
    if len(pairs) < 3:
        raise ValueError("at least three diverse calibration pairs are required")
    rows = []
    for a, b in pairs:
        rows.append(_left_quaternion(np.asarray(_quaternion(a[:3, :3]))) - _right_quaternion(np.asarray(_quaternion(b[:3, :3]))))
    _, _, vectors = np.linalg.svd(np.concatenate(rows))
    quaternion = vectors[-1]
    quaternion /= np.linalg.norm(quaternion)
    rotation = _matrix([0, 0, 0], quaternion)[:3, :3]
    lhs, rhs = [], []
    for a, b in pairs:
        lhs.append(a[:3, :3] - np.eye(3))
        rhs.append(rotation @ b[:3, 3] - a[:3, 3])
    translation, *_ = np.linalg.lstsq(np.concatenate(lhs), np.concatenate(rhs), rcond=None)
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = rotation, translation
    return result


def solve_samples(mode: str, samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Solve eye-in-hand or eye-to-hand samples captured by this module."""
    if mode not in {"eye_in_hand", "eye_to_hand"}:
        raise ValueError("mode must be eye_in_hand or eye_to_hand")
    if len(samples) < 4:
        raise ValueError("capture at least four poses; 12+ diverse poses is recommended")
    gripper = [_matrix(item["world_T_tcp"]["xyz"], item["world_T_tcp"]["quaternion_xyzw"]) for item in samples]
    camera_target = [_matrix(item["camera_T_target"]["xyz"], item["camera_T_target"]["quaternion_xyzw"]) for item in samples]
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for index in range(len(samples) - 1):
        a = _inverse(gripper[index]) @ gripper[index + 1]
        b = (camera_target[index] @ _inverse(camera_target[index + 1]) if mode == "eye_in_hand"
             else _inverse(camera_target[index]) @ camera_target[index + 1])
        pairs.append((a, b))
    result = {"schema_version": 1, "mode": mode, "samples": len(samples)}
    if mode == "eye_in_hand":
        result["tcp_T_camera"] = _pose(solve_ax_xb(pairs))
    else:
        tcp_target = solve_ax_xb(pairs)
        world_camera = gripper[0] @ tcp_target @ _inverse(camera_target[0])
        result["world_T_camera"] = _pose(world_camera)
        result["tcp_T_calibration_target"] = _pose(tcp_target)
    return result


def _detect_marker(image: np.ndarray, camera_matrix: np.ndarray, distortion: np.ndarray, marker_id: int, marker_size_m: float):
    import cv2
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    corners, ids, _ = cv2.aruco.detectMarkers(image, dictionary)
    if ids is None or marker_id not in ids.reshape(-1):
        return None
    corner = corners[list(ids.reshape(-1)).index(marker_id)][0]
    half = marker_size_m * 0.5
    object_points = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], dtype=np.float32)
    ok, rvec, tvec = cv2.solvePnP(object_points, corner, camera_matrix, distortion)
    if not ok:
        return None
    rotation, _ = cv2.Rodrigues(rvec)
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = rotation, tvec.reshape(3)
    return transform


def capture_samples(args) -> None:
    import cv2
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import CompressedImage

    intrinsics = yaml.safe_load(Path(args.intrinsics).read_text(encoding="utf-8"))
    matrix = np.array([[intrinsics["fx"], 0, intrinsics["ppx"]], [0, intrinsics["fy"], intrinsics["ppy"]], [0, 0, 1]], dtype=float)
    distortion = np.asarray(intrinsics.get("distortion", [0, 0, 0, 0, 0]), dtype=float)

    class Collector(Node):
        def __init__(self):
            super().__init__("camera_handeye_calibration")
            self.latest_pose = None
            self.samples: list[dict[str, Any]] = []
            self.lock = threading.Lock()
            self.create_subscription(PoseStamped, f"/robot/{args.arm}_arm/tcp_pose", self.pose, 10)
            self.create_subscription(CompressedImage, f"/camera/{args.camera}/color/image_raw/compressed", self.image, 10)

        def pose(self, message):
            with self.lock:
                self.latest_pose = _pose(_matrix([message.pose.position.x, message.pose.position.y, message.pose.position.z], [message.pose.orientation.x, message.pose.orientation.y, message.pose.orientation.z, message.pose.orientation.w]))

        def image(self, message):
            with self.lock:
                if self.latest_pose is None or len(self.samples) >= args.samples:
                    return
                image = cv2.imdecode(np.frombuffer(bytes(message.data), np.uint8), cv2.IMREAD_COLOR)
                target = _detect_marker(image, matrix, distortion, args.marker_id, args.marker_size_m)
                if target is None:
                    return
                # Require motion between retained views; otherwise a stationary
                # camera frame burst would make AX=XB rank deficient.
                if self.samples and np.linalg.norm(np.asarray(self.latest_pose["xyz"]) - np.asarray(self.samples[-1]["world_T_tcp"]["xyz"])) < args.min_translation_m:
                    return
                self.samples.append({"world_T_tcp": self.latest_pose, "camera_T_target": _pose(target)})
                self.get_logger().info(f"captured calibration view {len(self.samples)}/{args.samples}")

    rclpy.init()
    node = Collector()
    try:
        while rclpy.ok() and len(node.samples) < args.samples:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        Path(args.output).write_text(yaml.safe_dump({"schema_version": 1, "camera": args.camera, "mode": args.mode, "samples": node.samples}, sort_keys=False), encoding="utf-8")
        node.destroy_node()
        rclpy.shutdown()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Collect/solve safe teleop hand-eye calibration samples")
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture", help="observe printed marker while operator moves arm through normal teleop")
    capture.add_argument("--camera", choices=("head", "left_wrist", "right_wrist"), required=True)
    capture.add_argument("--arm", choices=("left", "right"), required=True)
    capture.add_argument("--mode", choices=("eye_in_hand", "eye_to_hand"), required=True)
    capture.add_argument("--intrinsics", required=True, help="YAML with fx, fy, ppx, ppy from the camera record")
    capture.add_argument("--marker-id", type=int, default=0)
    capture.add_argument("--marker-size-m", type=float, required=True)
    capture.add_argument("--samples", type=int, default=15)
    capture.add_argument("--min-translation-m", type=float, default=0.03)
    capture.add_argument("--output", required=True)
    solve = commands.add_parser("solve", help="solve captured samples and write camera extrinsic")
    solve.add_argument("--samples", required=True)
    solve.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "capture":
        capture_samples(args)
        return 0
    document = yaml.safe_load(Path(args.samples).read_text(encoding="utf-8"))
    Path(args.output).write_text(yaml.safe_dump(solve_samples(document["mode"], document["samples"]), sort_keys=False), encoding="utf-8")
    return 0
