"""Small, dependency-light quaternion and rotation helpers.

All public quaternions use ROS/Isaac ordering ``[x, y, z, w]``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def normalize_quaternion_xyzw(value: Any) -> np.ndarray:
    quat = np.asarray(value, dtype=float).reshape(-1)
    if quat.shape != (4,) or not np.all(np.isfinite(quat)):
        raise ValueError("quaternion must contain four finite values")
    norm = float(np.linalg.norm(quat))
    if norm < 1e-9:
        raise ValueError("quaternion norm is too small")
    return quat / norm


def quat_multiply_xyzw(left: Any, right: Any) -> np.ndarray:
    x1, y1, z1, w1 = normalize_quaternion_xyzw(left)
    x2, y2, z2, w2 = normalize_quaternion_xyzw(right)
    result = np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=float,
    )
    return normalize_quaternion_xyzw(result)


def quat_inverse_xyzw(value: Any) -> np.ndarray:
    quat = normalize_quaternion_xyzw(value)
    return np.array([-quat[0], -quat[1], -quat[2], quat[3]], dtype=float)


def quat_to_rotvec(value: Any) -> np.ndarray:
    quat = normalize_quaternion_xyzw(value)
    if quat[3] < 0.0:
        quat = -quat
    vector_norm = float(np.linalg.norm(quat[:3]))
    if vector_norm < 1e-10:
        return 2.0 * quat[:3]
    angle = 2.0 * math.atan2(vector_norm, float(quat[3]))
    return quat[:3] * (angle / vector_norm)


def rotvec_to_quat(value: Any) -> np.ndarray:
    rotvec = np.asarray(value, dtype=float).reshape(-1)
    if rotvec.shape != (3,) or not np.all(np.isfinite(rotvec)):
        raise ValueError("rotation vector must contain three finite values")
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-10:
        vector = 0.5 * rotvec
        return normalize_quaternion_xyzw([vector[0], vector[1], vector[2], 1.0])
    half = 0.5 * angle
    vector = rotvec * (math.sin(half) / angle)
    return normalize_quaternion_xyzw(
        [vector[0], vector[1], vector[2], math.cos(half)]
    )


def quat_to_matrix(value: Any) -> np.ndarray:
    x, y, z, w = normalize_quaternion_xyzw(value)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def matrix_to_quat_xyzw(value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("rotation matrix must be finite and 3x3")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(max(1e-15, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])) * 2.0
            quat = np.array(
                [
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                ]
            )
        elif index == 1:
            scale = math.sqrt(max(1e-15, 1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])) * 2.0
            quat = np.array(
                [
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                ]
            )
        else:
            scale = math.sqrt(max(1e-15, 1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])) * 2.0
            quat = np.array(
                [
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                ]
            )
    return normalize_quaternion_xyzw(quat)


def relative_quaternion_xyzw(current: Any, reference: Any) -> np.ndarray:
    return quat_multiply_xyzw(current, quat_inverse_xyzw(reference))


def conjugate_rotation_by_matrix(quat_xyzw: Any, axis_rotation: Any) -> np.ndarray:
    axis = np.asarray(axis_rotation, dtype=float)
    return matrix_to_quat_xyzw(axis @ quat_to_matrix(quat_xyzw) @ axis.T)


def scaled_quaternion_xyzw(quat_xyzw: Any, gain: float) -> np.ndarray:
    return rotvec_to_quat(float(gain) * quat_to_rotvec(quat_xyzw))


def clip_norm(value: Any, limit: float) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm <= limit or norm < 1e-12:
        return vector.copy()
    return vector * (float(limit) / norm)

