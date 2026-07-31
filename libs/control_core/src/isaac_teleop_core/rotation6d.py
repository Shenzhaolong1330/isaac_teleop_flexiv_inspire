"""Strict conversion utilities for continuous Rotation-6D.

The convention is Zhou et al.'s first two matrix columns:

``[R00, R10, R20, R01, R11, R21]``.

No function silently repairs invalid external input. Matrix validation is strict;
Gram--Schmidt is used only where it is part of the Rotation-6D decoding
definition.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

IDENTITY_ROT6D = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)
_VECTOR_EPS = 1.0e-9
_COLLINEAR_SIN_MIN = 1.0e-5
_ORTHO_ATOL = 1.0e-7
_DET_ATOL = 1.0e-7


class RotationError(ValueError):
    """An invalid or degenerate rotation representation."""


def _finite_vector(value: Any, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (size,):
        raise RotationError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise RotationError(f"{name} contains NaN or Inf")
    return array


def validate_rotation_matrix(
    value: Any,
    *,
    orthogonality_atol: float = _ORTHO_ATOL,
    determinant_atol: float = _DET_ATOL,
) -> np.ndarray:
    """Return a copy of a finite, right-handed SO(3) matrix or raise."""

    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise RotationError(f"rotation matrix must have shape (3, 3), got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise RotationError("rotation matrix contains NaN or Inf")
    gram = matrix.T @ matrix
    if not np.allclose(gram, np.eye(3), atol=orthogonality_atol, rtol=0.0):
        raise RotationError("rotation matrix is not orthonormal")
    determinant = float(np.linalg.det(matrix))
    if not math.isfinite(determinant) or abs(determinant - 1.0) > determinant_atol:
        raise RotationError(f"rotation matrix determinant is {determinant}, expected +1")
    return matrix.copy()


def rotation6d_to_matrix(
    value: Any,
    *,
    vector_eps: float = _VECTOR_EPS,
    collinear_sin_min: float = _COLLINEAR_SIN_MIN,
) -> np.ndarray:
    """Decode first-two-columns Rotation-6D with strict degeneracy checks."""

    rot6d = _finite_vector(value, 6, "rotation6d")
    a1 = rot6d[:3]
    a2 = rot6d[3:]
    n1 = float(np.linalg.norm(a1))
    n2 = float(np.linalg.norm(a2))
    if n1 <= vector_eps:
        raise RotationError("rotation6d first column norm is too small")
    if n2 <= vector_eps:
        raise RotationError("rotation6d second column norm is too small")

    b1 = a1 / n1
    residual = a2 - float(np.dot(b1, a2)) * b1
    residual_norm = float(np.linalg.norm(residual))
    # This normalized measure is invariant to the magnitude of a2.
    if residual_norm / n2 < collinear_sin_min:
        raise RotationError("rotation6d columns are collinear or nearly collinear")
    b2 = residual / residual_norm
    b3 = np.cross(b1, b2)
    b3_norm = float(np.linalg.norm(b3))
    if b3_norm <= vector_eps:
        raise RotationError("rotation6d orthogonalization failed")
    b3 /= b3_norm
    matrix = np.column_stack((b1, b2, b3))
    return validate_rotation_matrix(matrix, orthogonality_atol=5e-7, determinant_atol=5e-7)


def matrix_to_rotation6d(value: Any) -> np.ndarray:
    """Encode an SO(3) matrix as its first two columns."""

    matrix = validate_rotation_matrix(value)
    return np.concatenate((matrix[:, 0], matrix[:, 1]))


def normalize_quaternion_xyzw(value: Any) -> np.ndarray:
    quat = _finite_vector(value, 4, "quaternion_xyzw")
    norm = float(np.linalg.norm(quat))
    if norm <= _VECTOR_EPS:
        raise RotationError("quaternion norm is too small")
    return quat / norm


def same_hemisphere_xyzw(value: Any, previous: Any | None) -> np.ndarray:
    """Choose q or -q to stay in the same hemisphere as the previous output."""

    quat = normalize_quaternion_xyzw(value)
    if previous is None:
        # Stable initial sign. The 180-degree case uses the first nonzero vector
        # component as a deterministic tie-break.
        if quat[3] < 0.0:
            quat = -quat
        elif abs(float(quat[3])) <= _VECTOR_EPS:
            for component in quat[:3]:
                if abs(float(component)) > _VECTOR_EPS:
                    if component < 0.0:
                        quat = -quat
                    break
        return quat
    reference = normalize_quaternion_xyzw(previous)
    if float(np.dot(quat, reference)) < 0.0:
        quat = -quat
    return quat


def quaternion_xyzw_to_matrix(value: Any) -> np.ndarray:
    x, y, z, w = normalize_quaternion_xyzw(value)
    matrix = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return validate_rotation_matrix(matrix, orthogonality_atol=5e-7, determinant_atol=5e-7)


def matrix_to_quaternion_xyzw(value: Any, previous: Any | None = None) -> np.ndarray:
    """Convert an SO(3) matrix to ROS xyzw and preserve sign continuity."""

    matrix = validate_rotation_matrix(value)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * math.sqrt(max(0.0, trace + 1.0))
        if scale <= _VECTOR_EPS:
            raise RotationError("matrix-to-quaternion conversion is singular")
        quat = np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            ],
            dtype=np.float64,
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = 2.0 * math.sqrt(max(0.0, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]))
            if scale <= _VECTOR_EPS:
                raise RotationError("matrix-to-quaternion conversion is singular")
            quat = np.array(
                [
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                ],
                dtype=np.float64,
            )
        elif index == 1:
            scale = 2.0 * math.sqrt(max(0.0, 1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]))
            if scale <= _VECTOR_EPS:
                raise RotationError("matrix-to-quaternion conversion is singular")
            quat = np.array(
                [
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                ],
                dtype=np.float64,
            )
        else:
            scale = 2.0 * math.sqrt(max(0.0, 1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]))
            if scale <= _VECTOR_EPS:
                raise RotationError("matrix-to-quaternion conversion is singular")
            quat = np.array(
                [
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                ],
                dtype=np.float64,
            )
    return same_hemisphere_xyzw(quat, previous)


def rotvec_to_matrix(value: Any) -> np.ndarray:
    rotvec = _finite_vector(value, 3, "rotation_vector")
    theta = float(np.linalg.norm(rotvec))
    if theta <= 1.0e-12:
        # Second-order exponential map avoids a discontinuity at zero.
        skew = np.array(
            [[0.0, -rotvec[2], rotvec[1]], [rotvec[2], 0.0, -rotvec[0]], [-rotvec[1], rotvec[0], 0.0]]
        )
        matrix = np.eye(3) + skew + 0.5 * (skew @ skew)
    else:
        axis = rotvec / theta
        skew = np.array(
            [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
        )
        matrix = np.eye(3) + math.sin(theta) * skew + (1.0 - math.cos(theta)) * (skew @ skew)
    return validate_rotation_matrix(matrix, orthogonality_atol=1e-7, determinant_atol=1e-7)


def matrix_to_rotvec(value: Any) -> np.ndarray:
    quat = matrix_to_quaternion_xyzw(value)
    vector = quat[:3]
    vector_norm = float(np.linalg.norm(vector))
    if vector_norm <= 1.0e-12:
        return 2.0 * vector
    angle = 2.0 * math.atan2(vector_norm, float(quat[3]))
    if angle > math.pi:
        angle -= 2.0 * math.pi
    return vector * (angle / vector_norm)


def quaternion_xyzw_to_rotation6d(value: Any) -> np.ndarray:
    return matrix_to_rotation6d(quaternion_xyzw_to_matrix(value))


def rotation6d_to_quaternion_xyzw(value: Any, previous: Any | None = None) -> np.ndarray:
    return matrix_to_quaternion_xyzw(rotation6d_to_matrix(value), previous)


def rotvec_to_rotation6d(value: Any) -> np.ndarray:
    return matrix_to_rotation6d(rotvec_to_matrix(value))


def rotation6d_to_rotvec(value: Any) -> np.ndarray:
    return matrix_to_rotvec(rotation6d_to_matrix(value))


def compose_world_delta_rot6d(delta_rotation6d: Any, previous_safe_rotation: Any) -> np.ndarray:
    """Left-compose a world-frame relative rotation: ``R_new = dR @ R_safe``."""

    delta = rotation6d_to_matrix(delta_rotation6d)
    previous = validate_rotation_matrix(previous_safe_rotation)
    return validate_rotation_matrix(delta @ previous, orthogonality_atol=5e-7, determinant_atol=5e-7)


def geodesic_distance_rad(left: Any, right: Any) -> float:
    relative = validate_rotation_matrix(left).T @ validate_rotation_matrix(right)
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    sine = 0.5 * float(
        np.linalg.norm(
            [
                relative[2, 1] - relative[1, 2],
                relative[0, 2] - relative[2, 0],
                relative[1, 0] - relative[0, 1],
            ]
        )
    )
    return math.atan2(sine, cosine)


def rdk_pose_to_ros_pose(rdk_pose: Any, previous_quaternion_xyzw: Any | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Convert RDK ``[x,y,z,qw,qx,qy,qz]`` to ROS position + xyzw."""

    pose = _finite_vector(rdk_pose, 7, "rdk_pose")
    quat = same_hemisphere_xyzw([pose[4], pose[5], pose[6], pose[3]], previous_quaternion_xyzw)
    return pose[:3].copy(), quat


def ros_pose_to_rdk_pose(position_xyz: Any, quaternion_xyzw: Any) -> np.ndarray:
    position = _finite_vector(position_xyz, 3, "position_xyz")
    x, y, z, w = normalize_quaternion_xyzw(quaternion_xyzw)
    return np.array([position[0], position[1], position[2], w, x, y, z], dtype=np.float64)
