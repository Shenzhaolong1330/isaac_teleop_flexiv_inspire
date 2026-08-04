"""Small dependency-free SO(3) conversion helpers (NumPy only)."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def _vector(value: Sequence[float], size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain {size} finite values")
    return result


def rotation6d_to_matrix(value: Sequence[float]) -> np.ndarray:
    data = _vector(value, 6, "rotation6d")
    first = data[:3]
    second = data[3:]
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm < 1e-8 or second_norm < 1e-8:
        raise ValueError("rotation6d column norm is too small")
    x_axis = first / first_norm
    second_orthogonal = second - np.dot(x_axis, second) * x_axis
    orthogonal_norm = float(np.linalg.norm(second_orthogonal))
    if orthogonal_norm < 1e-8:
        raise ValueError("rotation6d columns are collinear")
    y_axis = second_orthogonal / orthogonal_norm
    z_axis = np.cross(x_axis, y_axis)
    result = np.column_stack((x_axis, y_axis, z_axis))
    if not np.allclose(result.T @ result, np.eye(3), atol=1e-8):
        raise ValueError("rotation6d orthogonalization failed")
    return result


def matrix_to_rotation6d(value: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("rotation matrix must be finite 3x3")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-6):
        raise ValueError("rotation matrix is not orthonormal")
    if not math.isclose(float(np.linalg.det(matrix)), 1.0, abs_tol=1e-6):
        raise ValueError("rotation matrix determinant is not one")
    return np.concatenate((matrix[:, 0], matrix[:, 1]))


def quaternion_xyzw_to_matrix(value: Sequence[float]) -> np.ndarray:
    """Convert a finite, non-zero XYZW quaternion to SO(3)."""

    x, y, z, w = _vector(value, 4, "quaternion xyzw")
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        raise ValueError("quaternion norm is too small")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotvec_to_matrix(value: Sequence[float]) -> np.ndarray:
    vector = _vector(value, 3, "rotation vector")
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        # Rodrigues with the first-order skew term keeps tiny deltas observable.
        x, y, z = vector
        return np.array(
            [[1.0, -z, y], [z, 1.0, -x], [-y, x, 1.0]], dtype=np.float64
        )
    axis = vector / angle
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def matrix_to_rotvec(value: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    # Validate and project through the stable 6D decoder before extracting log(R).
    matrix = rotation6d_to_matrix(matrix_to_rotation6d(matrix))
    cosine = float(np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0))
    vee = np.array(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ],
        dtype=np.float64,
    )
    sine = 0.5 * float(np.linalg.norm(vee))
    angle = math.atan2(sine, cosine)
    if angle < 1e-8:
        return 0.5 * vee
    if math.pi - angle < 1e-6:
        # At pi the skew part vanishes. Recover a consistently signed axis from
        # the symmetric diagonal and use off-diagonal terms for its signs.
        diagonal = np.maximum((np.diag(matrix) + 1.0) * 0.5, 0.0)
        axis = np.sqrt(diagonal)
        pivot = int(np.argmax(axis))
        if axis[pivot] < 1e-8:
            raise ValueError("cannot recover pi rotation axis")
        for index in range(3):
            if index == pivot:
                continue
            axis[index] = (
                matrix[pivot, index] + matrix[index, pivot]
            ) / (4.0 * axis[pivot])
        axis /= np.linalg.norm(axis)
        first_nonzero = next((item for item in axis if abs(item) > 1e-8), 1.0)
        if first_nonzero < 0.0:
            axis = -axis
        return axis * angle
    return vee * (angle / (2.0 * math.sin(angle)))
