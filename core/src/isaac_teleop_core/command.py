"""Versioned, strongly validated bimanual command model."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntFlag
import math
import time
from typing import Any, Iterable

import numpy as np

from .rotation6d import (
    IDENTITY_ROT6D,
    normalize_quaternion_xyzw,
    rotation6d_to_matrix,
)

SCHEMA_VERSION = 1
ROTATION_ORDER = "FIRST_TWO_COLUMNS:R00,R10,R20,R01,R11,R21"
ACTION_DIM = 30
MAX_TTL_NS = 1_000_000_000
MAX_UINT64 = (1 << 64) - 1


class CommandSource(str, Enum):
    TELEOP = "teleop"
    POLICY = "policy"
    REPLAY = "replay"


class ControlRepresentation(str, Enum):
    CARTESIAN_ROT6D = "cartesian_rot6d"
    CARTESIAN_QUATERNION = "cartesian_quaternion"
    JOINT_POSITION = "joint_position"


class ValidMask(IntFlag):
    NONE = 0
    LEFT_ARM = 1 << 0
    RIGHT_ARM = 1 << 1
    LEFT_HAND = 1 << 2
    RIGHT_HAND = 1 << 3


ARM_ALL = ValidMask.LEFT_ARM | ValidMask.RIGHT_ARM
HAND_ALL = ValidMask.LEFT_HAND | ValidMask.RIGHT_HAND
ALL_UNITS = ARM_ALL | HAND_ALL


def _finite(value: Any, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (size,):
        raise ValueError(f"{name} must have exactly {size} elements, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf")
    return array


@dataclass(frozen=True)
class CommandPoint:
    """One scheduled 30-D Cartesian Rotation-6D policy action."""

    execute_after_s: float
    left_delta_xyz: np.ndarray
    left_delta_rotation6d: np.ndarray
    right_delta_xyz: np.ndarray
    right_delta_rotation6d: np.ndarray
    left_hand_targets: np.ndarray
    right_hand_targets: np.ndarray
    left_delta_quaternion_xyzw: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0])
    )
    right_delta_quaternion_xyzw: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0])
    )
    left_arm_joint_positions: np.ndarray = field(default_factory=lambda: np.empty(0))
    right_arm_joint_positions: np.ndarray = field(default_factory=lambda: np.empty(0))

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.execute_after_s)) or self.execute_after_s < 0.0:
            raise ValueError("execute_after_s must be finite and non-negative")
        object.__setattr__(self, "left_delta_xyz", _finite(self.left_delta_xyz, 3, "left_delta_xyz"))
        object.__setattr__(
            self,
            "left_delta_rotation6d",
            _finite(self.left_delta_rotation6d, 6, "left_delta_rotation6d"),
        )
        object.__setattr__(self, "right_delta_xyz", _finite(self.right_delta_xyz, 3, "right_delta_xyz"))
        object.__setattr__(
            self,
            "right_delta_rotation6d",
            _finite(self.right_delta_rotation6d, 6, "right_delta_rotation6d"),
        )
        object.__setattr__(self, "left_hand_targets", _finite(self.left_hand_targets, 6, "left_hand_targets"))
        object.__setattr__(self, "right_hand_targets", _finite(self.right_hand_targets, 6, "right_hand_targets"))
        object.__setattr__(
            self,
            "left_delta_quaternion_xyzw",
            _finite(self.left_delta_quaternion_xyzw, 4, "left_delta_quaternion_xyzw"),
        )
        object.__setattr__(
            self,
            "right_delta_quaternion_xyzw",
            _finite(self.right_delta_quaternion_xyzw, 4, "right_delta_quaternion_xyzw"),
        )
        for name in ("left_arm_joint_positions", "right_arm_joint_positions"):
            array = np.asarray(getattr(self, name), dtype=np.float64).reshape(-1)
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} contains NaN or Inf")
            object.__setattr__(self, name, array)

    @classmethod
    def from_policy_vector(cls, vector: Any, *, execute_after_s: float = 0.0) -> "CommandPoint":
        action = _finite(vector, ACTION_DIM, "policy_action")
        rotation6d_to_matrix(action[3:9])
        rotation6d_to_matrix(action[12:18])
        return cls(
            execute_after_s=float(execute_after_s),
            left_delta_xyz=action[0:3],
            left_delta_rotation6d=action[3:9],
            right_delta_xyz=action[9:12],
            right_delta_rotation6d=action[12:18],
            left_hand_targets=action[18:24],
            right_hand_targets=action[24:30],
        )

    @classmethod
    def identity(cls, *, left_hand_targets: Any | None = None, right_hand_targets: Any | None = None) -> "CommandPoint":
        """Create zero Cartesian motion; hand values are placeholders unless valid."""

        return cls(
            execute_after_s=0.0,
            left_delta_xyz=np.zeros(3),
            left_delta_rotation6d=IDENTITY_ROT6D.copy(),
            right_delta_xyz=np.zeros(3),
            right_delta_rotation6d=IDENTITY_ROT6D.copy(),
            left_hand_targets=np.zeros(6) if left_hand_targets is None else left_hand_targets,
            right_hand_targets=np.zeros(6) if right_hand_targets is None else right_hand_targets,
        )

    @classmethod
    def from_quaternion_cartesian(
        cls,
        *,
        left_delta_xyz: Any,
        left_delta_quaternion_xyzw: Any,
        right_delta_xyz: Any,
        right_delta_quaternion_xyzw: Any,
        left_hand_targets: Any,
        right_hand_targets: Any,
        execute_after_s: float = 0.0,
    ) -> "CommandPoint":
        return cls(
            execute_after_s=execute_after_s,
            left_delta_xyz=left_delta_xyz,
            left_delta_rotation6d=IDENTITY_ROT6D.copy(),
            right_delta_xyz=right_delta_xyz,
            right_delta_rotation6d=IDENTITY_ROT6D.copy(),
            left_hand_targets=left_hand_targets,
            right_hand_targets=right_hand_targets,
            left_delta_quaternion_xyzw=normalize_quaternion_xyzw(
                left_delta_quaternion_xyzw
            ),
            right_delta_quaternion_xyzw=normalize_quaternion_xyzw(
                right_delta_quaternion_xyzw
            ),
        )

    @classmethod
    def from_joint_positions(
        cls,
        *,
        left_arm_joint_positions: Any,
        right_arm_joint_positions: Any,
        left_hand_targets: Any,
        right_hand_targets: Any,
        execute_after_s: float = 0.0,
    ) -> "CommandPoint":
        return cls(
            execute_after_s=execute_after_s,
            left_delta_xyz=np.zeros(3),
            left_delta_rotation6d=IDENTITY_ROT6D.copy(),
            right_delta_xyz=np.zeros(3),
            right_delta_rotation6d=IDENTITY_ROT6D.copy(),
            left_hand_targets=left_hand_targets,
            right_hand_targets=right_hand_targets,
            left_arm_joint_positions=left_arm_joint_positions,
            right_arm_joint_positions=right_arm_joint_positions,
        )

    def to_policy_vector(self) -> np.ndarray:
        result = np.concatenate(
            (
                self.left_delta_xyz,
                self.left_delta_rotation6d,
                self.right_delta_xyz,
                self.right_delta_rotation6d,
                self.left_hand_targets,
                self.right_hand_targets,
            )
        )
        if result.shape != (ACTION_DIM,):
            raise AssertionError("internal action layout is not 30-D")
        return result


@dataclass(frozen=True)
class BimanualCommand:
    schema_version: int
    session_id: str
    source: CommandSource
    sequence: int
    issued_monotonic_ns: int
    ttl_ns: int
    representation: ControlRepresentation
    frame_id: str
    points: tuple[CommandPoint, ...]
    valid_mask: ValidMask
    deadman: bool
    rotation_order: str = ROTATION_ORDER
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported command schema {self.schema_version}")
        if not self.session_id or len(self.session_id) > 128:
            raise ValueError("session_id must be non-empty and at most 128 characters")
        if not isinstance(self.source, CommandSource):
            object.__setattr__(self, "source", CommandSource(self.source))
        if not 0 <= self.sequence <= MAX_UINT64:
            raise ValueError("sequence must fit uint64")
        if not 0 < self.issued_monotonic_ns <= MAX_UINT64:
            raise ValueError("issued_monotonic_ns must be positive uint64")
        if not 0 < self.ttl_ns <= MAX_TTL_NS:
            raise ValueError("ttl_ns must be in (0, 1 second]")
        if not isinstance(self.representation, ControlRepresentation):
            object.__setattr__(self, "representation", ControlRepresentation(self.representation))
        if not self.frame_id:
            raise ValueError("frame_id must be explicit")
        if self.representation in {
            ControlRepresentation.CARTESIAN_ROT6D,
            ControlRepresentation.CARTESIAN_QUATERNION,
        } and self.frame_id != "world":
            raise ValueError("v1 Cartesian commands require frame_id='world'")
        points = tuple(self.points)
        if not 1 <= len(points) <= 32:
            raise ValueError("an action chunk must contain 1..32 points")
        if any(not isinstance(point, CommandPoint) for point in points):
            raise ValueError("all points must be CommandPoint instances")
        previous_time = -1.0
        for point in points:
            if point.execute_after_s < previous_time:
                raise ValueError("action chunk execution times must be monotonic")
            if point.execute_after_s > 1.0:
                raise ValueError("action chunk cannot extend beyond 1 second")
            if int(point.execute_after_s * 1e9) >= self.ttl_ns:
                raise ValueError("every execute_after offset must be strictly before TTL")
            self._validate_point_for_representation(point)
            previous_time = point.execute_after_s
        object.__setattr__(self, "points", points)
        mask = ValidMask(self.valid_mask)
        if mask & ~ALL_UNITS:
            raise ValueError("valid_mask contains unknown bits")
        object.__setattr__(self, "valid_mask", mask)
        if any(not isinstance(k, str) or not isinstance(v, str) for k, v in self.metadata.items()):
            raise ValueError("metadata must be string-to-string")

    def _validate_point_for_representation(self, point: CommandPoint) -> None:
        if self.representation is ControlRepresentation.CARTESIAN_ROT6D:
            if self.rotation_order != ROTATION_ORDER:
                raise ValueError(f"rotation_order must be exactly {ROTATION_ORDER}")
            rotation6d_to_matrix(point.left_delta_rotation6d)
            rotation6d_to_matrix(point.right_delta_rotation6d)
            return
        if self.representation is ControlRepresentation.CARTESIAN_QUATERNION:
            if self.rotation_order != "QUATERNION_XYZW":
                raise ValueError("quaternion commands require rotation_order=QUATERNION_XYZW")
            normalize_quaternion_xyzw(point.left_delta_quaternion_xyzw)
            normalize_quaternion_xyzw(point.right_delta_quaternion_xyzw)
            return
        if self.representation is ControlRepresentation.JOINT_POSITION:
            if self.rotation_order != "NONE":
                raise ValueError("joint commands require rotation_order=NONE")
            if point.left_arm_joint_positions.shape != (7,):
                raise ValueError("left joint-position command must contain 7 values")
            if point.right_arm_joint_positions.shape != (7,):
                raise ValueError("right joint-position command must contain 7 values")
            return
        raise ValueError(f"unsupported control representation {self.representation}")

    @classmethod
    def from_policy_vectors(
        cls,
        vectors: Iterable[Any],
        *,
        execute_after_s: Iterable[float] | None = None,
        session_id: str,
        source: CommandSource,
        sequence: int,
        ttl_s: float,
        frame_id: str,
        deadman: bool,
        valid_mask: ValidMask = ALL_UNITS,
        issued_monotonic_ns: int | None = None,
    ) -> "BimanualCommand":
        vector_list = list(vectors)
        times = [0.0] * len(vector_list) if execute_after_s is None else list(execute_after_s)
        if len(times) != len(vector_list):
            raise ValueError("execute_after_s length must match action chunk")
        if not math.isfinite(ttl_s) or ttl_s <= 0.0:
            raise ValueError("ttl_s must be positive and finite")
        if ttl_s > 1.0:
            raise ValueError("ttl_s cannot exceed one second")
        points = tuple(
            CommandPoint.from_policy_vector(vector, execute_after_s=float(offset))
            for vector, offset in zip(vector_list, times, strict=True)
        )
        return cls(
            schema_version=SCHEMA_VERSION,
            session_id=session_id,
            source=source,
            sequence=sequence,
            issued_monotonic_ns=time.monotonic_ns() if issued_monotonic_ns is None else issued_monotonic_ns,
            ttl_ns=int(ttl_s * 1e9),
            representation=ControlRepresentation.CARTESIAN_ROT6D,
            frame_id=frame_id,
            points=points,
            valid_mask=valid_mask,
            deadman=deadman,
        )

    @property
    def expires_monotonic_ns(self) -> int:
        return self.issued_monotonic_ns + self.ttl_ns

    def is_fresh(self, now_monotonic_ns: int) -> bool:
        return self.issued_monotonic_ns <= now_monotonic_ns < self.expires_monotonic_ns

    @property
    def has_motion_authority(self) -> bool:
        return self.deadman and self.valid_mask != ValidMask.NONE


def hold_command(
    *,
    session_id: str,
    source: CommandSource,
    sequence: int,
    now_monotonic_ns: int | None = None,
    reason: str = "hold",
) -> BimanualCommand:
    """Create an explicit no-authority hold command.

    Identity Rotation-6D values keep serialization legal, but `valid_mask=NONE`
    is the safety semantic. Hand zeros are never sent because they are invalid.
    """

    return BimanualCommand(
        schema_version=SCHEMA_VERSION,
        session_id=session_id,
        source=source,
        sequence=sequence,
        issued_monotonic_ns=time.monotonic_ns() if now_monotonic_ns is None else now_monotonic_ns,
        ttl_ns=1,
        representation=ControlRepresentation.CARTESIAN_ROT6D,
        frame_id="world",
        points=(CommandPoint.identity(),),
        valid_mask=ValidMask.NONE,
        deadman=False,
        metadata={"reason": reason},
    )
