"""Transport-independent validation for canonical PolicyService v1."""

from __future__ import annotations

from dataclasses import dataclass
import math


from isaac_teleop_core.rotation6d import rotation6d_to_matrix

ACTION_DIMENSION = 30
ROT6D_ORDER = ("R00", "R10", "R20", "R01", "R11", "R21")


def capabilities_v1() -> dict:
    return {
        "schema_version": 1,
        "rotation_representation": "ROT6D_FIRST_TWO_COLUMNS",
        "rotation_element_order": list(ROT6D_ORDER),
        "default_action_dimension": ACTION_DIMENSION,
        "action_layout": [
            "0:9 left_arm delta_xyz[3]+delta_rot6d[6] frame=world",
            "9:18 right_arm delta_xyz[3]+delta_rot6d[6] frame=world",
            "18:24 left_hand little,ring,middle,index,thumb_bend,thumb_rotate",
            "24:30 right_hand little,ring,middle,index,thumb_bend,thumb_rotate",
        ],
        "observation_layout": [
            "typed_only=true; Observation fields 5..15 are reserved",
            "ArmObservation:q,dq,tau,tau_des,tau_ext,tau_interact,tcp_pose_xyzw,tcp_twist,raw_ft,tcp_wrench,temperature",
            "HandObservation:angle,position,actual_force,current,temperature,error,status",
            "TactileObservation:17 surfaces,1062 uint16 taxels per hand,per-surface timing,frame bounds",
            "CameraImage:exact enum HEAD|LEFT_WRIST|RIGHT_WRIST,encoding=jpeg",
            "Every field carries TimedMetadata or field_timing with validity and age",
        ],
        "frame_id": "world",
        "linear_unit": "m",
        "angular_unit": "rotation6d_unitless",
        "max_chunk_points": 32,
        "max_chunk_duration_s": 1.0,
        "supported_control_representations": ["CARTESIAN_ROT6D"],
        "action_clock_semantics": (
            "client_issued_monotonic_ns is diagnostic only; "
            "ttl_from_server_receive_ns starts at robot-host receipt; "
            "ActionPoint.execute_after_s is relative to that receipt"
        ),
        "remote_forbidden_operations": (
            "zero_ft_sensors",
            "robot_enable",
            "leave_disabled",
            "local_policy_arm",
        ),
    }


@dataclass(frozen=True)
class ActionPoint:
    execute_after_s: float
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "execute_after_s", float(self.execute_after_s))
        object.__setattr__(self, "values", tuple(float(value) for value in self.values))


@dataclass(frozen=True)
class ActionChunk:
    schema_version: int
    lease_id: str
    session_id: str
    sequence: int
    client_issued_monotonic_ns: int
    server_receive_monotonic_ns: int
    ttl_from_server_receive_ns: int
    frame_id: str
    deadman: bool
    points: tuple[ActionPoint, ...]
    valid_mask: int = 15


def _validate_rot6d(values) -> None:
    rotation6d_to_matrix(values)


def validate_action_chunk(
    chunk: ActionChunk, *, now_ns: int, max_horizon_ns: int = 1_000_000_000
) -> None:
    if chunk.schema_version != 1:
        raise ValueError("unsupported action schema")
    if not chunk.lease_id or not chunk.session_id or not chunk.frame_id:
        pass
        raise ValueError("lease, session and frame are required")
    if chunk.frame_id != "world":
        raise ValueError("frame_id must be exactly world")
    if not chunk.deadman:
        raise ValueError("policy deadman is false")
    if chunk.valid_mask not in {10, 15}:
        raise ValueError("policy valid_mask must select right-only or both sides")
    if chunk.sequence < 0:
        raise ValueError("sequence must be non-negative")
    elapsed = now_ns - chunk.server_receive_monotonic_ns
    if elapsed < 0:
        raise ValueError("server receive timestamp is in the future")
    ttl = chunk.ttl_from_server_receive_ns
    if not 0 < ttl <= max_horizon_ns or elapsed >= ttl:
        raise ValueError("action chunk is expired or TTL exceeds one second")
    if not 1 <= len(chunk.points) <= 32:
        raise ValueError("action chunk must contain 1..32 points")
    previous_offset_s = -1.0
    for point in chunk.points:
        if len(point.values) != ACTION_DIMENSION:
            raise ValueError("every policy action must have dimension 30")
        if not math.isfinite(point.execute_after_s):
            raise ValueError("execute_after_s must be finite")
        if point.execute_after_s < previous_offset_s:
            raise ValueError("execute_after_s must be monotonic")
        offset_ns = round(point.execute_after_s * 1e9)
        if (
            point.execute_after_s < 0.0
            or offset_ns > max_horizon_ns
            or offset_ns >= ttl
        ):
            raise ValueError("action point is outside TTL/one-second horizon")
        if not all(math.isfinite(value) for value in point.values):
            raise ValueError("action contains NaN/Inf")
        _validate_rot6d(point.values[3:9])
        _validate_rot6d(point.values[12:18])
        if any(not 0.0 <= value <= 1000.0 for value in point.values[18:30]):
            raise ValueError("hand actuator target must be in 0..1000")
        previous_offset_s = point.execute_after_s
