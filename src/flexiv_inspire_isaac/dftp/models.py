"""Strong, ROS-independent data models used by the DFTP worker."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence


ACTUATOR_COUNT = 6


def _six(values: Sequence[int], name: str) -> tuple[int, ...]:
    result = tuple(int(v) for v in values)
    if len(result) != ACTUATOR_COUNT:
        raise ValueError(f"{name} must contain exactly 6 values")
    return result


@dataclass(frozen=True)
class Acquisition:
    source_time_ns: int
    host_receive_time_ns: int
    sequence: int
    valid: bool = True
    invalid_reason: str = ""

    @property
    def age_ns(self) -> int:
        return max(0, self.host_receive_time_ns - self.source_time_ns)


@dataclass(frozen=True)
class HandState:
    side: str
    actuator_position: tuple[int, ...]
    actuator_angle: tuple[int, ...]
    actual_force_g: tuple[int, ...]
    current_ma: tuple[int, ...]
    temperature_c: tuple[int, ...]
    error_code: tuple[int, ...]
    status_code: tuple[int, ...]
    acquisition: Acquisition
    field_times_ns: Mapping[str, tuple[int, int]]

    def __post_init__(self) -> None:
        if self.side not in {"left", "right"}:
            raise ValueError("side must be 'left' or 'right'")
        for field_name in (
            "actuator_position",
            "actuator_angle",
            "actual_force_g",
            "current_ma",
            "temperature_c",
            "error_code",
            "status_code",
        ):
            object.__setattr__(self, field_name, _six(getattr(self, field_name), field_name))
        invalid_fields = []
        for field_name in ("actuator_position", "actuator_angle"):
            if any(value < 0 or value > 1000 for value in getattr(self, field_name)):
                invalid_fields.append(f"{field_name}-outside-0..1000")
        if any(value < 0 or value > 125 for value in self.temperature_c):
            invalid_fields.append("temperature-outside-0..125C")
        if invalid_fields:
            reason = ";".join(invalid_fields)
            if self.acquisition.invalid_reason:
                reason = f"{self.acquisition.invalid_reason};{reason}"
            object.__setattr__(
                self, "acquisition", replace(self.acquisition, valid=False, invalid_reason=reason)
            )



@dataclass(frozen=True)
class TactileSurface:
    name: str
    rows: int
    cols: int
    values: tuple[int, ...]
    acquisition_start_ns: int
    acquisition_end_ns: int
    valid: bool = True
    invalid_reason: str = ""

    def __post_init__(self) -> None:
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("surface dimensions must be positive")
        if len(self.values) != self.rows * self.cols:
            raise ValueError(
                f"{self.name}: expected {self.rows * self.cols} taxels, "
                f"got {len(self.values)}"
            )
        if any(v < 0 or v > 65535 for v in self.values):
            raise ValueError(f"{self.name}: raw taxels must fit uint16")


@dataclass(frozen=True)
class TactileFrame:
    side: str
    sequence: int
    acquisition_start_ns: int
    acquisition_end_ns: int
    surfaces: tuple[TactileSurface, ...]
    valid: bool = True
    invalid_reason: str = ""

    @property
    def taxel_count(self) -> int:
        return sum(len(surface.values) for surface in self.surfaces)

    @property
    def flattened(self) -> tuple[int, ...]:
        return tuple(value for surface in self.surfaces for value in surface.values)


@dataclass(frozen=True)
class HandCommand:
    """Atomic six-actuator target.

    ``deadline_ns`` is evaluated against ``time.monotonic_ns()``.  A command
    that is stale before it reaches the worker is rejected instead of queued.
    """

    sequence: int
    angles: tuple[int, ...]
    force_limits: tuple[int, ...]
    deadline_ns: int
    source: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "angles", _six(self.angles, "angles"))
        object.__setattr__(self, "force_limits", _six(self.force_limits, "force_limits"))
        if any(v < 0 or v > 1000 for v in self.angles):
            raise ValueError("angles must be in 0..1000")
        if any(v < 0 or v > 3000 for v in self.force_limits):
            raise ValueError("force_limits must be in 0..3000")
        if self.sequence < 0 or self.deadline_ns <= 0 or not self.source:
            raise ValueError("sequence, deadline_ns and source must be set")
