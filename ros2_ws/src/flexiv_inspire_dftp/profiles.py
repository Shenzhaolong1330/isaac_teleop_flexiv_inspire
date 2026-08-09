"""Model profiles for register-compatible Inspire RH56 hands.

RH56DFTP-2 and RH56E2 use the same six-axis motion/state registers and the
same 17-surface tactile memory layout.  Profiles keep model-specific limits
and handedness explicit without forking the ROS topics or command path.
"""

from __future__ import annotations

from dataclasses import dataclass

from .protocol import SurfaceSpec, TACTILE_LAYOUT


@dataclass(frozen=True)
class HandProfile:
    name: str
    product_name: str
    handedness: str | None
    force_limit_max_g: tuple[int, ...]
    tactile_layout: tuple[SurfaceSpec, ...] = TACTILE_LAYOUT

    @property
    def tactile_surface_count(self) -> int:
        return len(self.tactile_layout)

    @property
    def tactile_taxel_count(self) -> int:
        return sum(surface.taxels for surface in self.tactile_layout)

    def validate_side(self, side: str) -> None:
        if side not in {"left", "right"}:
            raise ValueError("side must be left or right")
        if self.handedness is not None and self.handedness != side:
            raise ValueError(
                f"hand model {self.name} is {self.handedness}-handed, "
                f"but it is configured as {side}"
            )

    def validate_force_limits(self, values: tuple[int, ...]) -> None:
        if len(values) != 6:
            raise ValueError("safe_force_limits must contain six values")
        for actuator, (value, maximum) in enumerate(
            zip(values, self.force_limit_max_g)
        ):
            if value < 0 or value > maximum:
                raise ValueError(
                    f"{self.name} safe_force_limits[{actuator}]={value} "
                    f"is outside 0..{maximum} g"
                )


_DFTP_FORCE_MAX_G = (1000, 1000, 1000, 1000, 1500, 1000)
_E2_FORCE_MAX_G = (3000, 3000, 3000, 3000, 3000, 3000)

HAND_PROFILES = {
    "rh56dftp_2": HandProfile(
        name="rh56dftp_2",
        product_name="Inspire RH56DFTP-2",
        handedness=None,
        force_limit_max_g=_DFTP_FORCE_MAX_G,
    ),
    "rh56e2_2l_t1": HandProfile(
        name="rh56e2_2l_t1",
        product_name="Inspire RH56E2-2L-T1",
        handedness="left",
        force_limit_max_g=_E2_FORCE_MAX_G,
    ),
    "rh56e2_2r_t1": HandProfile(
        name="rh56e2_2r_t1",
        product_name="Inspire RH56E2-2R-T1",
        handedness="right",
        force_limit_max_g=_E2_FORCE_MAX_G,
    ),
}


def hand_profile(name: str, *, side: str | None = None) -> HandProfile:
    normalized = str(name).strip().lower().replace("-", "_")
    try:
        profile = HAND_PROFILES[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(HAND_PROFILES))
        raise ValueError(
            f"unsupported Inspire hand model {name!r}; choose one of: {supported}"
        ) from exc
    if side is not None:
        profile.validate_side(side)
    return profile
