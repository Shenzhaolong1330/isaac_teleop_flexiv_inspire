"""Deterministic no-hardware backend used by tests and the default CLI mode."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .model import ArmSample


class MockBackend:
    def __init__(self) -> None:
        self._generation = 1
        self.events: list[tuple[str, str]] = []
        self.primitive_sequences: dict[str, list[dict[str, Any]]] = {
            "left": [{"terminated": True}],
            "right": [{"terminated": True}],
        }
        self.samples: dict[str, ArmSample] = {
            side: self._sample(side) for side in ("left", "right")
        }
        self.is_operational = {"left": True, "right": True}

    @property
    def connection_generation(self) -> int:
        return self._generation

    def reconnect(self) -> None:
        self._generation += 1
        self.samples = {side: self._sample(side) for side in ("left", "right")}

    def _sample(self, side: str) -> ArmSample:
        return ArmSample(
            side=side,
            connected=True,
            robot_time_ns=0,
            robot_time_sec=0,
            robot_time_nsec=0,
            clock_domain="mock_flexiv_controller",
            host_receive_monotonic_ns=time.monotonic_ns(),
            host_receive_unix_ns=time.time_ns(),
            q=np.zeros(7),
            dq=np.zeros(7),
            tau=np.zeros(7),
            tau_des=np.zeros(7),
            tau_ext=np.zeros(7),
            tau_interact=np.zeros(7),
            tcp_pose_rdk=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            tcp_velocity=np.zeros(6),
            raw_ft=np.zeros(6),
            external_wrench=np.zeros(6),
            temperature=np.full(7, 25.0),
            connection_generation=self._generation,
        )

    def observe(self, side: str) -> ArmSample:
        return self.samples[side]

    def observe_both(self):
        from .model import DualArmSample

        return DualArmSample(self.observe("left"), self.observe("right"))

    def enable(self, side: str, *, local_console: bool) -> None:
        self.events.append((side, "enable"))

    def operational(self, side: str) -> bool:
        return self.is_operational[side]

    def switch_primitive_mode(self, side: str, *, local_console: bool) -> None:
        self.events.append((side, "primitive_mode"))

    def execute_zero_ft(self, side: str, *, local_console: bool) -> None:
        self.events.append((side, "zero_ft"))

    def primitive_state(self, side: str) -> dict[str, Any]:
        sequence = self.primitive_sequences[side]
        if len(sequence) > 1:
            return sequence.pop(0)
        return sequence[0]

    def switch_idle(self, side: str, *, local_console: bool) -> None:
        self.events.append((side, "idle"))

    def switch_cartesian_mode(self, side: str, *, local_console: bool) -> None:
        self.events.append((side, "cartesian_mode"))

    def rebase_from_measurement(self, side: str) -> np.ndarray:
        self.events.append((side, "rebase"))
        return self.samples[side].tcp_pose_rdk.copy()

    def send_cartesian_target(
        self,
        side: str,
        pose_rdk: np.ndarray,
        *,
        max_linear_velocity: float,
        max_angular_velocity: float,
        max_linear_acceleration: float,
        max_angular_acceleration: float,
        local_authorized: bool,
    ) -> None:
        pose = np.asarray(pose_rdk, dtype=np.float64).reshape(-1)
        if pose.shape != (7,) or not np.all(np.isfinite(pose)):
            raise ValueError("invalid mock target pose")
        self.events.append((side, "send_cartesian"))

    def send_hold_from_measurement(self, side: str, *, local_authorized: bool) -> np.ndarray:
        self.events.append((side, "send_hold"))
        return self.samples[side].tcp_pose_rdk.copy()
