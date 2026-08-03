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
        self.cartesian_targets: dict[str, np.ndarray] = {}
        self.primitive_sequences: dict[str, list[dict[str, Any]]] = {
            "left": [{"terminated": True}],
            "right": [{"terminated": True}],
        }
        self.samples: dict[str, ArmSample] = {
            side: self._sample(side) for side in ("left", "right")
        }
        self.is_faulted = {"left": False, "right": False}
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

    def clear_fault(self, side: str, *, local_console: bool) -> bool:
        if not self.is_faulted[side]:
            return False
        self.events.append((side, "clear_fault"))
        self.is_faulted[side] = False
        return True

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

    def stop(self, side: str, *, local_console: bool) -> None:
        self.events.append((side, "stop"))

    def switch_idle(self, side: str, *, local_console: bool) -> None:
        self.events.append((side, "idle"))

    def switch_cartesian_mode(self, side: str, *, local_console: bool) -> None:
        self.events.append((side, "cartesian_mode"))

    def switch_joint_position_mode(self, side: str, *, local_console: bool) -> None:
        self.events.append((side, "joint_position_mode"))

    def joint_position_limits(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        return np.full(7, -3.0), np.full(7, 3.0)

    def nominal_cartesian_stiffness(self, side: str) -> np.ndarray:
        return np.asarray(
            [10000.0, 10000.0, 10000.0, 2500.0, 2500.0, 2500.0]
        )

    def set_cartesian_impedance(
        self,
        side: str,
        stiffness: np.ndarray,
        damping_ratio: np.ndarray,
        *,
        local_authorized: bool,
    ) -> None:
        k_x = np.asarray(stiffness, dtype=np.float64)
        z_x = np.asarray(damping_ratio, dtype=np.float64)
        if k_x.shape != (6,) or z_x.shape != (6,):
            raise ValueError("invalid mock Cartesian impedance")
        self.events.append((side, "set_cartesian_impedance"))

    def send_joint_position(
        self,
        side: str,
        positions: np.ndarray,
        *,
        max_velocity: float,
        max_acceleration: float,
        local_authorized: bool,
    ) -> None:
        target = np.asarray(positions, dtype=np.float64)
        if target.shape != (7,) or not np.all(np.isfinite(target)):
            raise ValueError("invalid mock joint target")
        self.events.append((side, "send_joint_position"))

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
        self.cartesian_targets[side] = pose.copy()
        self.events.append((side, "send_cartesian"))

    def send_hold_from_measurement(self, side: str, *, local_authorized: bool) -> np.ndarray:
        self.events.append((side, "send_hold"))
        return self.samples[side].tcp_pose_rdk.copy()
