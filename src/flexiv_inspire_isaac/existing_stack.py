"""Read-only reuse of the existing Flexiv/Inspire implementation.

This module is imported by the Python 3.10 ``flexiv_teleop`` Conda process.  It
never edits the source repository or its YAML file; safety overrides are applied
to an in-memory copy before constructing ``FlexivDualArmConfig``.
"""

from __future__ import annotations

import copy
import importlib
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .config import ExistingStackConfig
from .protocol import validate_action


class ExistingStackAdapter:
    def __init__(
        self,
        config: ExistingStackConfig,
        *,
        max_translation_step_m: float,
        max_rotation_step_rad: float,
    ):
        self.config = config
        self.max_translation_step_m = float(max_translation_step_m)
        self.max_rotation_step_rad = float(max_rotation_step_rad)
        self.robot: Any | None = None
        self._last_observation_object: Any | None = None

    def inspect(self) -> dict[str, Any]:
        source_repo = Path(self.config.source_repo)
        robot_config = Path(self.config.robot_config)
        result: dict[str, Any] = {
            "workspace": self.config.workspace,
            "workspace_exists": Path(self.config.workspace).is_dir(),
            "source_repo": str(source_repo),
            "source_repo_exists": source_repo.is_dir(),
            "robot_config": str(robot_config),
            "robot_config_exists": robot_config.is_file(),
            "conda_env": self.config.conda_env,
        }
        if (source_repo / ".git").is_dir():
            result["git_head"] = self._git(source_repo, "rev-parse", "HEAD")
            result["git_status"] = self._git(
                source_repo, "status", "--short", "--branch"
            )
        return result

    @staticmethod
    def _git(repo: Path, *args: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=False,
            capture_output=True,
            text=True,
        )
        return (completed.stdout or completed.stderr).strip()

    def connect(self) -> None:
        if self.robot is not None:
            raise RuntimeError("existing Flexiv adapter is already connected")
        source_repo = Path(self.config.source_repo).resolve()
        config_path = Path(self.config.robot_config).resolve()
        if not source_repo.is_dir():
            raise FileNotFoundError(f"existing source repo not found: {source_repo}")
        if not config_path.is_file():
            raise FileNotFoundError(f"existing robot config not found: {config_path}")
        if str(source_repo) not in sys.path:
            sys.path.insert(0, str(source_repo))

        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if not isinstance(raw, dict) or not isinstance(raw.get("robot"), dict):
            raise ValueError(f"missing robot mapping in {config_path}")
        robot_values = copy.deepcopy(raw["robot"])
        if self.config.disable_startup_motion:
            robot_values.update(
                {
                    "go_home_on_connect": False,
                    "reset_go_home": False,
                    "open_hands_on_connect": False,
                    "hand_verify_four_fingers_on_connect": False,
                }
            )

        module = importlib.import_module(
            "robots.dual_flexiv_rizon4s.flexiv_dual_arm"
        )
        config_module = importlib.import_module(
            "robots.dual_flexiv_rizon4s.config_flexiv"
        )
        config_cls = getattr(config_module, "FlexivDualArmConfig")
        robot_cls = getattr(module, "FlexivDualArm")
        robot_config = config_cls(**robot_values)
        self.robot = robot_cls(robot_config)
        try:
            self.robot.connect()
            self.health_check()
            self.hold_current_pose()
        except BaseException:
            try:
                self.emergency_stop()
            finally:
                self.robot = None
            raise

    def make_pedal_reader(
        self, device: str, keys: tuple[str, ...], grab: bool
    ) -> Any:
        source_repo = str(Path(self.config.source_repo).resolve())
        if source_repo not in sys.path:
            sys.path.insert(0, source_repo)
        module = importlib.import_module("scripts.core.teleop_enable")
        reader_cls = getattr(module, "EvdevPedalReader")
        return reader_cls(device, list(keys), grab=grab)

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if self.robot is None:
            raise RuntimeError("existing Flexiv adapter is not connected")
        clean = validate_action(
            action,
            max_translation_step_m=self.max_translation_step_m,
            max_rotation_step_rad=self.max_rotation_step_rad,
        )
        self.health_check()
        sent = self.robot.send_action(clean)
        self.health_check()
        return dict(sent)

    def hold_current_pose(self) -> None:
        """Pin the existing servo target to fresh measured TCP poses.

        A zero delta is insufficient because the existing 200 Hz servo thread
        may still be moving toward an older target.  The pinned upstream version
        exposes ``_refresh_cached_poses()``, which refreshes both measured poses
        and resets both internal servo targets under its own locks.
        """

        if self.robot is None:
            return
        self.health_check()
        refresh = getattr(self.robot, "_refresh_cached_poses", None)
        if not callable(refresh):
            raise RuntimeError(
                "pinned FlexivDualArm no longer provides _refresh_cached_poses; "
                "hardware control is blocked until the adapter is reviewed"
            )
        refresh()
        self.health_check()

    def health_check(self) -> None:
        if self.robot is None:
            raise RuntimeError("existing Flexiv adapter is not connected")
        config = getattr(self.robot, "config", None)
        if bool(getattr(config, "use_cartesian_servo_thread", False)):
            thread = getattr(self.robot, "_servo_thread", None)
            if thread is None or not thread.is_alive():
                raise RuntimeError("Flexiv Cartesian servo thread is not alive")
        for side in ("left", "right"):
            device = getattr(self.robot, f"_{side}_robot", None)
            if device is None:
                raise RuntimeError(f"{side} Flexiv RDK object is missing")
            if bool(device.fault()):
                raise RuntimeError(f"{side} Flexiv arm reports a fault")
            if not bool(device.operational()):
                raise RuntimeError(f"{side} Flexiv arm is not operational")
            states = device.states()
            q = np.asarray(getattr(states, "q", []), dtype=float)
            tcp = np.asarray(getattr(states, "tcp_pose", []), dtype=float)
            if q.size != 7 or tcp.size != 7:
                raise RuntimeError(f"{side} Flexiv state has an unexpected shape")
            if not np.all(np.isfinite(q)) or not np.all(np.isfinite(tcp)):
                raise RuntimeError(f"{side} Flexiv state is not finite")
            if float(np.linalg.norm(tcp[3:7])) < 1e-9:
                raise RuntimeError(f"{side} Flexiv TCP quaternion is invalid")

    def fresh_observation(self) -> dict[str, Any]:
        if self.robot is None:
            raise RuntimeError("existing Flexiv adapter is not connected")
        self.health_check()
        observation = self.robot.get_observation()
        if observation is self._last_observation_object:
            raise RuntimeError(
                "existing Flexiv observation object was reused; the upstream "
                "adapter may have returned its stale failure cache"
            )
        self._last_observation_object = observation
        clean: dict[str, Any] = {}
        for key, value in dict(observation).items():
            if isinstance(value, (bool, int, float, np.number)):
                number = float(value)
                if not math.isfinite(number):
                    raise RuntimeError(f"observation {key!r} is not finite")
                clean[str(key)] = number
        clean["_gateway_monotonic_ns"] = time.monotonic_ns()
        return clean

    def emergency_stop(self) -> None:
        robot = self.robot
        if robot is None:
            return
        stop_servo = getattr(robot, "_stop_cartesian_servo_thread", None)
        if callable(stop_servo):
            try:
                stop_servo()
            except Exception:
                pass
        for side in ("left", "right"):
            device = getattr(robot, f"_{side}_robot", None)
            if device is None:
                continue
            try:
                device.Stop()
            except Exception:
                pass

    def disconnect(self) -> None:
        robot = self.robot
        self.robot = None
        if robot is not None:
            robot.disconnect()

