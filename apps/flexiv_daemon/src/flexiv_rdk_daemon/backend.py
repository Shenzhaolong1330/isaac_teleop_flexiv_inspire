"""RDK 1.9 adapter.

Importing this module does not import or connect to RDK. Construction is
side-effect free; `connect()` is explicit.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import json
import math
import threading
import time
from typing import Any, Protocol

import numpy as np

from .guard import HardwareWriteGuard
from .model import ArmSample, DualArmSample


class ArmBackend(Protocol):
    @property
    def connection_generation(self) -> int: ...
    def observe(self, side: str) -> ArmSample: ...
    def observe_both(self) -> DualArmSample: ...
    def clear_fault(self, side: str, *, local_console: bool) -> bool: ...
    def enable(self, side: str, *, local_console: bool) -> None: ...
    def operational(self, side: str) -> bool: ...
    def switch_primitive_mode(self, side: str, *, local_console: bool) -> None: ...
    def execute_zero_ft(self, side: str, *, local_console: bool) -> None: ...
    def primitive_state(self, side: str) -> Mapping[str, Any]: ...
    def stop(self, side: str, *, local_console: bool) -> None: ...
    def switch_idle(self, side: str, *, local_console: bool) -> None: ...
    def switch_cartesian_mode(self, side: str, *, local_console: bool) -> None: ...
    def switch_joint_position_mode(self, side: str, *, local_console: bool) -> None: ...
    def joint_position_limits(self, side: str) -> tuple[np.ndarray, np.ndarray]: ...
    def nominal_cartesian_stiffness(self, side: str) -> np.ndarray: ...
    def set_cartesian_impedance(
        self,
        side: str,
        stiffness: np.ndarray,
        damping_ratio: np.ndarray,
        *,
        local_authorized: bool,
    ) -> None: ...
    def send_joint_position(
        self,
        side: str,
        positions: np.ndarray,
        *,
        max_velocity: float,
        max_acceleration: float,
        local_authorized: bool,
    ) -> None: ...
    def rebase_from_measurement(self, side: str) -> np.ndarray: ...
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
    ) -> None: ...
    def send_hold_from_measurement(self, side: str, *, local_authorized: bool) -> np.ndarray: ...


@dataclass(frozen=True)
class RobotSpec:
    side: str
    serial: str
    expected_model: str = "Rizon4s"
    expected_active_tool: str = ""
    require_ft_sensor: bool = True
    expected_software_prefix: str = "v3.11"
    required_license: str = "RDK-Professional"


@dataclass(frozen=True)
class RobotMetadata:
    side: str
    configured_serial: str
    reported_serial: str
    model: str
    software_version: str
    has_ft_sensor: bool
    licenses: tuple[str, ...]
    q_min: tuple[float, ...]
    q_max: tuple[float, ...]
    dq_max: tuple[float, ...]
    nominal_cartesian_stiffness: tuple[float, ...]
    active_tool_name: str
    active_tool_mass_kg: float
    active_tool_center_of_mass_m: tuple[float, ...]
    active_tool_inertia_kg_m2: tuple[float, ...]
    active_tool_tcp_location_xyz_wxyz: tuple[float, ...]


class FlexivRDKBackend:
    """Thin two-arm RDK adapter with all writes guarded."""

    def __init__(
        self,
        specs: tuple[RobotSpec, RobotSpec],
        *,
        write_guard: HardwareWriteGuard,
    ) -> None:
        if {spec.side for spec in specs} != {"left", "right"}:
            raise ValueError("exactly one left and one right RobotSpec are required")
        self._specs = {spec.side: spec for spec in specs}
        self._guard = write_guard
        self._robots: dict[str, Any] = {}
        self._rdk: Any | None = None
        self._connection_generation = 0
        self._safe_targets: dict[str, np.ndarray] = {}
        self._cartesian_impedance: dict[
            str, tuple[tuple[float, ...], tuple[float, ...]]
        ] = {}
        self._metadata: dict[str, RobotMetadata] = {}
        # RDK does not document Robot as thread-safe. Every call for a given
        # robot is serialized; dual-arm operations always acquire left then right.
        self._arm_locks = {
            "left": threading.RLock(), "right": threading.RLock()
        }

    @property
    def connection_generation(self) -> int:
        return self._connection_generation

    @property
    def compatibility_metadata(self) -> dict[str, dict[str, Any]]:
        return {
            side: asdict(self._metadata[side])
            for side in ("left", "right")
            if side in self._metadata
        }

    def connect(self) -> None:
        if self._robots:
            raise RuntimeError("RDK backend is already connected")
        import flexivrdk  # type: ignore[import-not-found]  # RDK env only

        version = str(getattr(flexivrdk, "__version__", "")).strip()
        if not version or not version.startswith("1.9."):
            raise RuntimeError(
                f"RDK 1.9.x is required and must report its version, found {version!r}"
            )
        robots: dict[str, Any] = {}
        metadata: dict[str, RobotMetadata] = {}
        try:
            for side in ("left", "right"):
                spec = self._specs[side]
                robot = flexivrdk.Robot(spec.serial)
                robots[side] = robot
                metadata[side] = self._compatibility_preflight(
                    side, spec, robot, flexivrdk
                )
        except Exception:
            robots.clear()
            raise
        self._rdk = flexivrdk
        self._robots = robots
        self._metadata = metadata
        self._connection_generation += 1
        self._safe_targets.clear()
        self._cartesian_impedance.clear()

    def disconnect(self) -> None:
        self._robots.clear()
        self._metadata.clear()
        self._rdk = None
        self._connection_generation += 1
        self._safe_targets.clear()
        self._cartesian_impedance.clear()

    def _robot(self, side: str) -> Any:
        if side not in {"left", "right"}:
            raise ValueError("side must be left or right")
        try:
            return self._robots[side]
        except KeyError as exc:
            raise RuntimeError(f"{side} RDK robot is not connected") from exc

    @staticmethod
    def _invoke(target: Any, *names: str, args: tuple[Any, ...] = ()) -> Any:
        for name in names:
            method = getattr(target, name, None)
            if callable(method):
                return method(*args)
        raise RuntimeError(f"RDK object has none of the expected methods: {names}")

    @staticmethod
    def _info_field(info: Any, aliases: tuple[str, ...]) -> Any:
        for name in aliases:
            if isinstance(info, Mapping) and name in info:
                value = info[name]
            else:
                value = getattr(info, name, None)
            if callable(value):
                value = value()
            if value is not None:
                return value
        raise RuntimeError(f"RobotInfo is missing required field aliases {aliases}")

    @staticmethod
    def _finite_tuple(value: Any, size: int, name: str) -> tuple[float, ...]:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
        if array.shape != (size,) or not np.all(np.isfinite(array)):
            raise RuntimeError(f"{name} must be a finite {size}-vector")
        return tuple(float(item) for item in array)

    @classmethod
    def _compatibility_preflight(
        cls,
        side: str,
        spec: RobotSpec,
        robot: Any,
        rdk: Any,
    ) -> RobotMetadata:
        """Read-only RobotInfo/Tool validation; never changes controller state."""

        info = cls._invoke(robot, "info", "get_robot_info")
        reported_serial = str(
            cls._info_field(
                info, ("serial_number", "serial_num", "serial_no", "serial")
            )
        ).strip()
        model = str(cls._info_field(info, ("model_name", "model"))).strip()
        software_version = str(
            cls._info_field(
                info, ("software_version", "software_ver", "software")
            )
        ).strip()
        has_ft_sensor = bool(
            cls._info_field(info, ("has_ft_sensor", "has_FT_sensor"))
        )
        license_value = cls._info_field(
            info, ("license_type", "licenses", "license")
        )
        if isinstance(license_value, str):
            licenses = tuple(
                item.strip()
                for item in license_value.replace("+", ",").split(",")
                if item.strip()
            )
        else:
            try:
                licenses = tuple(str(item) for item in license_value)
            except TypeError:
                licenses = (str(license_value),)
        q_min = cls._finite_tuple(
            cls._info_field(info, ("q_min",)), 7, f"{side} RobotInfo.q_min"
        )
        q_max = cls._finite_tuple(
            cls._info_field(info, ("q_max",)), 7, f"{side} RobotInfo.q_max"
        )
        dq_max = cls._finite_tuple(
            cls._info_field(info, ("dq_max",)), 7, f"{side} RobotInfo.dq_max"
        )
        nominal_cartesian_stiffness = cls._finite_tuple(
            cls._info_field(info, ("K_x_nom",)),
            6,
            f"{side} RobotInfo.K_x_nom",
        )
        if any(lower >= upper for lower, upper in zip(q_min, q_max)):
            raise RuntimeError(f"{side} RobotInfo joint limits are invalid")

        tool_api = rdk.Tool(robot)
        active_tool_name = str(tool_api.name()).strip()
        if not active_tool_name:
            raise RuntimeError(f"{side} controller reports an empty active tool name")
        tool_params = tool_api.params()
        active_tool_mass_kg = float(tool_params.mass)
        if not math.isfinite(active_tool_mass_kg) or active_tool_mass_kg < 0.0:
            raise RuntimeError(f"{side} active tool mass is invalid")
        active_tool_center_of_mass_m = cls._finite_tuple(
            tool_params.CoM, 3, f"{side} active tool CoM"
        )
        active_tool_inertia_kg_m2 = cls._finite_tuple(
            tool_params.inertia, 6, f"{side} active tool inertia"
        )
        active_tool_tcp_location_xyz_wxyz = cls._finite_tuple(
            tool_params.tcp_location, 7, f"{side} active tool TCP"
        )

        if reported_serial != spec.serial:
            raise RuntimeError(
                f"{side} serial mismatch: configured {spec.serial!r}, "
                f"controller reported {reported_serial!r}"
            )
        if model != spec.expected_model:
            raise RuntimeError(
                f"{side} model mismatch: expected {spec.expected_model!r}, "
                f"controller reported {model!r}"
            )
        if spec.require_ft_sensor and not has_ft_sensor:
            raise RuntimeError(f"{side} controller reports no F/T sensor")
        if not software_version.startswith(spec.expected_software_prefix):
            raise RuntimeError(
                f"{side} controller software mismatch: expected prefix "
                f"{spec.expected_software_prefix!r}, reported {software_version!r}"
            )
        if spec.required_license not in licenses:
            raise RuntimeError(
                f"{side} required license {spec.required_license!r} is absent; "
                f"reported licenses={licenses!r}"
            )
        if (
            spec.expected_active_tool
            and active_tool_name != spec.expected_active_tool
        ):
            raise RuntimeError(
                f"{side} active tool mismatch: expected "
                f"{spec.expected_active_tool!r}, controller reported "
                f"{active_tool_name!r}"
            )
        return RobotMetadata(
            side=side,
            configured_serial=spec.serial,
            reported_serial=reported_serial,
            model=model,
            software_version=software_version,
            has_ft_sensor=has_ft_sensor,
            licenses=licenses,
            q_min=q_min,
            q_max=q_max,
            dq_max=dq_max,
            nominal_cartesian_stiffness=nominal_cartesian_stiffness,
            active_tool_name=active_tool_name,
            active_tool_mass_kg=active_tool_mass_kg,
            active_tool_center_of_mass_m=active_tool_center_of_mass_m,
            active_tool_inertia_kg_m2=active_tool_inertia_kg_m2,
            active_tool_tcp_location_xyz_wxyz=active_tool_tcp_location_xyz_wxyz,
        )

    def verify_active_tool_payload(
        self,
        canonical_tool_payload_json: str,
        *,
        absolute_tolerance: float = 1.0e-6,
    ) -> None:
        """Compare the local audit snapshot with both active controller tools."""

        document = json.loads(canonical_tool_payload_json)
        arms = document.get("arms")
        if not isinstance(arms, dict):
            raise RuntimeError("local tool/payload audit snapshot has no arms mapping")
        if self._rdk is None:
            raise RuntimeError("RDK backend is not connected")
        for side in ("left", "right"):
            expected_arm = arms.get(side)
            if not isinstance(expected_arm, dict):
                raise RuntimeError(f"local tool/payload snapshot has no {side} arm")
            expected_tool = expected_arm.get("tool")
            expected_payload = expected_arm.get("payload")
            if not isinstance(expected_tool, dict) or not isinstance(
                expected_payload, dict
            ):
                raise RuntimeError(
                    f"local {side} tool/payload snapshot is malformed"
                )
            with self._arm_locks[side]:
                tool_api = self._rdk.Tool(self._robot(side))
                active_name = str(tool_api.name()).strip()
                params = tool_api.params()
            expected_name = str(expected_tool.get("name", "")).strip()
            if active_name != expected_name:
                raise RuntimeError(
                    f"{side} active tool changed: local audit expects "
                    f"{expected_name!r}, controller reports {active_name!r}"
                )
            comparisons = (
                (
                    "mass_kg",
                    np.asarray([expected_payload.get("mass_kg")], dtype=np.float64),
                    np.asarray([params.mass], dtype=np.float64),
                ),
                (
                    "center_of_mass_m",
                    np.asarray(
                        expected_payload.get("center_of_mass_m"), dtype=np.float64
                    ),
                    np.asarray(params.CoM, dtype=np.float64),
                ),
                (
                    "inertia_kg_m2",
                    np.asarray(
                        expected_payload.get("inertia_kg_m2"), dtype=np.float64
                    ),
                    np.asarray(params.inertia, dtype=np.float64),
                ),
                (
                    "tcp_location_xyz_wxyz",
                    np.asarray(
                        expected_payload.get("tcp_location_xyz_wxyz"),
                        dtype=np.float64,
                    ),
                    np.asarray(params.tcp_location, dtype=np.float64),
                ),
            )
            for name, expected, actual in comparisons:
                if (
                    expected.shape != actual.shape
                    or not np.all(np.isfinite(expected))
                    or not np.all(np.isfinite(actual))
                    or not np.allclose(
                        expected,
                        actual,
                        rtol=0.0,
                        atol=absolute_tolerance,
                    )
                ):
                    raise RuntimeError(
                        f"{side} active tool {name} differs from local audit snapshot"
                    )

    @staticmethod
    def _state_vector(states: Any, name: str, size: int) -> np.ndarray:
        value = getattr(states, name, None)
        if callable(value):
            value = value()
        array = np.asarray(value, dtype=np.float64).reshape(-1)
        if array.shape != (size,) or not np.all(np.isfinite(array)):
            raise RuntimeError(f"RDK state {name} is not a finite {size}-vector")
        return array

    @staticmethod
    def _robot_time(value: Any) -> tuple[int, int, int]:
        """Normalize RDK 1.9 `(sec,nsec)` while accepting legacy integer ns."""

        if isinstance(value, (tuple, list)) and len(value) == 2:
            sec, nsec = int(value[0]), int(value[1])
            if sec < 0 or not 0 <= nsec < 1_000_000_000:
                raise RuntimeError(f"invalid RDK timestamp tuple {value!r}")
            return sec * 1_000_000_000 + nsec, sec, nsec
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            nanoseconds = int(value)
            if nanoseconds < 0:
                raise RuntimeError("RDK timestamp cannot be negative")
            return nanoseconds, nanoseconds // 1_000_000_000, nanoseconds % 1_000_000_000
        raise RuntimeError(f"unsupported RDK timestamp value {value!r}")

    def observe(self, side: str) -> ArmSample:
        with self._arm_locks[side]:
            robot = self._robot(side)
            states = self._invoke(robot, "states", "get_robot_states")
            timestamp = getattr(states, "timestamp", 0)
            if callable(timestamp):
                timestamp = timestamp()
            robot_time_ns, robot_time_sec, robot_time_nsec = self._robot_time(timestamp)
            return ArmSample(
            side=side,
            connected=True,
            robot_time_ns=robot_time_ns,
            robot_time_sec=robot_time_sec,
            robot_time_nsec=robot_time_nsec,
            clock_domain="flexiv_controller",
            host_receive_monotonic_ns=time.monotonic_ns(),
            host_receive_unix_ns=time.time_ns(),
            q=self._state_vector(states, "q", 7),
            dq=self._state_vector(states, "dq", 7),
            tau=self._state_vector(states, "tau", 7),
            tau_des=self._state_vector(states, "tau_des", 7),
            tau_ext=self._state_vector(states, "tau_ext", 7),
            tau_interact=self._state_vector(states, "tau_interact", 7),
            tcp_pose_rdk=self._state_vector(states, "tcp_pose", 7),
            tcp_velocity=self._state_vector(states, "tcp_vel", 6),
            raw_ft=self._state_vector(states, "ft_sensor_raw", 6),
            external_wrench=self._state_vector(states, "ext_wrench_in_tcp", 6),
            temperature=self._state_vector(states, "temperature", 7),
            connection_generation=self._connection_generation,
        )

    def observe_both(self) -> DualArmSample:
        with self._arm_locks["left"], self._arm_locks["right"]:
            return DualArmSample(
                left=self.observe("left"), right=self.observe("right"))

    def _clear_fault_locked(self, side: str) -> bool:
        robot = self._robot(side)
        if not bool(self._invoke(robot, "fault")):
            return False
        cleared = self._invoke(robot, "clear_fault", "ClearFault")
        if not bool(cleared):
            raise RuntimeError(f"{side} Flexiv controller fault could not be cleared")
        self._safe_targets.pop(side, None)
        return True

    def clear_fault(self, side: str, *, local_console: bool) -> bool:
        """Clear a controller fault as an explicit local Reset operation."""

        self._guard.require("ClearFault", local_console=local_console)
        with self._arm_locks[side]:
            return self._clear_fault_locked(side)

    def enable(self, side: str, *, local_console: bool) -> None:
        self._guard.require("Enable", local_console=local_console)
        with self._arm_locks[side]:
            # A minor controller fault makes Enable/SwitchMode fail until RDK
            # ClearFault succeeds.  Reset owns this local write path, so make
            # Enable resilient to a fault that appeared after its initial
            # preflight instead of requiring a daemon restart.
            if bool(self._invoke(self._robot(side), "fault")):
                self._guard.require("ClearFault", local_console=local_console)
                self._clear_fault_locked(side)
            self._invoke(self._robot(side), "enable", "Enable")

    def operational(self, side: str) -> bool:
        with self._arm_locks[side]:
            return bool(self._invoke(self._robot(side), "operational", "Operational"))

    def _mode(self, name: str) -> Any:
        if self._rdk is None:
            raise RuntimeError("RDK is not connected")
        mode_type = getattr(self._rdk, "Mode", None)
        if mode_type is None or not hasattr(mode_type, name):
            raise RuntimeError(f"RDK 1.9 Mode.{name} is unavailable")
        return getattr(mode_type, name)

    def _switch_mode(self, side: str, name: str, *, local_console: bool) -> None:
        self._guard.require(f"SwitchMode({name})", local_console=local_console)
        with self._arm_locks[side]:
            self._invoke(self._robot(side), "switch_mode", "SwitchMode", args=(self._mode(name),))

    def switch_primitive_mode(self, side: str, *, local_console: bool) -> None:
        self._switch_mode(side, "NRT_PRIMITIVE_EXECUTION", local_console=local_console)

    def execute_zero_ft(self, side: str, *, local_console: bool) -> None:
        self._guard.require("ZeroFTSensor", local_console=local_console)
        with self._arm_locks[side]:
            self._invoke(
                self._robot(side),
                "execute_primitive",
                "ExecutePrimitive",
                args=("ZeroFTSensor", {}),
            )

    def primitive_state(self, side: str) -> Mapping[str, Any]:
        with self._arm_locks[side]:
            state = self._invoke(self._robot(side), "primitive_states", "PrimitiveStates")
        if not isinstance(state, Mapping):
            raise RuntimeError("RDK primitive_states did not return a mapping")
        return state

    def stop(self, side: str, *, local_console: bool) -> None:
        """Stop ongoing motion and let the controller hold in IDLE.

        A measured Cartesian target with a low acceleration limit is not an
        emergency/routine stop: the RDK trajectory generator can continue the
        previous velocity for seconds while converging to that target.  RDK
        ``Stop`` is the controller-native transition to IDLE and is therefore
        the correct boundary operation when a clutch or safety hold latches.
        """

        self._guard.require("Stop", local_console=local_console)
        with self._arm_locks[side]:
            self._invoke(self._robot(side), "stop", "Stop")
            self._safe_targets.pop(side, None)

    def switch_idle(self, side: str, *, local_console: bool) -> None:
        self._switch_mode(side, "IDLE", local_console=local_console)

    def switch_cartesian_mode(self, side: str, *, local_console: bool) -> None:
        # Flexiv retains the force-control-axis selection independently from
        # the pose/wrench target. A previous force-control session can leave
        # one or more axes ignoring Cartesian position after a mode switch.
        self._guard.require(
            "SwitchMode(NRT_CARTESIAN_MOTION_FORCE)",
            local_console=local_console,
        )
        self._guard.require("SetForceControlAxis", local_console=local_console)
        with self._arm_locks[side]:
            robot = self._robot(side)
            self._invoke(
                robot,
                "switch_mode",
                "SwitchMode",
                args=(self._mode("NRT_CARTESIAN_MOTION_FORCE"),),
            )
            self._invoke(
                robot,
                "set_force_control_axis",
                "SetForceControlAxis",
                args=([False] * 6,),
            )
            # A mode switch may restore controller-default Cartesian gains.
            self._cartesian_impedance.pop(side, None)

    def switch_joint_position_mode(self, side: str, *, local_console: bool) -> None:
        self._switch_mode(side, "NRT_JOINT_POSITION", local_console=local_console)

    def joint_position_limits(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        try:
            metadata = self._metadata[side]
        except KeyError as exc:
            raise RuntimeError(f"{side} RobotInfo metadata is unavailable") from exc
        return (
            np.asarray(metadata.q_min, dtype=np.float64),
            np.asarray(metadata.q_max, dtype=np.float64),
        )

    def nominal_cartesian_stiffness(self, side: str) -> np.ndarray:
        try:
            values = self._metadata[side].nominal_cartesian_stiffness
        except KeyError as exc:
            raise RuntimeError(f"{side} RobotInfo metadata is unavailable") from exc
        return np.asarray(values, dtype=np.float64)

    def set_cartesian_impedance(
        self,
        side: str,
        stiffness: np.ndarray,
        damping_ratio: np.ndarray,
        *,
        local_authorized: bool,
    ) -> None:
        """Apply one explicit RDK Cartesian impedance profile."""

        self._guard.require("SetCartesianImpedance", local_console=local_authorized)
        k_x = np.asarray(stiffness, dtype=np.float64).reshape(-1)
        z_x = np.asarray(damping_ratio, dtype=np.float64).reshape(-1)
        if k_x.shape != (6,) or not np.all(np.isfinite(k_x)):
            raise ValueError("Cartesian stiffness must be a finite 6-vector")
        if z_x.shape != (6,) or not np.all(np.isfinite(z_x)):
            raise ValueError("Cartesian damping ratio must be a finite 6-vector")
        nominal = self.nominal_cartesian_stiffness(side)
        if np.any(k_x < 0.0) or np.any(k_x > nominal):
            raise ValueError("Cartesian stiffness exceeds RobotInfo.K_x_nom")
        if np.any(z_x < 0.3) or np.any(z_x > 0.8):
            raise ValueError("Cartesian damping ratio must be in [0.3,0.8]")
        signature = (
            tuple(float(value) for value in k_x),
            tuple(float(value) for value in z_x),
        )
        with self._arm_locks[side]:
            if self._cartesian_impedance.get(side) == signature:
                return
            self._invoke(
                self._robot(side),
                "set_cartesian_impedance",
                "SetCartesianImpedance",
                args=(k_x.tolist(), z_x.tolist()),
            )
            self._cartesian_impedance[side] = signature

    def send_joint_position(
        self,
        side: str,
        positions: np.ndarray,
        *,
        max_velocity: float,
        max_acceleration: float,
        local_authorized: bool,
    ) -> None:
        """Send one smoothed RDK NRT joint-position target."""

        self._guard.require("SendJointPosition", local_console=local_authorized)
        target = np.asarray(positions, dtype=np.float64).reshape(-1)
        if target.shape != (7,) or not np.all(np.isfinite(target)):
            raise ValueError("joint target must be a finite 7-vector")
        lower, upper = self.joint_position_limits(side)
        if np.any(target < lower) or np.any(target > upper):
            raise ValueError("joint target exceeds RobotInfo position limits")
        if (
            not math.isfinite(max_velocity)
            or max_velocity <= 0.0
            or not math.isfinite(max_acceleration)
            or max_acceleration <= 0.0
        ):
            raise ValueError("joint velocity/acceleration limits must be positive")
        metadata = self._metadata[side]
        if max_velocity > min(metadata.dq_max):
            raise ValueError("joint max velocity exceeds RobotInfo.dq_max")
        with self._arm_locks[side]:
            self._invoke(
                self._robot(side),
                "send_joint_position",
                "SendJointPosition",
                args=(
                    target.tolist(),
                    [0.0] * 7,
                    [float(max_velocity)] * 7,
                    [float(max_acceleration)] * 7,
                ),
            )

    def rebase_from_measurement(self, side: str) -> np.ndarray:
        """Update the daemon's safe target without issuing an RDK command."""

        with self._arm_locks[side]:
            pose = self.observe(side).tcp_pose_rdk.copy()
            self._safe_targets[side] = pose
        return pose.copy()

    @staticmethod
    def _validated_pose(value: Any) -> np.ndarray:
        pose = np.asarray(value, dtype=np.float64).reshape(-1)
        if pose.shape != (7,) or not np.all(np.isfinite(pose)):
            raise ValueError("RDK pose must contain seven finite values")
        quaternion_norm = float(np.linalg.norm(pose[3:]))
        if abs(quaternion_norm - 1.0) > 1.0e-6:
            raise ValueError("RDK target quaternion must be normalized")
        return pose

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
        """Send one RDK 1.9 NRT Cartesian pure-motion target."""

        self._guard.require("SendCartesianMotionForce", local_console=local_authorized)
        pose = self._validated_pose(pose_rdk)
        limits = (
            max_linear_velocity,
            max_angular_velocity,
            max_linear_acceleration,
            max_angular_acceleration,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in limits):
            raise ValueError("Cartesian velocity/acceleration limits must be positive and finite")
        with self._arm_locks[side]:
            self._invoke(
                self._robot(side),
                "send_cartesian_motion_force",
                "SendCartesianMotionForce",
                args=(
                pose.tolist(),
                [0.0] * 6,
                [0.0] * 6,
                float(max_linear_velocity),
                float(max_angular_velocity),
                float(max_linear_acceleration),
                float(max_angular_acceleration),
            ),
            )
            self._safe_targets[side] = pose.copy()

    def send_hold_from_measurement(self, side: str, *, local_authorized: bool) -> np.ndarray:
        pose = self.rebase_from_measurement(side)
        self.send_cartesian_target(
            side,
            pose,
            max_linear_velocity=0.01,
            max_angular_velocity=0.05,
            max_linear_acceleration=0.05,
            max_angular_acceleration=0.1,
            local_authorized=local_authorized,
        )
        return pose
