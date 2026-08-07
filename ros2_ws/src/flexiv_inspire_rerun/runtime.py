"""ROS-independent Rerun logging primitives.

All methods consume plain dictionaries so the same code is usable by the ROS 2
subscriber and by the offline synthetic smoke test.  The module never publishes
commands and has no robot, Modbus, camera, or ROS side effects.
"""

from __future__ import annotations

import base64
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

ROT6D_FIRST_TWO_COLUMNS = "R00,R10,R20,R01,R11,R21"
ROT6D_IDENTITY = np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
TACTILE_ATLAS_SHAPE = (48, 58)


def quaternion_xyzw_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    """Return a proper rotation matrix from a ROS-order quaternion."""

    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        raise ValueError("quaternion must contain four finite xyzw values")
    norm = float(np.linalg.norm(q))
    if norm < 1.0e-12:
        raise ValueError("quaternion norm is too small")
    x, y, z, w = q / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_rotation6d(matrix: Sequence[Sequence[float]]) -> np.ndarray:
    rotation = np.asarray(matrix, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("rotation matrix must be finite and 3x3")
    return np.concatenate((rotation[:, 0], rotation[:, 1]))


def rotation6d_to_matrix(values: Sequence[float]) -> np.ndarray:
    """Decode Zhou Rotation-6D with strict degenerate-input rejection."""

    raw = np.asarray(values, dtype=np.float64)
    if raw.shape != (6,) or not np.all(np.isfinite(raw)):
        raise ValueError("Rotation-6D must contain six finite values")
    a1, a2 = raw[:3], raw[3:]
    n1 = float(np.linalg.norm(a1))
    if n1 < 1.0e-8:
        raise ValueError("Rotation-6D first column norm is too small")
    b1 = a1 / n1
    projected = a2 - float(np.dot(b1, a2)) * b1
    n2 = float(np.linalg.norm(projected))
    if n2 < 1.0e-8:
        raise ValueError("Rotation-6D columns are nearly collinear")
    b2 = projected / n2
    b3 = np.cross(b1, b2)
    rotation = np.column_stack((b1, b2, b3))
    determinant = float(np.linalg.det(rotation))
    if not np.all(np.isfinite(rotation)) or abs(determinant - 1.0) > 1.0e-6:
        raise ValueError("Rotation-6D orthogonalization did not produce SO(3)")
    return rotation


def tactile_atlas(
    surfaces: Sequence[Mapping[str, Any]], *, side: str = "left"
) -> np.ndarray:
    """Arrange the 17 DFTP surfaces as an anatomical palm-view atlas.

    Missing or malformed surfaces are rejected rather than silently filled.  No
    normalization is performed: the value in Rerun is the Modbus uint16 taxel.
    The left atlas is drawn as a palm facing the viewer (little finger on the
    left, thumb on the right); the right atlas is its horizontal mirror.
    """

    if side not in {"left", "right"}:
        raise ValueError("tactile side must be left or right")

    by_name = {str(surface["name"]): surface for surface in surfaces}
    required = [
        *(
            f"{finger}_{suffix}"
            for finger in ("little", "ring", "middle", "index")
            for suffix in ("end", "tip", "pad")
        ),
        "thumb_end",
        "thumb_tip",
        "thumb_middle",
        "thumb_pad",
        "palm",
    ]
    missing = [name for name in required if name not in by_name]
    if missing:
        raise ValueError(f"missing tactile surfaces: {','.join(missing)}")

    canvas = np.zeros(TACTILE_ATLAS_SHAPE, dtype=np.uint16)

    def image_for(name: str, *, palm: bool = False) -> np.ndarray:
        surface = by_name[name]
        rows = int(surface["rows"])
        columns = int(surface["columns"])
        taxels = np.asarray(surface["taxels"], dtype=np.uint16)
        if taxels.size != rows * columns:
            raise ValueError(
                f"{name}: expected {rows * columns} taxels, got {taxels.size}"
            )
        return taxels.reshape(rows, columns, order="F" if palm else "C")

    # Four upright finger columns: distal end, fingertip and finger pad.
    for x, finger in zip((2, 13, 24, 35), ("little", "ring", "middle", "index")):
        end = image_for(f"{finger}_end")
        tip = image_for(f"{finger}_tip")
        pad = image_for(f"{finger}_pad")
        canvas[1 : 1 + end.shape[0], x + 2 : x + 2 + end.shape[1]] = end
        canvas[5 : 5 + tip.shape[0], x : x + tip.shape[1]] = tip
        canvas[18 : 18 + pad.shape[0], x : x + pad.shape[1]] = pad

    # Thumb surfaces follow a diagonal from the outer tip towards the palm.
    thumb_layout = {
        "end": (8, 52),
        "tip": (12, 47),
        "middle": (25, 45),
        "pad": (29, 40),
    }
    for suffix, (y, x) in thumb_layout.items():
        image = image_for(f"thumb_{suffix}")
        canvas[y : y + image.shape[0], x : x + image.shape[1]] = image

    palm_image = image_for("palm", palm=True)
    canvas[36 : 36 + palm_image.shape[0], 17 : 17 + palm_image.shape[1]] = palm_image
    return canvas if side == "left" else np.fliplr(canvas).copy()


def tactile_sensor_mask(
    surfaces: Sequence[Mapping[str, Any]], *, side: str = "left"
) -> np.ndarray:
    """Return the anatomical sensor footprint, including zero-value taxels."""

    footprint_surfaces = []
    for surface in surfaces:
        rows = int(surface["rows"])
        columns = int(surface["columns"])
        footprint_surfaces.append(
            {
                **surface,
                "taxels": np.ones(rows * columns, dtype=np.uint16),
            }
        )
    return tactile_atlas(footprint_surfaces, side=side).astype(bool)


def tactile_heatmap(
    atlas: np.ndarray, sensor_mask: np.ndarray | None = None
) -> np.ndarray:
    """Make an RGB palm view with a visible zero baseline and outline.

    The exact raw atlas is still logged separately. Typical tactile values use
    only a small part of uint16, so drawing them directly can look completely
    black even while the sensor is updating. Zero-value taxels are dark blue,
    the sensor boundary is grey-blue, and active taxels use blue-to-red color.
    """

    raw = np.asarray(atlas)
    if raw.shape != TACTILE_ATLAS_SHAPE or raw.dtype != np.uint16:
        raise ValueError("tactile atlas must be a uint16 palm image")
    mask = raw > 0 if sensor_mask is None else np.asarray(sensor_mask, dtype=bool)
    if mask.shape != raw.shape:
        raise ValueError("tactile sensor mask must match the atlas shape")

    rgb = np.zeros((*raw.shape, 3), dtype=np.uint8)
    rgb[mask] = (24, 58, 96)
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    interior = (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    rgb[mask & ~interior] = (105, 135, 165)

    active_mask = mask & (raw > 0)
    active = raw[active_mask].astype(np.float32)
    if active.size == 0:
        return rgb
    low = float(np.percentile(active, 2.0))
    high = float(np.percentile(active, 99.0))
    if high <= low:
        high = low + 1.0
    level = np.clip(
        (raw[active_mask].astype(np.float32) - low) / (high - low),
        0.0,
        1.0,
    )
    low_color = np.asarray((0.0, 145.0, 255.0), dtype=np.float32)
    mid_color = np.asarray((60.0, 235.0, 90.0), dtype=np.float32)
    high_color = np.asarray((255.0, 55.0, 0.0), dtype=np.float32)
    colors = np.empty((level.size, 3), dtype=np.float32)
    lower = level <= 0.5
    colors[lower] = low_color + (mid_color - low_color) * (level[lower, None] * 2.0)
    colors[~lower] = mid_color + (high_color - mid_color) * (
        (level[~lower, None] - 0.5) * 2.0
    )
    rgb[active_mask] = np.clip(colors, 0.0, 255.0).astype(np.uint8)
    return rgb


def command_action_vector(command: Mapping[str, Any]) -> np.ndarray:
    """Return the first command point in the public policy action layout.

    Rotation-6D Cartesian commands return exactly 30 values:
    left xyz+rot6d, right xyz+rot6d, left hand, right hand.
    Quaternion Cartesian commands are converted to the same layout. Joint-space
    commands remain variable length and are clearly identified by the caller.
    """

    trajectory = command.get("trajectory") or ()
    if not trajectory:
        raise ValueError("command has no trajectory point")
    point = trajectory[0]
    representation = int(command.get("representation", 1))
    left_hand = np.asarray(point.get("left_hand_targets", ()), dtype=np.float64)
    right_hand = np.asarray(point.get("right_hand_targets", ()), dtype=np.float64)
    if left_hand.shape != (6,) or right_hand.shape != (6,):
        raise ValueError("hand targets must contain six values per side")

    if representation == 1:
        left_rotation = np.asarray(
            point.get("left_delta_rotation6d", ()), dtype=np.float64
        )
        right_rotation = np.asarray(
            point.get("right_delta_rotation6d", ()), dtype=np.float64
        )
    elif representation == 2:
        left_rotation = matrix_to_rotation6d(
            quaternion_xyzw_to_matrix(point.get("left_delta_quaternion_xyzw", ()))
        )
        right_rotation = matrix_to_rotation6d(
            quaternion_xyzw_to_matrix(point.get("right_delta_quaternion_xyzw", ()))
        )
    elif representation == 3:
        left_joints = np.asarray(
            point.get("left_arm_joint_positions", ()), dtype=np.float64
        )
        right_joints = np.asarray(
            point.get("right_arm_joint_positions", ()), dtype=np.float64
        )
        result = np.concatenate((left_joints, right_joints, left_hand, right_hand))
        if not np.all(np.isfinite(result)):
            raise ValueError("joint command contains NaN or Inf")
        return result
    else:
        raise ValueError(f"unsupported control representation {representation}")

    left_xyz = np.asarray(point.get("left_delta_xyz", ()), dtype=np.float64)
    right_xyz = np.asarray(point.get("right_delta_xyz", ()), dtype=np.float64)
    if left_xyz.shape != (3,) or right_xyz.shape != (3,):
        raise ValueError("Cartesian translation must contain three values per side")
    if left_rotation.shape != (6,) or right_rotation.shape != (6,):
        raise ValueError("Cartesian Rotation-6D must contain six values per side")
    result = np.concatenate(
        (left_xyz, left_rotation, right_xyz, right_rotation, left_hand, right_hand)
    )
    if result.shape != (30,) or not np.all(np.isfinite(result)):
        raise ValueError("Cartesian action is not a finite 30-vector")
    return result


@dataclass(frozen=True)
class DispatcherStats:
    submitted: int
    processed: int
    dropped: int
    failed: int


class LatestOnlyDispatcher:
    """One background consumer with one pending item per stream key.

    A ROS callback only takes a short lock and replaces an old pending sample.
    Slow rendering therefore drops old observations instead of blocking the
    executor or accumulating latency.
    """

    def __init__(self, *, name: str = "rerun-latest-only") -> None:
        self._condition = threading.Condition()
        self._pending: OrderedDict[str, tuple[Callable[[Any], None], Any]] = (
            OrderedDict()
        )
        self._closed = False
        self._submitted = 0
        self._processed = 0
        self._dropped = 0
        self._failed = 0
        self._last_error = ""
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def submit(self, key: str, function: Callable[[Any], None], payload: Any) -> bool:
        with self._condition:
            if self._closed:
                return False
            self._submitted += 1
            if key in self._pending:
                self._dropped += 1
                self._pending.pop(key)
            self._pending[key] = (function, payload)
            self._condition.notify()
            return True

    @property
    def stats(self) -> DispatcherStats:
        with self._condition:
            return DispatcherStats(
                submitted=self._submitted,
                processed=self._processed,
                dropped=self._dropped,
                failed=self._failed,
            )

    @property
    def last_error(self) -> str:
        with self._condition:
            return self._last_error

    def close(self, *, drain: bool = True, timeout_s: float = 5.0) -> None:
        with self._condition:
            self._closed = True
            if not drain:
                self._dropped += len(self._pending)
                self._pending.clear()
            self._condition.notify_all()
        self._thread.join(timeout=timeout_s)
        if self._thread.is_alive():
            raise TimeoutError("Rerun dispatcher did not stop")

    def _run(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._pending or self._closed)
                if not self._pending:
                    if self._closed:
                        return
                    continue
                _, (function, payload) = self._pending.popitem(last=False)
            try:
                function(payload)
            except Exception as exc:  # visualization failure must never stop control
                with self._condition:
                    self._failed += 1
                    self._last_error = f"{type(exc).__name__}: {exc}"
            else:
                with self._condition:
                    self._processed += 1


class RerunVisualizer:
    """Version-0.33-compatible logging facade."""

    def __init__(
        self,
        *,
        save_path: str | Path | None = None,
        connect_url: str | None = None,
        spawn: bool = False,
        viewer_port: int = 9876,
        application_id: str = "isaac_teleop_flexiv_inspire",
        recording_id: str | None = None,
    ) -> None:
        try:
            import rerun as rr
        except ImportError as exc:  # pragma: no cover - exercised by deployment
            raise RuntimeError(
                "rerun-sdk==0.33.1 is required in envs/ros-py312"
            ) from exc
        selected = sum((save_path is not None, connect_url is not None, bool(spawn)))
        if selected != 1:
            raise ValueError("select exactly one Rerun sink: save, connect, or spawn")
        self.rr = rr
        self.stream = rr.RecordingStream(application_id, recording_id=recording_id)
        self.save_path = Path(save_path).expanduser().resolve() if save_path else None
        if self.save_path is not None:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            self.stream.save(self.save_path)
        elif connect_url is not None:
            self.stream.connect_grpc(connect_url)
        else:
            self.stream.spawn(
                port=int(viewer_port), connect=True, hide_welcome_screen=True
            )
        self._send_default_blueprint()
        self._series_configured: set[str] = set()
        self._last_text: dict[str, str] = {}
        self._closed = False
        self._log_static_metadata()

    def _send_default_blueprint(self) -> None:
        """Install a deterministic live layout instead of auto-view clutter."""

        import rerun.blueprint as rrb

        tactile = rrb.Horizontal(
            rrb.Spatial2DView(
                origin="/robot/left_hand/tactile",
                contents="/robot/left_hand/tactile/atlas_heatmap_u8",
                name="Left hand tactile (palm view)",
            ),
            rrb.Spatial2DView(
                origin="/robot/right_hand/tactile",
                contents="/robot/right_hand/tactile/atlas_heatmap_u8",
                name="Right hand tactile (palm view)",
            ),
            column_shares=[1.0, 1.0],
            name="Bimanual tactile",
        )
        cameras = rrb.Vertical(
            rrb.Horizontal(
                *(
                    rrb.Spatial2DView(
                        origin=f"/camera/{camera}",
                        contents=f"/camera/{camera}/color",
                        name=f"{camera} RGB",
                    )
                    for camera in ("head", "left_wrist", "right_wrist")
                )
            ),
            rrb.Horizontal(
                *(
                    rrb.Spatial2DView(
                        origin=f"/camera/{camera}",
                        contents=f"/camera/{camera}/depth_m",
                        name=f"{camera} depth",
                    )
                    for camera in ("head", "left_wrist", "right_wrist")
                )
            ),
            name="RGB + depth",
        )
        pointclouds = rrb.Horizontal(
            *(
                rrb.Spatial3DView(
                    origin=f"/camera/{camera}",
                    contents=f"/camera/{camera}/pointcloud",
                    name=f"{camera} point cloud",
                )
                for camera in ("head", "left_wrist", "right_wrist")
            ),
            name="Point clouds",
        )
        arms = rrb.Horizontal(
            *(
                rrb.TimeSeriesView(
                    origin=f"/robot/{side}_arm",
                    contents=[
                        f"/robot/{side}_arm/q",
                        f"/robot/{side}_arm/dq",
                        f"/robot/{side}_arm/external_wrench",
                    ],
                    name=f"{side} arm",
                )
                for side in ("left", "right")
            ),
            name="Arm state",
        )
        hands = rrb.Horizontal(
            *(
                rrb.TimeSeriesView(
                    origin=f"/robot/{side}_hand",
                    contents=[
                        f"/robot/{side}_hand/position",
                        f"/robot/{side}_hand/manus_ergonomics_rad",
                        f"/robot/{side}_hand/actual_force",
                        f"/robot/{side}_hand/current",
                    ],
                    name=f"{side} hand",
                )
                for side in ("left", "right")
            ),
            name="Hand state",
        )
        control = rrb.TimeSeriesView(
            origin="/control",
            contents=[
                "/control/requested/action",
                "/control/safe/action",
                "/control/sent/action",
                "/control/state/id",
                "/control/state/physical_pedal",
            ],
            name="Control",
        )
        # Keep every live modality visible in one deterministic dashboard.
        # Nested containers preserve usable camera sizes while avoiding the
        # six-tab workflow that required the operator to keep switching views.
        overview = rrb.Vertical(
            cameras,
            rrb.Horizontal(
                tactile,
                pointclouds,
                column_shares=[2.0, 3.0],
                name="Contact + geometry",
            ),
            rrb.Horizontal(
                arms,
                hands,
                control,
                column_shares=[2.0, 2.0, 1.0],
                name="Robot + control",
            ),
            row_shares=[2.4, 1.2, 1.4],
            name="Live overview",
        )
        self.stream.send_blueprint(
            rrb.Blueprint(
                overview,
                auto_views=False,
                collapse_panels=True,
            ),
            make_active=True,
            make_default=True,
        )

    def _log_static_metadata(self) -> None:
        self.stream.log(
            "metadata/rotation6d_order",
            self.rr.TextDocument(ROT6D_FIRST_TWO_COLUMNS),
            static=True,
        )
        self.stream.log(
            "metadata/tactile_units",
            self.rr.TextDocument("raw Modbus uint16; no normalization"),
            static=True,
        )
        self.stream.log(
            "metadata/mode",
            self.rr.TextDocument("read-only visualization; no command publishers"),
            static=True,
        )

    def close(self) -> None:
        if self._closed:
            return
        errors: list[str] = []
        try:
            self.stream.flush(timeout_sec=10.0)
        except Exception as exc:
            errors.append(f"flush: {exc}")
        try:
            self.stream.disconnect()
        except Exception as exc:
            errors.append(f"disconnect: {exc}")
        finally:
            self._closed = True
        if errors:
            raise RuntimeError("; ".join(errors))

    def __enter__(self) -> "RerunVisualizer":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _set_time(
        self,
        stamp_ns: int,
        *,
        sequence: int | None = None,
        mapped_host_ns: int | None = None,
    ) -> None:
        self.stream.reset_time()
        self.stream.set_time("ros_time", timestamp=np.datetime64(stamp_ns, "ns"))
        if sequence is not None:
            self.stream.set_time("source_sequence", sequence=int(sequence))
        if mapped_host_ns is not None and mapped_host_ns > 0:
            self.stream.set_time(
                "host_monotonic",
                timestamp=np.datetime64(mapped_host_ns, "ns"),
            )

    def _vector(
        self, path: str, values: Sequence[float], labels: Sequence[str]
    ) -> None:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if array.size != len(labels):
            raise ValueError(f"{path}: {array.size} values but {len(labels)} labels")
        if path not in self._series_configured:
            self.stream.log(path, self.rr.SeriesLines(names=list(labels)), static=True)
            self._series_configured.add(path)
        self.stream.log(path, self.rr.Scalars(array))

    @staticmethod
    def _semantic_vector_labels(path: str, size: int) -> list[str]:
        """Return stable, unique legend labels for live and offline curves."""

        field = path.rstrip("/").rsplit("/", 1)[-1]
        if field in {"q", "dq", "tau", "tau_des", "tau_ext", "tau_interact"}:
            if size == 7:
                return [f"{field}_j{index + 1}" for index in range(size)]
        if field in {"external_wrench", "tcp_wrench", "raw_ft"} and size == 6:
            prefix = (
                "wrench" if field in {"external_wrench", "tcp_wrench"} else "raw_ft"
            )
            return [f"{prefix}_{axis}" for axis in ("fx", "fy", "fz", "tx", "ty", "tz")]
        if field in {"tcp_twist", "tcp_velocity"} and size == 6:
            axes = ("vx", "vy", "vz", "wx", "wy", "wz")
            return [f"tcp_twist_{axis}" for axis in axes]
        hand_fields = {
            "angle": "angle",
            "angle_rad": "angle",
            "position": "position",
            "actual_force": "force",
            "current": "current",
            "temperature": "temperature",
            "temperature_c": "temperature",
            "error": "error",
            "error_code": "error",
            "status": "status",
            "status_code": "status",
        }
        if "_hand/" in path and field in hand_fields and size == 6:
            fingers = (
                "little",
                "ring",
                "middle",
                "index",
                "thumb_bend",
                "thumb_rotate",
            )
            return [f"{hand_fields[field]}_{finger}" for finger in fingers]
        if field in {"rotation6d", "tcp_pose_rotation6d"} and size == 6:
            return [f"rotation6d_{name}" for name in ROT6D_FIRST_TWO_COLUMNS.split(",")]
        return [f"{field}_{index}" for index in range(size)]

    def _scalar(self, path: str, value: float | int | bool) -> None:
        self.stream.log(path, self.rr.Scalars(float(value)))

    def _text_if_changed(self, path: str, value: str, *, level: str = "INFO") -> None:
        if self._last_text.get(path) == value:
            return
        self._last_text[path] = value
        self.stream.log(path, self.rr.TextLog(value, level=level))

    def _log_acquisition(self, path: str, acquisition: Mapping[str, Any]) -> None:
        valid = bool(acquisition.get("valid", False))
        timing_valid = bool(acquisition.get("timing_valid", False))
        self._scalar(f"{path}/valid", valid)
        self._scalar(f"{path}/timing_valid", timing_valid)
        self._scalar(
            f"{path}/age_seconds", float(acquisition.get("age_ns", 0)) * 1.0e-9
        )
        self._scalar(
            f"{path}/acquisition_duration_seconds",
            max(
                0,
                int(acquisition.get("acquisition_end_ns", 0))
                - int(acquisition.get("acquisition_start_ns", 0)),
            )
            * 1.0e-9,
        )
        self.stream.log(
            f"{path}/timestamps",
            self.rr.AnyValues(
                source_time_ns=np.int64(acquisition.get("source_time_ns", 0)),
                host_receive_time_ns=np.int64(
                    acquisition.get("host_receive_time_ns", 0)
                ),
                acquisition_start_ns=np.int64(
                    acquisition.get("acquisition_start_ns", 0)
                ),
                acquisition_end_ns=np.int64(acquisition.get("acquisition_end_ns", 0)),
                mapped_host_time_ns=np.int64(acquisition.get("mapped_host_time_ns", 0)),
                source_sequence=np.uint64(acquisition.get("sequence", 0)),
                valid=valid,
                timing_valid=timing_valid,
                invalid_reason=str(acquisition.get("invalid_reason", "")),
            ),
        )
        self._scalar(
            f"{path}/source_time_seconds",
            int(acquisition.get("source_time_ns", 0)) * 1.0e-9,
        )
        self._scalar(
            f"{path}/host_receive_time_seconds",
            int(acquisition.get("host_receive_time_ns", 0)) * 1.0e-9,
        )
        self._scalar(
            f"{path}/mapped_host_time_seconds",
            int(acquisition.get("mapped_host_time_ns", 0)) * 1.0e-9,
        )
        self._scalar(f"{path}/source_sequence", int(acquisition.get("sequence", 0)))
        self._text_if_changed(
            f"{path}/source_clock_domain",
            str(acquisition.get("source_clock_domain", "")),
        )
        self._text_if_changed(
            f"{path}/host_clock_domain",
            str(acquisition.get("host_clock_domain", "")),
        )
        reason = str(acquisition.get("invalid_reason", ""))
        if not reason and (not valid or not timing_valid):
            reason = "invalid without reason"
        self._text_if_changed(f"{path}/invalid_reason", reason, level="WARN")

    def log_camera(self, payload: Mapping[str, Any]) -> None:
        camera = str(payload["camera"])
        stamp_ns = int(payload["stamp_ns"])
        acquisition = payload.get("acquisition")
        sequence = (
            acquisition.get("sequence")
            if acquisition is not None
            else payload.get("sequence")
        )
        mapped_host_ns = (
            int(acquisition.get("mapped_host_time_ns", 0))
            if acquisition is not None
            else None
        )
        self._set_time(
            stamp_ns,
            sequence=int(sequence) if sequence is not None else None,
            mapped_host_ns=mapped_host_ns,
        )
        jpeg = bytes(payload["jpeg"])
        if not jpeg:
            raise ValueError(f"{camera}: empty compressed image")
        media_type = "image/jpeg"
        image_format = str(payload.get("format", "")).lower()
        if "png" in image_format:
            media_type = "image/png"
        self.stream.log(
            f"camera/{camera}/color",
            self.rr.EncodedImage(contents=jpeg, media_type=media_type),
        )
        if payload.get("width") is not None:
            self._scalar(f"camera/{camera}/width", int(payload["width"]))
        if payload.get("height") is not None:
            self._scalar(f"camera/{camera}/height", int(payload["height"]))
        self._scalar(f"camera/{camera}/header_stamp_valid", stamp_ns > 0)
        if acquisition is not None:
            self._log_acquisition(f"camera/{camera}/acquisition", acquisition)
        elif bool(payload.get("timing_unpaired", False)):
            self._scalar(f"camera/{camera}/acquisition/valid", False)
            self._scalar(f"camera/{camera}/acquisition/timing_valid", False)
            self._text_if_changed(
                f"camera/{camera}/acquisition/invalid_reason",
                "legacy compressed image has no atomically paired AcquisitionInfo",
                level="WARN",
            )

    def log_depth(self, payload: Mapping[str, Any]) -> None:
        """Display the live ROS Z16 image on its own latest-only stream."""

        camera = str(payload["camera"])
        width = int(payload["width"])
        height = int(payload["height"])
        step = int(payload["step"])
        if str(payload["encoding"]).lower() not in {"16uc1", "mono16"}:
            raise ValueError(f"{camera}: unsupported live depth encoding")
        if width <= 0 or height <= 0 or step < width * 2:
            raise ValueError(f"{camera}: invalid live depth layout")
        raw = bytes(payload["data"])
        if len(raw) < step * height:
            raise ValueError(f"{camera}: truncated live depth image")
        dtype = np.dtype(">u2" if bool(payload.get("is_bigendian", False)) else "<u2")
        depth = np.ndarray(
            (height, width), dtype=dtype, buffer=raw, strides=(step, 2)
        ).astype(np.float32)
        self._set_time(int(payload["stamp_ns"]))
        self.stream.log(
            f"camera/{camera}/depth_m",
            self.rr.DepthImage(
                depth * float(payload.get("meter_per_unit", 0.001)), meter=1.0
            ),
        )

    def log_pointcloud(self, payload: Mapping[str, Any]) -> None:
        """Decode the XYZ fields of the live PointCloud2 without ROS helpers."""

        camera = str(payload["camera"])
        width = int(payload["width"])
        height = int(payload["height"])
        point_step = int(payload["point_step"])
        row_step = int(payload["row_step"])
        if (
            width <= 0
            or height <= 0
            or point_step < 12
            or row_step < point_step * width
        ):
            raise ValueError(f"{camera}: invalid live point-cloud layout")
        fields = payload.get("fields", {})
        if not isinstance(fields, Mapping):
            raise ValueError(f"{camera}: point-cloud fields are invalid")
        offsets: list[int] = []
        for axis in ("x", "y", "z"):
            field = fields.get(axis)
            if not isinstance(field, Mapping):
                raise ValueError(f"{camera}: point-cloud is missing {axis}")
            # sensor_msgs/PointField.FLOAT32 == 7.
            if int(field.get("datatype", 0)) != 7 or int(field.get("count", 0)) != 1:
                raise ValueError(f"{camera}: {axis} must be one float32 field")
            offset = int(field.get("offset", -1))
            if offset < 0 or offset + 4 > point_step:
                raise ValueError(f"{camera}: {axis} offset is invalid")
            offsets.append(offset)
        raw = bytes(payload["data"])
        if len(raw) < row_step * height:
            raise ValueError(f"{camera}: truncated live point cloud")
        dtype = np.dtype(">f4" if bool(payload.get("is_bigendian", False)) else "<f4")
        points = np.empty((height, width, 3), dtype=np.float32)
        for index, offset in enumerate(offsets):
            points[..., index] = np.ndarray(
                (height, width),
                dtype=dtype,
                buffer=raw,
                offset=offset,
                strides=(row_step, point_step),
            )
        points = points.reshape(-1, 3)
        points = points[np.all(np.isfinite(points), axis=1)]
        self._set_time(int(payload["stamp_ns"]))
        self.stream.log(f"camera/{camera}/pointcloud", self.rr.Points3D(points))
        self._text_if_changed(
            f"camera/{camera}/pointcloud_frame",
            str(payload.get("frame_id", "")),
        )

    def log_camera_acquisition(self, payload: Mapping[str, Any]) -> None:
        camera = str(payload["camera"])
        acquisition = payload["acquisition"]
        self._set_time(
            int(payload["stamp_ns"]),
            sequence=int(acquisition.get("sequence", 0)),
            mapped_host_ns=int(acquisition.get("mapped_host_time_ns", 0)),
        )
        self._log_acquisition(f"camera/{camera}/acquisition", acquisition)

    def log_arm(self, payload: Mapping[str, Any]) -> None:
        side = str(payload["side"])
        acquisition = payload["acquisition"]
        self._set_time(
            int(payload["stamp_ns"]),
            sequence=int(acquisition.get("sequence", 0)),
            mapped_host_ns=int(acquisition.get("mapped_host_time_ns", 0)),
        )
        root = f"robot/{side}_arm"
        for name in ("q", "dq", "tau", "tau_des", "tau_ext", "tau_interact"):
            path = f"{root}/{name}"
            self._vector(path, payload[name], self._semantic_vector_labels(path, 7))
        temperature = payload.get("temperature", ())
        if temperature:
            self._vector(
                f"{root}/temperature_c",
                temperature,
                [f"temperature_sensor_{index}" for index in range(len(temperature))],
            )
        pose = payload["tcp_pose"]
        translation = np.asarray(pose["position"], dtype=np.float64)
        quaternion = np.asarray(pose["quaternion_xyzw"], dtype=np.float64)
        rotation = quaternion_xyzw_to_matrix(quaternion)
        rotation6d = matrix_to_rotation6d(rotation)
        self.stream.log(
            f"{root}/tcp",
            self.rr.Transform3D(
                translation=translation,
                quaternion=self.rr.Quaternion(xyzw=quaternion),
            ),
        )
        self._vector(
            f"{root}/tcp/rotation6d",
            rotation6d,
            [f"rotation6d_{name}" for name in ROT6D_FIRST_TWO_COLUMNS.split(",")],
        )
        self.stream.log(
            f"{root}/tcp/decoded_rotation",
            self.rr.Transform3D(mat3x3=rotation),
        )
        self._vector(
            f"{root}/tcp_twist",
            payload["tcp_twist"],
            (
                "tcp_twist_vx",
                "tcp_twist_vy",
                "tcp_twist_vz",
                "tcp_twist_wx",
                "tcp_twist_wy",
                "tcp_twist_wz",
            ),
        )
        self._vector(
            f"{root}/raw_ft",
            payload["raw_ft"],
            (
                "raw_ft_fx",
                "raw_ft_fy",
                "raw_ft_fz",
                "raw_ft_tx",
                "raw_ft_ty",
                "raw_ft_tz",
            ),
        )
        self._vector(
            f"{root}/external_wrench",
            payload["tcp_wrench"],
            (
                "wrench_fx",
                "wrench_fy",
                "wrench_fz",
                "wrench_tx",
                "wrench_ty",
                "wrench_tz",
            ),
        )
        self._scalar(f"{root}/connected", bool(payload.get("connected", False)))
        self._scalar(
            f"{root}/rdk_connection_generation", int(payload.get("generation", 0))
        )
        fault = str(payload.get("fault", ""))
        self._text_if_changed(f"{root}/fault", fault, level="ERROR")
        self._log_acquisition(f"{root}/acquisition", acquisition)

    def log_hand(self, payload: Mapping[str, Any]) -> None:
        side = str(payload["side"])
        acquisition = payload["acquisition"]
        self._set_time(
            int(payload["stamp_ns"]),
            sequence=int(payload.get("sequence", acquisition.get("sequence", 0))),
            mapped_host_ns=int(acquisition.get("mapped_host_time_ns", 0)),
        )
        root = f"robot/{side}_hand"
        for field, suffix in (
            ("angle", "angle_rad"),
            ("position", "position"),
            ("actual_force", "actual_force"),
            ("current", "current"),
            ("temperature", "temperature_c"),
            ("error", "error_code"),
            ("status", "status_code"),
        ):
            path = f"{root}/{suffix}"
            self._vector(path, payload[field], self._semantic_vector_labels(path, 6))
        self._scalar(f"{root}/connected", bool(payload.get("connected", False)))
        self._scalar(f"{root}/faulted", bool(payload.get("fault", False)))
        reason = str(payload.get("fault_reason", ""))
        self._text_if_changed(f"{root}/fault_reason", reason, level="ERROR")
        self._log_acquisition(f"{root}/acquisition", acquisition)

    def log_manus_ergonomics(self, payload: Mapping[str, Any]) -> None:
        side = str(payload["side"])
        names = [str(name) for name in payload["names"]]
        values = np.asarray(payload["values_rad"], dtype=np.float64)
        if side not in {"left", "right"}:
            raise ValueError(f"invalid MANUS side: {side}")
        if values.shape != (len(names),) or len(set(names)) != len(names):
            raise ValueError("invalid MANUS Ergonomics vector")
        self._set_time(int(payload["stamp_ns"]))
        self._vector(
            f"robot/{side}_hand/manus_ergonomics_rad",
            values,
            [f"manus_{name}" for name in names],
        )

    def log_tactile(self, payload: Mapping[str, Any]) -> None:
        side = str(payload["side"])
        acquisition = payload["acquisition"]
        self._set_time(
            int(payload["stamp_ns"]),
            sequence=int(payload.get("sequence", acquisition.get("sequence", 0))),
            mapped_host_ns=int(acquisition.get("mapped_host_time_ns", 0)),
        )
        root = f"robot/{side}_hand/tactile"
        surfaces = payload["surfaces"]
        atlas = tactile_atlas(surfaces, side=side)
        sensor_mask = tactile_sensor_mask(surfaces, side=side)
        self.stream.log(f"{root}/atlas_raw_u16", self.rr.Image(atlas))
        self.stream.log(
            f"{root}/atlas_heatmap_u8",
            self.rr.Image(tactile_heatmap(atlas, sensor_mask)),
        )
        flattened: list[int] = []
        for surface in surfaces:
            rows = int(surface["rows"])
            columns = int(surface["columns"])
            taxels = np.asarray(surface["taxels"], dtype=np.uint16)
            if taxels.size != rows * columns:
                raise ValueError(f"{surface['name']}: tactile shape mismatch")
            name = str(surface["name"])
            surface_acquisition = surface.get("acquisition")
            if surface_acquisition is None:
                raise ValueError(f"{name}: missing per-surface AcquisitionInfo")
            flattened.extend(int(value) for value in taxels)
        if len(flattened) != int(payload.get("taxel_count", len(flattened))):
            raise ValueError("tactile taxel_count does not match surfaces")
        self._scalar(f"{root}/taxel_count", len(flattened))
        self._scalar(f"{root}/raw_min", min(flattened) if flattened else 0)
        self._scalar(f"{root}/raw_max", max(flattened) if flattened else 0)
        self._scalar(
            f"{root}/raw_mean", float(np.mean(flattened)) if flattened else 0.0
        )
        self._scalar(f"{root}/frame_valid", bool(payload.get("valid", False)))
        frame_reason = str(payload.get("invalid_reason", ""))
        self._text_if_changed(f"{root}/invalid_reason", frame_reason, level="WARN")
        self._log_acquisition(f"{root}/acquisition", acquisition)

    def log_command(self, stage: str, command: Mapping[str, Any]) -> np.ndarray:
        if stage not in {"requested", "safe", "sent"}:
            raise ValueError(f"unknown control stage {stage}")
        self._set_time(
            int(command["stamp_ns"]), sequence=int(command.get("sequence", 0))
        )
        root = f"control/{stage}"
        action = command_action_vector(command)
        labels = (
            [f"left_dxyz_{axis}" for axis in "xyz"]
            + [f"left_rot6d_{index}" for index in range(6)]
            + [f"right_dxyz_{axis}" for axis in "xyz"]
            + [f"right_rot6d_{index}" for index in range(6)]
            + [f"left_hand_{index}" for index in range(6)]
            + [f"right_hand_{index}" for index in range(6)]
        )
        if action.size == 30:
            self._vector(f"{root}/action", action, labels)
        else:
            self._vector(
                f"{root}/joint_action",
                action,
                [f"value_{index}" for index in range(action.size)],
            )
        self._scalar(f"{root}/trajectory_points", len(command.get("trajectory", ())))
        self._scalar(f"{root}/deadman", bool(command.get("deadman", False)))
        self._scalar(f"{root}/valid_mask", int(command.get("valid_mask", 0)))
        self._scalar(f"{root}/ttl_seconds", int(command.get("ttl_ns", 0)) * 1.0e-9)
        self._text_if_changed(f"{root}/source", str(command.get("source", "")))
        self._text_if_changed(
            f"{root}/representation",
            str(command.get("representation_name", command.get("representation", ""))),
        )
        if action.size >= 18 and int(command.get("representation", 1)) in (1, 2):
            for side, start in (("left", 3), ("right", 12)):
                rotation = rotation6d_to_matrix(action[start : start + 6])
                self.stream.log(
                    f"{root}/{side}_decoded_delta_rotation",
                    self.rr.Transform3D(mat3x3=rotation),
                )
        return action

    def log_trace(self, payload: Mapping[str, Any]) -> None:
        stamp_ns = int(payload["stamp_ns"])
        self._set_time(stamp_ns, sequence=int(payload.get("trace_sequence", 0)))
        actions: dict[str, np.ndarray] = {}
        for stage in ("requested", "safe", "sent"):
            command = payload.get(stage)
            valid = bool(payload.get(f"{stage}_valid", False))
            self._scalar(f"control/trace/{stage}_valid", valid)
            if command and valid:
                command = dict(command)
                command["stamp_ns"] = stamp_ns
                try:
                    actions[stage] = self.log_command(stage, command)
                except ValueError as exc:
                    self._text_if_changed(
                        f"control/trace/{stage}_decode_error", str(exc), level="WARN"
                    )
        self._set_time(stamp_ns, sequence=int(payload.get("trace_sequence", 0)))
        comparisons = (
            ("safe_minus_requested", "safe", "requested"),
            ("sent_minus_safe", "sent", "safe"),
            ("sent_minus_requested", "sent", "requested"),
        )
        for name, left, right in comparisons:
            if (
                left in actions
                and right in actions
                and actions[left].shape == actions[right].shape
            ):
                difference = actions[left] - actions[right]
                self._vector(
                    f"control/difference/{name}",
                    difference,
                    [f"value_{index}" for index in range(difference.size)],
                )
                self._scalar(
                    f"control/difference/{name}_l2",
                    float(np.linalg.norm(difference)),
                )
        reason = str(payload.get("rejection_reason", ""))
        self._text_if_changed("control/trace/rejection_reason", reason, level="WARN")
        self._scalar(
            "control/trace/validation_latency_seconds",
            int(payload.get("validation_latency_ns", 0)) * 1.0e-9,
        )
        self._scalar(
            "control/trace/send_latency_seconds",
            int(payload.get("send_latency_ns", 0)) * 1.0e-9,
        )

    def log_control_state(self, payload: Mapping[str, Any]) -> None:
        self._set_time(
            int(payload["stamp_ns"]), sequence=int(payload.get("generation", 0))
        )
        self._scalar("control/state/id", int(payload.get("state", 0)))
        for field in (
            "local_permission",
            "physical_pedal",
            "ft_zeroed_for_session",
            "arms_online",
            "hands_online",
        ):
            self._scalar(f"control/state/{field}", bool(payload.get(field, False)))
        self._scalar(
            "control/state/rdk_connection_generation",
            int(payload.get("generation", 0)),
        )
        self._text_if_changed("control/state/name", str(payload.get("state_name", "")))
        self._text_if_changed(
            "control/state/active_source", str(payload.get("active_source", ""))
        )
        hold_reason = str(payload.get("hold_reason", ""))
        self._text_if_changed("control/state/hold_reason", hold_reason, level="WARN")

    def _log_numeric_tree(self, path: str, value: Any, *, depth: int = 0) -> None:
        """Log bounded numeric leaves from a native DeviceIO payload."""

        if depth > 5:
            return
        if isinstance(value, bool):
            self._scalar(path, value)
            return
        if isinstance(value, (int, float, np.integer, np.floating)):
            if math.isfinite(float(value)):
                self._scalar(path, value)
            return
        if isinstance(value, str):
            if len(value) <= 512:
                self._text_if_changed(path, value)
            return
        if isinstance(value, Mapping):
            for key, child in value.items():
                self._log_numeric_tree(
                    f"{path}/{str(key).strip('/').replace(' ', '_')}",
                    child,
                    depth=depth + 1,
                )
            return
        if isinstance(value, Sequence) and not isinstance(
            value, (bytes, bytearray, memoryview)
        ):
            if not value or len(value) > 512:
                return
            if all(isinstance(item, (bool, int, float)) for item in value):
                array = np.asarray(value, dtype=np.float64)
                if np.all(np.isfinite(array)):
                    labels = self._semantic_vector_labels(path, array.size)
                    self._vector(path, array, labels)

    @staticmethod
    def _duration_payload_ns(value: Any) -> int:
        if isinstance(value, Mapping):
            return int(value.get("sec", 0)) * 1_000_000_000 + int(
                value.get("nanosec", 0)
            )
        return int(value or 0)

    @staticmethod
    def _offline_entity_root(normalized: str, payload: Any) -> str:
        """Map native DeviceIO topics onto the same entities as live ROS.

        Native recording keeps the source topic in the envelope (for example
        ``robot/left_arm/state``).  The live visualizer intentionally presents
        that state at ``robot/left_arm``.  Without this mapping, the offline
        curves exist in Rerun but the shared blueprint cannot find them.
        """

        if not isinstance(payload, Mapping):
            return normalized or "unknown"
        side = str(payload.get("side", "")).strip().lower()
        if side not in {"left", "right"}:
            if "/left_" in f"/{normalized}":
                side = "left"
            elif "/right_" in f"/{normalized}":
                side = "right"
            else:
                side = ""
        if side and normalized.endswith("_arm/state"):
            return f"robot/{side}_arm"
        if side and normalized.endswith("_hand/state"):
            return f"robot/{side}_hand"
        if side and normalized.endswith("tactile_raw"):
            return f"robot/{side}_hand/tactile"
        return normalized or "unknown"

    def log_deviceio(
        self,
        *,
        topic: str,
        payload: Any,
        playback_time_ns: int,
        original_time_ns: int,
        sequence: int,
        valid: bool,
        timing_valid: bool,
        invalid_reason: str = "",
    ) -> None:
        """Log one recorded native envelope without creating ROS publishers."""

        normalized = topic.strip("/")
        root = self._offline_entity_root(normalized, payload)
        # DeviceIO sequence numbers are local to each producer/topic. Exposing
        # them as one global Rerun timeline makes, for example, camera frame
        # #1139 coexist with arm sample #23876 and leaves the arm plots blank
        # while viewing the camera. Offline playback therefore uses only the
        # common playback clock; the original sequence remains inspectable as
        # ordinary per-stream metadata below.
        self._set_time(playback_time_ns)
        self._scalar(f"{root}/record/source_sequence", sequence)
        self._scalar(f"{root}/record/valid", valid)
        self._scalar(f"{root}/record/timing_valid", timing_valid)
        self._scalar(f"{root}/record/original_time_seconds", original_time_ns * 1e-9)
        self._text_if_changed(
            f"{root}/record/invalid_reason", invalid_reason, level="WARN"
        )
        if not isinstance(payload, Mapping):
            return

        if normalized.startswith("camera/"):
            camera = str(payload.get("camera_name", normalized.split("/")[1]))
            jpeg_b64 = payload.get("jpeg_b64")
            if isinstance(jpeg_b64, str) and jpeg_b64:
                self.stream.log(
                    f"camera/{camera}/color",
                    self.rr.EncodedImage(
                        contents=base64.b64decode(jpeg_b64), media_type="image/jpeg"
                    ),
                )
            raw_b64 = payload.get("raw_rgb_b64")
            if isinstance(raw_b64, str) and raw_b64:
                width, height = int(payload["width"]), int(payload["height"])
                rgb = np.frombuffer(base64.b64decode(raw_b64), dtype=np.uint8)
                if rgb.size != width * height * 3:
                    raise ValueError(f"{camera}: raw RGB byte count mismatch")
                self.stream.log(
                    f"camera/{camera}/color",
                    self.rr.Image(rgb.reshape(height, width, 3)),
                )
            depth_b64 = payload.get("depth_z16_b64")
            if isinstance(depth_b64, str) and depth_b64:
                width = int(payload["depth_width"])
                height = int(payload["depth_height"])
                depth = np.frombuffer(base64.b64decode(depth_b64), dtype="<u2")
                if depth.size != width * height:
                    raise ValueError(f"{camera}: depth sample count mismatch")
                depth_m = depth.reshape(height, width).astype(np.float32) * float(
                    payload["depth_scale_m"]
                )
                self.stream.log(
                    f"camera/{camera}/depth_m",
                    self.rr.DepthImage(depth_m, meter=1.0),
                )
            points_b64 = payload.get("pointcloud_xyz_f32_b64")
            if isinstance(points_b64, str) and points_b64:
                points = np.frombuffer(base64.b64decode(points_b64), dtype="<f4")
                if points.size % 3:
                    raise ValueError(f"{camera}: point cloud XYZ count mismatch")
                points = points.reshape(-1, 3)
                points = points[np.all(np.isfinite(points), axis=1)]
                self.stream.log(f"camera/{camera}/pointcloud", self.rr.Points3D(points))

        if normalized.endswith("_arm/state"):
            pose = payload.get("tcp_pose_rdk_xyz_wxyz")
            if isinstance(pose, Sequence) and len(pose) == 7:
                pose = np.asarray(pose, dtype=np.float64)
                self.stream.log(
                    f"{root}/tcp",
                    self.rr.Transform3D(
                        translation=pose[:3],
                        quaternion=self.rr.Quaternion(
                            xyzw=[pose[4], pose[5], pose[6], pose[3]]
                        ),
                    ),
                )

        if normalized.endswith("tactile_raw"):
            surfaces = payload.get("surfaces")
            if isinstance(surfaces, Sequence) and surfaces:
                side = str(payload.get("side", "")).strip().lower()
                if side not in {"left", "right"}:
                    side = (
                        "left"
                        if "left_hand" in normalized
                        else "right"
                        if "right_hand" in normalized
                        else ""
                    )
                self.stream.log(
                    f"{root}/atlas_raw_u16",
                    self.rr.Image(tactile_atlas(surfaces, side=side)),
                )
                atlas = tactile_atlas(surfaces, side=side)
                self.stream.log(
                    f"{root}/atlas_heatmap_u8",
                    self.rr.Image(
                        tactile_heatmap(
                            atlas,
                            tactile_sensor_mask(surfaces, side=side),
                        )
                    ),
                )

        command_stage = {
            "control/requested_command": "requested",
            "control/safe_command": "safe",
            "control/sent_command": "sent",
        }.get(normalized)
        if command_stage is not None:
            command = dict(payload)
            command["stamp_ns"] = playback_time_ns
            command["ttl_ns"] = self._duration_payload_ns(command.get("ttl"))
            try:
                self.log_command(command_stage, command)
            except ValueError as exc:
                self._text_if_changed(f"{root}/decode_error", str(exc), level="WARN")

        # Curves for arm/hand/control diagnostics and all remaining bounded
        # scalar/vector fields. Large image/tactile blobs are handled above.
        filtered = {
            key: value
            for key, value in payload.items()
            if not str(key).endswith("_b64") and key not in {"surfaces", "trajectory"}
        }
        self._log_numeric_tree(root, filtered)
