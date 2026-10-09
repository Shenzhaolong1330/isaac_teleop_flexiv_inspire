"""Production bridge integration for explicitly selected fixed mapping mode."""

from __future__ import annotations
from dataclasses import replace
import copy
import hashlib
import json
import threading
import time
from pathlib import Path
import numpy as np
import yaml
from std_msgs.msg import String
from isaac_teleop_core.command import CommandSource
from isaac_teleop_core.control import ControlState
from isaac_teleop_core.world_teleop import (
    FixedMapping,
    WorldController,
    Limits,
    ApprovedRegion,
    Pose,
)
from isaac_teleop_core.rotation6d import rdk_pose_to_ros_pose
from .frames import rdk_pose_base_to_world


class WorldBridge:
    def _init_world(self):
        self.declare_parameter("teleop_mapping_mode", "incremental")
        self.declare_parameter("world_installation", "")
        self.declare_parameter("world_calibration", "")
        self.declare_parameter("world_execution_config", "")
        self.declare_parameter("world_tool_payload_config", "")
        self._world = None
        self._world_tracking = None
        self._world_send_lock = threading.RLock()
        self._world_lock = threading.RLock()
        self._world_state_pub = self.create_publisher(String, "/teleop/world_state", 10)
        if self.get_parameter("teleop_mapping_mode").value == "world_absolute":
            # Missing/invalid calibration permits observation but never motion.
            try:
                mapper = FixedMapping.load(
                    self.get_parameter("world_installation").value,
                    self.get_parameter("world_calibration").value,
                )
                for s in ("left", "right"):
                    configured = Pose.read(mapper.raw["world_from_base"][s])
                    actual = self._world_from_base[s]
                    if not np.allclose(
                        configured.p, actual.translation_m
                    ) or not np.allclose(configured.r, actual.rotation):
                        raise ValueError(
                            "world calibration bases disagree with bridge frame config"
                        )
                cfg = yaml.safe_load(
                    Path(self.get_parameter("world_execution_config").value).read_text()
                )
                if cfg.get("schema_version") != 1:
                    raise ValueError("invalid world execution config version")
                self._world_tool_path = Path(
                    self.get_parameter("world_tool_payload_config").value
                )
                self._world_tool_stat = None
                self._check_world_tool(mapper)
                self._world = WorldController(
                    mapper,
                    Limits(**cfg["limits"]),
                    ApprovedRegion(cfg.get("approved_region")),
                )
            except Exception as exc:
                self.get_logger().error(
                    f"world_absolute INVALID; observation only: {exc}"
                )
        self.create_subscription(
            String, "/teleop/world_operation", self._world_operation, 10
        )

    def _check_world_tool(self, mapper):
        path = getattr(self, "_world_tool_path", None)
        if path is None:
            return  # Dependency-isolated test bridge; constructor always binds this.
        stat = path.stat()
        signature = (stat.st_mtime_ns, stat.st_size, stat.st_ino)
        if signature != self._world_tool_stat:
            value = yaml.safe_load(path.read_text())
            canonical = json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            if (
                hashlib.sha256(canonical.encode()).hexdigest()
                != mapper.raw["tool_identity"]
            ):
                if self._world:
                    self._world.hold("tool_payload_identity_changed", invalid=True)
                raise ValueError(
                    "tool_identity must match audited tool/payload canonical SHA256; recalibrate on change"
                )
            self._world_tool_stat = signature

    def _world_selected(self):
        return self.get_parameter("teleop_mapping_mode").value == "world_absolute"

    def _world_motion_authority(self):
        world = getattr(self, "_world", None)
        return world is not None and world.authority and not self._home_inflight

    def _world_feedback(self):
        self._check_world_tool(self._world.mapping)
        with self._state_lock:
            result = {"receive_ns": min(self._last_arm_mapped_ns.values()), "q": {}}
            for s in ("left", "right"):
                wire = self._latest_wire[s]
                if not wire.get("connected") or wire.get("fault"):
                    raise ValueError("robot_offline_or_fault")
                q_measured = np.asarray(wire["q"], dtype=float)
                if (
                    not self._arm_safety_ok[s]
                    or q_measured.shape != (7,)
                    or not np.isfinite(q_measured).all()
                ):
                    raise ValueError("robot_measurement_invalid_or_unsafe")
                result["q"][s] = q_measured.tolist()
                p, q = rdk_pose_to_ros_pose(
                    rdk_pose_base_to_world(
                        np.asarray(wire["tcp_pose_rdk_xyz_wxyz"], dtype=float),
                        self._world_from_base[s],
                    )
                )
                result[s] = [*p.tolist(), *q.tolist()]
            return result

    def _world_event(self):
        world = self._world
        payload = (
            world.audit()
            if world
            else {"state": "INVALID", "reason": "calibration_missing_or_invalid"}
        )
        payload["physical_pedal"] = self._physical_pedal
        msg = String()
        msg.data = json.dumps(payload, separators=(",", ":"))
        self._world_state_pub.publish(msg)
        emitter = getattr(self, "_deviceio", None)
        if emitter is None or (
            hasattr(emitter, "socket_path") and not emitter.socket_path.is_socket()
        ):
            return
        signature = (
            payload["state"],
            payload["reason"],
            payload.get("execution_epoch"),
        )
        from isaac_teleop_core.deviceio import record_envelope

        now = time.monotonic_ns()
        common = dict(
            producer="control",
            source_time_ns=now,
            host_receive_time_ns=now,
            sequence=now,
        )
        try:
            emitter.emit(
                record_envelope(
                    topic="/maintenance/world_state", payload=payload, **common
                ),
                critical=True,
            )
            if signature != getattr(self, "_world_signature", None):
                emitter.emit(
                    record_envelope(
                        topic="/episode/events",
                        payload={
                            "schema": "episode-event-v2",
                            "event_type": "world_control_changed",
                            "effective_time_ns": now,
                            "discontinuity": True,
                            "world": payload,
                        },
                        **common,
                    ),
                    critical=True,
                )
                self._world_signature = signature
        except (BufferError, RuntimeError, ValueError) as exc:
            if world and world.authority:
                world.hold(f"world_audit_failed:{exc}")
                self._arbiter.stop()
                self._send_hold_once("world_audit_failed")
            self.get_logger().error(f"critical world audit failed: {exc}")

    def _world_receive(self, message):
        output = copy.deepcopy(message)
        metadata = dict(zip(message.metadata_keys, message.metadata_values))
        try:
            tracking = json.loads(metadata["world_tracking"])
            self._world_tracking = tracking
            if (
                self._world
                and not self._world.authority
                and self._world.state not in {"INVALID", "HOMING"}
            ):
                feedback = self._world_feedback()
                self._world._validate(tracking, feedback, time.monotonic_ns())
                self._world.desired = self._world.mapping.targets(tracking)
                self._world.measured = {
                    s: Pose.read(feedback[s]) for s in ("left", "right")
                }
                self._world.measured_q = feedback.get("q", {})
                self._world.last_tracking = tracking.copy()
            enabled = metadata.get("world_command_enabled") == "true"
            if not enabled and self._world and self._world.authority:
                self._world.hold("input_command_disabled")
            authority = enabled and self._world_motion_authority()
            if (
                self._world
                and self._world.fingers
                and any(
                    self._hand_angles[s] is None
                    or time.monotonic_ns() - self._last_hand_observation_ns[s]
                    > 150_000_000
                    for s in ("left", "right")
                )
            ):
                self._world.fingers = False
        except Exception as exc:
            authority = False
            if self._world:
                self._world.hold(f"input_metadata_invalid:{exc}")
        output.deadman = authority
        output.valid_mask = (
            (
                3
                | (
                    int(message.valid_mask) & 12
                    if self._world
                    and self._world.fingers
                    and self._world.state == "FOLLOW"
                    else 0
                )
            )
            if authority
            else 0
        )
        if not authority and self._arbiter.snapshot.state == ControlState.ACTIVE:
            self._arbiter.stop()
            self._send_hold_once("world_authority_lost")
        self._world_event()
        return output

    def _world_operation(self, message):
        """Local ROS operation is explicit intent; daemon local token is still required."""
        try:
            operation = json.loads(message.data)
            if (
                operation.get("session_id") != self._session_id
                or operation.get("confirm") != "WORLD-TELEOP-LOCAL"
            ):
                raise ValueError("explicit session confirmation required")
            if not self._world:
                raise ValueError(
                    "valid calibration required; reload bridge after explicit calibration"
                )
            with self._world_send_lock, self._world_lock:
                action = operation["operation"]
                if action in {"stop", "invalidate"}:
                    self._world.hold(
                        "explicit_" + action, invalid=action == "invalidate"
                    )
                    self._arbiter.observe_deadman_released(CommandSource.TELEOP)
                    self._arbiter.stop()
                    self._send_hold_once("world_" + action)
                elif action == "start":
                    if self._home_inflight:
                        raise ValueError("Home owns hardware")
                    snap = self._arbiter.snapshot
                    if snap.state not in {
                        ControlState.READY,
                        ControlState.HOLD_LATCHED,
                        ControlState.TELEOP_ARMED,
                    }:
                        raise ValueError(
                            "start requires locally prepared READY/HOLD/TELEOP_ARMED"
                        )
                    if not self._local_permission or (
                        not self._rdk_control_lease_active
                        and (
                            not self._pending_arm_token
                            or time.monotonic_ns() >= self._pending_arm_token_expiry_ns
                        )
                    ):
                        raise ValueError(
                            "fresh local control authorization required before start"
                        )
                    if snap.state == ControlState.HOLD_LATCHED:
                        self._arbiter.observe_deadman_released(CommandSource.TELEOP)
                        self._arbiter.clear_hold(local_acknowledged=True)
                    self._world.start(
                        self._world_tracking,
                        self._world_feedback(),
                        (self._daemon_instance_id, self._connection_generation),
                        time.monotonic_ns(),
                    )
                    self._update_gates(time.monotonic_ns())
                    self._try_arm_pending_authorization(time.monotonic_ns())
                elif action == "fingers":
                    if self._world.state != "FOLLOW":
                        raise ValueError("finger takeover requires FOLLOW")
                    # Explicit capture is allowed only near current measured hand angles.
                    tracking_hands = operation.get("targets", {})
                    for s in ("left", "right"):
                        values = np.asarray(tracking_hands[s], dtype=float)
                        if (
                            values.shape != (6,)
                            or not np.isfinite(values).all()
                            or self._hand_angles[s] is None
                            or time.monotonic_ns() - self._last_hand_observation_ns[s]
                            > 150_000_000
                            or np.max(np.abs(values - self._hand_angles[s]))
                            > self._world.limits.finger_takeover_tolerance
                        ):
                            raise ValueError(
                                "finger targets must match fresh measured angles before takeover"
                            )
                    self._world.finger_targets = {
                        s: np.asarray(tracking_hands[s], dtype=float)
                        for s in ("left", "right")
                    }
                    self._world.fingers = True
                else:
                    raise ValueError("unknown world operation")
                self._world_event()
        except Exception as exc:
            self.get_logger().error(f"world operation rejected: {exc}")
            self._world_event()

    def _world_point(self, command, point):
        tracking = json.loads(command.metadata["world_tracking"])
        delta = self._world.propose(
            tracking,
            self._world_feedback(),
            (self._daemon_instance_id, self._connection_generation),
            time.monotonic_ns(),
        )
        hands = {s: getattr(point, s + "_hand_targets") for s in ("left", "right")}
        if self._world.fingers:
            for s in ("left", "right"):
                measured = self._hand_angles[s]
                hands[s] = measured + np.clip(
                    hands[s] - measured,
                    -self._world.limits.finger_lead_units,
                    self._world.limits.finger_lead_units,
                )
                self._world.finger_limited_target = hands.copy()
        self._world.executed_point = replace(
            delta, left_hand_targets=hands["left"], right_hand_targets=hands["right"]
        )
        return self._world.executed_point
