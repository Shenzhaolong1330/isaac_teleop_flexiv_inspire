"""Read-only ROS 2 subscriber that forwards latest observations to Rerun."""

from __future__ import annotations

import time
from typing import Any

from .runtime import LatestOnlyDispatcher, RerunVisualizer


class VisualizationRateLimiter:
    """Monotonic per-stream limiter used only by the Rerun observer."""

    def __init__(self) -> None:
        self._last_ns: dict[str, int] = {}

    def allow(
        self, key: str, rate_hz: float, *, now_monotonic_ns: int | None = None
    ) -> bool:
        if not 0.1 <= float(rate_hz) <= 120.0:
            raise ValueError("Rerun visualization rate must be in [0.1, 120] Hz")
        now = time.monotonic_ns() if now_monotonic_ns is None else now_monotonic_ns
        period_ns = max(1, int(1_000_000_000 / float(rate_hz)))
        previous = self._last_ns.get(key)
        if previous is not None and now - previous < period_ns:
            return False
        self._last_ns[key] = now
        return True


def _time_ns(value: Any) -> int:
    return int(value.sec) * 1_000_000_000 + int(value.nanosec)


def _duration_ns(value: Any) -> int:
    return int(value.sec) * 1_000_000_000 + int(value.nanosec)


def acquisition_to_dict(value: Any) -> dict[str, Any]:
    return {
        "source_time_ns": _time_ns(value.source_time),
        "host_receive_time_ns": _time_ns(value.host_receive_time),
        "acquisition_start_ns": _time_ns(value.acquisition_start),
        "acquisition_end_ns": _time_ns(value.acquisition_end),
        "sequence": int(value.source_sequence),
        "valid": bool(value.valid),
        "age_ns": _duration_ns(value.age),
        "invalid_reason": str(value.invalid_reason),
        "source_clock_domain": str(value.source_clock_domain),
        "host_clock_domain": str(value.host_clock_domain),
        "mapped_host_time_ns": _time_ns(value.mapped_host_time),
        "timing_valid": bool(value.timing_valid),
    }


def command_to_dict(message: Any) -> dict[str, Any]:
    points = []
    for point in message.trajectory:
        points.append(
            {
                "execute_after_ns": _duration_ns(point.execute_after),
                "left_delta_xyz": list(point.left_delta_xyz),
                "left_delta_rotation6d": list(point.left_delta_rotation6d),
                "right_delta_xyz": list(point.right_delta_xyz),
                "right_delta_rotation6d": list(point.right_delta_rotation6d),
                "left_delta_quaternion_xyzw": list(point.left_delta_quaternion_xyzw),
                "right_delta_quaternion_xyzw": list(point.right_delta_quaternion_xyzw),
                "left_arm_joint_positions": list(point.left_arm_joint_positions),
                "right_arm_joint_positions": list(point.right_arm_joint_positions),
                "left_hand_targets": list(point.left_hand_targets),
                "right_hand_targets": list(point.right_hand_targets),
            }
        )
    representation = int(message.representation)
    representation_name = {
        int(getattr(type(message), "CARTESIAN_ROT6D", 1)): "CARTESIAN_ROT6D",
        int(getattr(type(message), "CARTESIAN_QUATERNION", 2)): "CARTESIAN_QUATERNION",
        int(getattr(type(message), "JOINT_POSITION", 3)): "JOINT_POSITION",
    }.get(representation, f"UNKNOWN_{representation}")
    return {
        "stamp_ns": _time_ns(message.header.stamp),
        "frame_id": str(message.frame_id),
        "schema_version": int(message.schema_version),
        "session_id": str(message.session_id),
        "source": str(message.source),
        "sequence": int(message.sequence),
        "ttl_ns": _duration_ns(message.ttl),
        "representation": representation,
        "representation_name": representation_name,
        "rotation_order": str(message.rotation_order),
        "valid_mask": int(message.valid_mask),
        "deadman": bool(message.deadman),
        "trajectory": points,
    }


def arm_to_dict(message: Any) -> dict[str, Any]:
    pose = message.tcp_pose
    twist = message.tcp_twist
    raw = message.raw_ft
    wrench = message.tcp_wrench
    return {
        "stamp_ns": _time_ns(message.header.stamp),
        "side": str(message.side),
        "acquisition": acquisition_to_dict(message.acquisition),
        "q": list(message.q),
        "dq": list(message.dq),
        "tau": list(message.tau),
        "tau_des": list(message.tau_des),
        "tau_ext": list(message.tau_ext),
        "tau_interact": list(message.tau_interact),
        "tcp_pose": {
            "position": [pose.position.x, pose.position.y, pose.position.z],
            "quaternion_xyzw": [
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ],
        },
        "tcp_twist": [
            twist.linear.x,
            twist.linear.y,
            twist.linear.z,
            twist.angular.x,
            twist.angular.y,
            twist.angular.z,
        ],
        "raw_ft": [
            raw.force.x,
            raw.force.y,
            raw.force.z,
            raw.torque.x,
            raw.torque.y,
            raw.torque.z,
        ],
        "tcp_wrench": [
            wrench.force.x,
            wrench.force.y,
            wrench.force.z,
            wrench.torque.x,
            wrench.torque.y,
            wrench.torque.z,
        ],
        "temperature": list(message.temperature),
        "generation": int(message.rdk_connection_generation),
        "connected": bool(message.connected),
        "fault": str(message.fault),
    }


def hand_to_dict(message: Any) -> dict[str, Any]:
    return {
        "stamp_ns": _time_ns(message.header.stamp),
        "side": str(message.side),
        "sequence": int(message.sequence),
        "acquisition": acquisition_to_dict(message.acquisition),
        "angle": list(message.angle),
        "position": list(message.position),
        "actual_force": list(message.actual_force),
        "current": list(message.current),
        "temperature": list(message.temperature),
        "error": list(message.error),
        "status": list(message.status),
        "connected": bool(message.connected),
        "fault": bool(message.fault),
        "fault_reason": str(message.fault_reason),
    }


def ergonomics_to_dict(side: str, message: Any) -> dict[str, Any]:
    if len(message.name) != len(message.position):
        raise ValueError("MANUS Ergonomics names/positions differ")
    return {
        "stamp_ns": _time_ns(message.header.stamp),
        "side": side,
        "names": [str(name) for name in message.name],
        "values_rad": [float(value) for value in message.position],
    }


def tactile_to_dict(message: Any) -> dict[str, Any]:
    surfaces = []
    for surface in message.surfaces:
        surface_acquisition = acquisition_to_dict(surface.acquisition)
        surfaces.append(
            {
                "name": str(surface.name),
                "rows": int(surface.rows),
                "columns": int(surface.columns),
                "taxels": list(surface.taxels),
                "acquisition": surface_acquisition,
                "valid": bool(surface_acquisition["valid"]),
                "invalid_reason": str(surface_acquisition["invalid_reason"]),
                "acquisition_start_ns": int(
                    surface_acquisition["acquisition_start_ns"]
                ),
                "acquisition_end_ns": int(surface_acquisition["acquisition_end_ns"]),
            }
        )
    return {
        "stamp_ns": _time_ns(message.header.stamp),
        "side": str(message.side),
        "sequence": int(message.sequence),
        "acquisition": acquisition_to_dict(message.acquisition),
        "surfaces": surfaces,
        "taxel_count": int(message.taxel_count),
        "valid": bool(message.valid),
        "invalid_reason": str(message.invalid_reason),
    }


def trace_to_dict(message: Any) -> dict[str, Any]:
    return {
        "stamp_ns": _time_ns(message.header.stamp),
        "session_id": str(message.session_id),
        "trace_sequence": int(message.trace_sequence),
        "requested": command_to_dict(message.requested),
        "safe": command_to_dict(message.safe),
        "sent": command_to_dict(message.sent),
        "requested_valid": bool(message.requested_valid),
        "safe_valid": bool(message.safe_valid),
        "sent_valid": bool(message.sent_valid),
        "rejection_reason": str(message.rejection_reason),
        "validation_latency_ns": _duration_ns(message.validation_latency),
        "send_latency_ns": _duration_ns(message.send_latency),
    }


def control_state_to_dict(message: Any) -> dict[str, Any]:
    return {
        "stamp_ns": _time_ns(message.header.stamp),
        "state": int(message.state),
        "state_name": str(message.state_name),
        "session_id": str(message.session_id),
        "active_source": str(message.active_source),
        "hold_reason": str(message.hold_reason),
        "local_permission": bool(message.local_permission),
        "physical_pedal": bool(message.physical_pedal),
        "ft_zeroed_for_session": bool(message.ft_zeroed_for_session),
        "arms_online": bool(message.arms_online),
        "hands_online": bool(message.hands_online),
        "generation": int(message.rdk_connection_generation),
    }


def camera_frame_to_dict(message: Any) -> dict[str, Any]:
    """Convert the atomic camera frame; never pair separate topics by proximity."""

    return {
        "camera": str(message.camera),
        "stamp_ns": _time_ns(message.header.stamp),
        "width": int(message.width),
        "height": int(message.height),
        "format": str(message.image.format),
        "jpeg": bytes(message.image.data),
        "acquisition": acquisition_to_dict(message.acquisition),
        "timing_unpaired": False,
    }


def depth_image_to_dict(camera: str, message: Any) -> dict[str, Any]:
    """Copy a live Z16 depth image without trying to time-pair it later."""

    return {
        "camera": camera,
        "stamp_ns": _time_ns(message.header.stamp),
        "width": int(message.width),
        "height": int(message.height),
        "step": int(message.step),
        "encoding": str(message.encoding),
        "is_bigendian": bool(message.is_bigendian),
        "data": bytes(message.data),
        # The site RealSense camera publishes standard Z16 millimetres.  The
        # point cloud below remains the metric/authoritative 3D modality.
        "meter_per_unit": 0.001,
    }


def pointcloud_to_dict(camera: str, message: Any) -> dict[str, Any]:
    """Copy the live PointCloud2 layout for deferred latest-only decoding."""

    return {
        "camera": camera,
        "stamp_ns": _time_ns(message.header.stamp),
        "frame_id": str(message.header.frame_id),
        "width": int(message.width),
        "height": int(message.height),
        "point_step": int(message.point_step),
        "row_step": int(message.row_step),
        "is_bigendian": bool(message.is_bigendian),
        "fields": {
            str(field.name): {
                "offset": int(field.offset),
                "datatype": int(field.datatype),
                "count": int(field.count),
            }
            for field in message.fields
        },
        "data": bytes(message.data),
    }


def run_ros(
    *,
    save_path: str | None,
    connect_url: str | None,
    spawn: bool,
    viewer_port: int,
    telemetry_hz: float = 5.0,
    tactile_hz: float = 10.0,
    image_hz: float = 10.0,
    pointcloud_hz: float = 2.0,
    legacy_camera_topics: bool = False,
    ros_args: list[str] | None = None,
) -> int:
    """Run until ROS shutdown. This function creates subscriptions only."""

    import rclpy
    from flexiv_inspire_interfaces.msg import (
        ArmState,
        BimanualCommand,
        CameraFrame,
        CommandTrace,
        ControlState,
        HandState,
        TactileFrame,
    )
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CompressedImage, Image, JointState, PointCloud2

    sensor_qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.BEST_EFFORT,
    )
    control_qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
    )

    rates = {
        "telemetry": float(telemetry_hz),
        "tactile": float(tactile_hz),
        "images": float(image_hz),
        "pointcloud": float(pointcloud_hz),
    }
    for name, rate in rates.items():
        if not 0.1 <= rate <= 120.0:
            raise ValueError(f"Rerun {name} rate must be in [0.1, 120] Hz")

    visualizer = RerunVisualizer(
        save_path=save_path,
        connect_url=connect_url,
        spawn=spawn,
        viewer_port=viewer_port,
    )
    # A large PointCloud2 decode must never queue behind (or in front of) the
    # live robot/tactile state. Each lane remains latest-only, while expensive
    # modalities can progress independently.
    telemetry_dispatcher = LatestOnlyDispatcher(name="rerun-telemetry")
    tactile_dispatcher = LatestOnlyDispatcher(name="rerun-tactile")
    image_dispatcher = LatestOnlyDispatcher(name="rerun-images")
    pointcloud_dispatcher = LatestOnlyDispatcher(name="rerun-pointcloud")
    dispatchers = {
        "telemetry": telemetry_dispatcher,
        "tactile": tactile_dispatcher,
        "images": image_dispatcher,
        "pointcloud": pointcloud_dispatcher,
    }

    class RerunNode(Node):
        def __init__(self) -> None:
            super().__init__("flexiv_inspire_rerun")
            self._viz_subscriptions = []
            self._tactile_frames = {"left": 0, "right": 0}
            self._rate_limiter = VisualizationRateLimiter()
            for camera in ("head", "left_wrist", "right_wrist"):
                self._viz_subscriptions.append(
                    self.create_subscription(
                        CameraFrame,
                        f"/camera/{camera}/color/frame",
                        lambda message, selected=camera: self._submit_if_due(
                            image_dispatcher,
                            f"camera:{selected}",
                            rates["images"],
                            visualizer.log_camera,
                            lambda: camera_frame_to_dict(message),
                        ),
                        sensor_qos,
                    )
                )
                self._viz_subscriptions.append(
                    self.create_subscription(
                        Image,
                        f"/camera/{camera}/depth/image_rect_raw",
                        lambda message, selected=camera: self._submit_if_due(
                            image_dispatcher,
                            f"depth:{selected}",
                            rates["images"],
                            visualizer.log_depth,
                            lambda: depth_image_to_dict(selected, message),
                        ),
                        sensor_qos,
                    )
                )
                self._viz_subscriptions.append(
                    self.create_subscription(
                        PointCloud2,
                        f"/camera/{camera}/depth/points",
                        lambda message, selected=camera: self._submit_if_due(
                            pointcloud_dispatcher,
                            f"pointcloud:{selected}",
                            rates["pointcloud"],
                            visualizer.log_pointcloud,
                            lambda: pointcloud_to_dict(selected, message),
                        ),
                        sensor_qos,
                    )
                )
                if legacy_camera_topics:
                    self._viz_subscriptions.append(
                        self.create_subscription(
                            CompressedImage,
                            f"/camera/{camera}/color/image_raw/compressed",
                            lambda message, selected=camera: self._submit_if_due(
                                image_dispatcher,
                                f"legacy-camera:{selected}",
                                rates["images"],
                                visualizer.log_camera,
                                lambda: {
                                    "camera": selected,
                                    "stamp_ns": _time_ns(message.header.stamp),
                                    "format": str(message.format),
                                    "jpeg": bytes(message.data),
                                    "timing_unpaired": True,
                                },
                            ),
                            sensor_qos,
                        )
                    )
            for side in ("left", "right"):
                self._viz_subscriptions.append(
                    self.create_subscription(
                        ArmState,
                        f"/robot/{side}_arm/state",
                        lambda message, selected=side: self._submit_if_due(
                            telemetry_dispatcher,
                            f"arm:{selected}",
                            rates["telemetry"],
                            visualizer.log_arm,
                            lambda: arm_to_dict(message),
                        ),
                        sensor_qos,
                    )
                )
                self._viz_subscriptions.append(
                    self.create_subscription(
                        HandState,
                        f"/robot/{side}_hand/state",
                        lambda message, selected=side: self._submit_if_due(
                            telemetry_dispatcher,
                            f"hand:{selected}",
                            rates["telemetry"],
                            visualizer.log_hand,
                            lambda: hand_to_dict(message),
                        ),
                        sensor_qos,
                    )
                )
                self._viz_subscriptions.append(
                    self.create_subscription(
                        JointState,
                        f"/manus/{side}/ergonomics",
                        lambda message, selected=side: self._submit_if_due(
                            telemetry_dispatcher,
                            f"manus-ergonomics:{selected}",
                            rates["telemetry"],
                            visualizer.log_manus_ergonomics,
                            lambda: ergonomics_to_dict(selected, message),
                        ),
                        sensor_qos,
                    )
                )
                self._viz_subscriptions.append(
                    self.create_subscription(
                        TactileFrame,
                        f"/robot/{side}_hand/tactile_raw",
                        lambda message, selected=side: self._on_tactile(
                            selected, message
                        ),
                        sensor_qos,
                    )
                )
            for stage in ("requested", "safe", "sent"):
                self._viz_subscriptions.append(
                    self.create_subscription(
                        BimanualCommand,
                        f"/control/{stage}_command",
                        lambda message, selected=stage: self._submit_if_due(
                            telemetry_dispatcher,
                            f"command:{selected}",
                            rates["telemetry"],
                            lambda command, stage_name=selected: visualizer.log_command(
                                stage_name, command
                            ),
                            lambda: command_to_dict(message),
                        ),
                        control_qos,
                    )
                )
            self._viz_subscriptions.append(
                self.create_subscription(
                    CommandTrace,
                    "/control/command_trace",
                    lambda message: self._submit_if_due(
                        telemetry_dispatcher,
                        "command-trace",
                        rates["telemetry"],
                        visualizer.log_trace,
                        lambda: trace_to_dict(message),
                    ),
                    control_qos,
                )
            )
            self._viz_subscriptions.append(
                self.create_subscription(
                    ControlState,
                    "/control/state",
                    lambda message: self._submit_if_due(
                        telemetry_dispatcher,
                        "control-state",
                        rates["telemetry"],
                        visualizer.log_control_state,
                        lambda: control_state_to_dict(message),
                    ),
                    control_qos,
                )
            )
            self.create_timer(5.0, self._report)
            sink = (
                f"save:{visualizer.save_path}"
                if visualizer.save_path is not None
                else (
                    f"connect:{connect_url}" if connect_url else f"spawn:{viewer_port}"
                )
            )
            self.get_logger().info(
                f"read-only latest-only Rerun subscriber active ({sink}); "
                f"display rates telemetry={rates['telemetry']:g}Hz "
                f"tactile={rates['tactile']:g}Hz images={rates['images']:g}Hz "
                f"pointcloud={rates['pointcloud']:g}Hz; "
                "atomic CameraFrame is authoritative; "
                "this node has no publishers or hardware access"
            )

        def _submit_if_due(
            self,
            dispatcher: LatestOnlyDispatcher,
            key: str,
            rate_hz: float,
            function,
            payload_factory,
        ) -> bool:
            if not self._rate_limiter.allow(key, rate_hz):
                return False
            return dispatcher.submit(key, function, payload_factory())

        def _on_tactile(self, side: str, message: TactileFrame) -> None:
            self._tactile_frames[side] += 1
            self._submit_if_due(
                tactile_dispatcher,
                f"tactile:{side}",
                rates["tactile"],
                visualizer.log_tactile,
                lambda: tactile_to_dict(message),
            )

        def _report(self) -> None:
            lane_stats = {
                name: dispatcher.stats for name, dispatcher in dispatchers.items()
            }
            message = "Rerun " + " ".join(
                f"{name}[in={stats.submitted},done={stats.processed},"
                f"drop={stats.dropped},fail={stats.failed}]"
                for name, stats in lane_stats.items()
            )
            message += (
                " tactile_rx="
                f"left:{self._tactile_frames['left']} "
                f"right:{self._tactile_frames['right']}"
            )
            failed = [
                (name, dispatcher.last_error)
                for name, dispatcher in dispatchers.items()
                if dispatcher.stats.failed
            ]
            if failed:
                message += " last_error=" + ";".join(
                    f"{name}:{error}" for name, error in failed
                )
                self.get_logger().warning(message)
            else:
                self.get_logger().info(message)

    rclpy.init(args=ros_args)
    node = RerunNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cleanup_errors: list[str] = []
        cleanup_steps = [
            ("node", node.destroy_node),
            *(
                (
                    f"dispatcher-{name}",
                    lambda selected=dispatcher: selected.close(drain=False),
                )
                for name, dispatcher in dispatchers.items()
            ),
            ("visualizer", visualizer.close),
        ]
        for name, cleanup in cleanup_steps:
            try:
                cleanup()
            except Exception as exc:
                cleanup_errors.append(f"{name}: {exc}")
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception as exc:
            cleanup_errors.append(f"rclpy: {exc}")
        if cleanup_errors:
            raise RuntimeError("; ".join(cleanup_errors))
    return 0
