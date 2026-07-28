"""Read-only ROS 2 subscriber that forwards latest observations to Rerun."""

from __future__ import annotations

from typing import Any

from .runtime import LatestOnlyDispatcher, RerunVisualizer


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
                "acquisition_start_ns": int(surface_acquisition["acquisition_start_ns"]),
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


def run_ros(
    *,
    save_path: str | None,
    connect_url: str | None,
    spawn: bool,
    viewer_port: int,
    legacy_camera_topics: bool = False,
    ros_args: list[str] | None = None,
) -> int:
    """Run until ROS shutdown. This function creates subscriptions only."""

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CompressedImage
    from flexiv_inspire_interfaces.msg import (
        ArmState,
        BimanualCommand,
        CameraFrame,
        CommandTrace,
        ControlState,
        HandState,
        TactileFrame,
    )

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

    visualizer = RerunVisualizer(
        save_path=save_path,
        connect_url=connect_url,
        spawn=spawn,
        viewer_port=viewer_port,
    )
    dispatcher = LatestOnlyDispatcher()

    class RerunNode(Node):
        def __init__(self) -> None:
            super().__init__("flexiv_inspire_rerun")
            self._viz_subscriptions = []
            for camera in ("head", "left_wrist", "right_wrist"):
                self._viz_subscriptions.append(
                    self.create_subscription(
                        CameraFrame,
                        f"/camera/{camera}/color/frame",
                        lambda message, selected=camera: dispatcher.submit(
                            f"camera:{selected}",
                            visualizer.log_camera,
                            camera_frame_to_dict(message),
                        ),
                        sensor_qos,
                    )
                )
                if legacy_camera_topics:
                    self._viz_subscriptions.append(
                        self.create_subscription(
                            CompressedImage,
                            f"/camera/{camera}/color/image_raw/compressed",
                            lambda message, selected=camera: dispatcher.submit(
                                f"legacy-camera:{selected}",
                                visualizer.log_camera,
                                {
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
                        lambda message, selected=side: dispatcher.submit(
                            f"arm:{selected}", visualizer.log_arm, arm_to_dict(message)
                        ),
                        sensor_qos,
                    )
                )
                self._viz_subscriptions.append(
                    self.create_subscription(
                        HandState,
                        f"/robot/{side}_hand/state",
                        lambda message, selected=side: dispatcher.submit(
                            f"hand:{selected}", visualizer.log_hand, hand_to_dict(message)
                        ),
                        sensor_qos,
                    )
                )
                self._viz_subscriptions.append(
                    self.create_subscription(
                        TactileFrame,
                        f"/robot/{side}_hand/tactile_raw",
                        lambda message, selected=side: dispatcher.submit(
                            f"tactile:{selected}",
                            visualizer.log_tactile,
                            tactile_to_dict(message),
                        ),
                        sensor_qos,
                    )
                )
            for stage in ("requested", "safe", "sent"):
                self._viz_subscriptions.append(
                    self.create_subscription(
                        BimanualCommand,
                        f"/control/{stage}_command",
                        lambda message, selected=stage: dispatcher.submit(
                            f"command:{selected}",
                            lambda command, stage_name=selected: visualizer.log_command(
                                stage_name, command
                            ),
                            command_to_dict(message),
                        ),
                        control_qos,
                    )
                )
            self._viz_subscriptions.append(
                self.create_subscription(
                    CommandTrace,
                    "/control/command_trace",
                    lambda message: dispatcher.submit(
                        "command-trace", visualizer.log_trace, trace_to_dict(message)
                    ),
                    control_qos,
                )
            )
            self._viz_subscriptions.append(
                self.create_subscription(
                    ControlState,
                    "/control/state",
                    lambda message: dispatcher.submit(
                        "control-state",
                        visualizer.log_control_state,
                        control_state_to_dict(message),
                    ),
                    control_qos,
                )
            )
            self.create_timer(5.0, self._report)
            sink = (
                f"save:{visualizer.save_path}"
                if visualizer.save_path is not None
                else (f"connect:{connect_url}" if connect_url else f"spawn:{viewer_port}")
            )
            self.get_logger().info(
                f"read-only latest-only Rerun subscriber active ({sink}); "
                "atomic CameraFrame is authoritative; "
                "this node has no publishers or hardware access"
            )

        def _report(self) -> None:
            stats = dispatcher.stats
            message = (
                f"Rerun submitted={stats.submitted} processed={stats.processed} "
                f"dropped_old={stats.dropped} failed={stats.failed}"
            )
            if stats.failed:
                message += f" last_error={dispatcher.last_error}"
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
        for name, cleanup in (
            ("node", node.destroy_node),
            ("dispatcher", lambda: dispatcher.close(drain=True)),
            ("visualizer", visualizer.close),
        ):
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
