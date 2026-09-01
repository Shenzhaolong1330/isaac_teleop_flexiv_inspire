"""Local, interactive F/T-zero preflight and ROS Action client."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Bool, Empty, String
from std_srvs.srv import Trigger
import yaml

from flexiv_inspire_interfaces.action import ZeroFTSensors
from flexiv_inspire_interfaces.msg import ControlState, HandState

from .ipc_client import RDKIPCClient


def _canonical_hash(path: Path) -> str:
    document = yaml.safe_load(path.read_bytes())
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError("invalid tool/payload configuration")
    for side in ("left", "right"):
        if not bool(document["arms"][side].get("locally_verified", False)):
            raise ValueError(f"{side} tool/payload is not locally verified")
    canonical = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class _Preflight(Node):
    def __init__(self) -> None:
        super().__init__("zero_ft_local")
        self.samples: dict[str, list[np.ndarray]] = {"left": [], "right": []}
        self.latest: dict[str, HandState] = {}
        self.hand_seen: dict[str, HandState] = {}
        self.control_state: ControlState | None = None
        self.home_status: dict[str, object] | None = None
        self._last_home_progress = ""
        for side in ("left", "right"):
            self.create_subscription(
                HandState,
                f"/robot/{side}_hand/state",
                lambda message, selected=side: self._hand(selected, message),
                qos_profile_sensor_data,
            )
        self.create_subscription(ControlState, "/control/state", self._state, 10)
        self.create_subscription(String, "/control/home_status", self._home, 10)
        self.action = ActionClient(
            self, ZeroFTSensors, "/maintenance/zero_ft_sensors"
        )
        self.hand_reset = self.create_client(
            Trigger, "/maintenance/cycle_hands"
        )

    def _hand(self, side: str, message: HandState) -> None:
        self.hand_seen[side] = message
        if message.connected and not message.fault:
            values = np.asarray(message.angle, dtype=np.float64)
            if values.shape == (6,) and np.all(np.isfinite(values)):
                self.latest[side] = message
                self.samples[side].append(values)

    def _state(self, message: ControlState) -> None:
        self.control_state = message

    def _home(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            return
        if isinstance(payload, dict):
            self.home_status = payload
            state = str(payload.get("state", "")).strip().lower()
            reason = str(payload.get("reason", "")).strip()
            if state == "moving" and reason != self._last_home_progress:
                if reason == "home_lift_in_progress":
                    print("Home: 正在抬升 TCP 并同步对齐 Home XY", flush=True)
                elif reason == "home_in_progress":
                    print("Home: 安全高度已到达，正在执行关节 Home", flush=True)
                self._last_home_progress = reason


def _spin_until(node: Node, predicate, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out waiting for local ROS state")
        rclpy.spin_once(node, timeout_sec=0.05)


def _initial_state_problem(node: _Preflight) -> str:
    missing: list[str] = []
    if node.control_state is None:
        missing.append("控制状态 /control/state")
    for side, label in (("left", "左手"), ("right", "右手")):
        if side in node.latest:
            continue
        message = node.hand_seen.get(side)
        if message is None:
            missing.append(f"{label}状态（未收到 ROS 消息）")
        elif not message.connected:
            missing.append(f"{label}状态（未连接：{message.fault_reason or '原因未知'}）")
        elif message.fault:
            missing.append(f"{label}状态（故障：{message.fault_reason or '原因未知'}）")
        else:
            missing.append(f"{label}状态（关节角无效）")
    return "、".join(missing)


def _publish_home_authorization_and_request(
    node: _Preflight,
    *,
    rdk_socket: Path,
    session_id: str,
    timeout_s: float,
    clear_hold_latched: bool,
) -> dict[str, object]:
    """Authorize and supervise Home after a successful F/T-zero transaction."""

    # Invoking the local Reset command is the operator's permission for this
    # reset motion and assertion that its configured Home path is clear.
    permission_publisher = node.create_publisher(
        Bool, "/control/local_permission", 1
    )
    collision_publisher = node.create_publisher(Bool, "/safety/collision_clear", 1)
    gate_deadline = time.monotonic() + 2.0
    while time.monotonic() < gate_deadline and (
        permission_publisher.get_subscription_count() == 0
        or collision_publisher.get_subscription_count() == 0
    ):
        rclpy.spin_once(node, timeout_sec=0.05)
    if permission_publisher.get_subscription_count() == 0:
        raise RuntimeError("Reset permission subscriber is unavailable")
    if collision_publisher.get_subscription_count() == 0:
        raise RuntimeError("Reset collision-clear subscriber is unavailable")
    reset_gate = Bool()
    reset_gate.data = True
    for _ in range(3):
        permission_publisher.publish(reset_gate)
        collision_publisher.publish(reset_gate)
        rclpy.spin_once(node, timeout_sec=0.05)

    authorization_publisher = node.create_publisher(
        String, "/control/local_home_authorization", 1
    )
    request_publisher = node.create_publisher(Empty, "/control/home_request", 1)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and (
        authorization_publisher.get_subscription_count() == 0
        or request_publisher.get_subscription_count() == 0
    ):
        rclpy.spin_once(node, timeout_sec=0.05)
    if authorization_publisher.get_subscription_count() == 0:
        raise RuntimeError("Home authorization subscriber is unavailable")
    if request_publisher.get_subscription_count() == 0:
        raise RuntimeError("Home request subscriber is unavailable")

    terminal = {"complete", "failed", "rejected", "authorization_rejected"}
    for attempt in range(2):
        client = RDKIPCClient(rdk_socket)
        try:
            kind, authorization = client.request(
                "authorize_home",
                {
                    "session_id": session_id,
                    "operator_confirmation": "FLEXIV-HOME-MOVE",
                    "clear_hold_latched": clear_hold_latched,
                    # A valid F/T zero may be reused after a controller Minor
                    # fault.  In that path no new ZeroFT transaction runs, so
                    # Reset must explicitly perform the same
                    # ClearFault -> Enable -> operational sequence used by the
                    # known-good one-shot reset implementation before Home.
                    "recover_robot_faults": clear_hold_latched,
                },
                timeout_s=30.0,
            )
        finally:
            client.close()
        if kind != "authorize_home_result" or not authorization.get(
            "authorized", False
        ):
            raise RuntimeError(authorization.get("reason", kind))

        outgoing = String()
        outgoing.data = json.dumps(
            {
                "session_id": session_id,
                "one_time_token": authorization["one_time_token"],
                "expires_monotonic_ns": authorization["expires_monotonic_ns"],
                "clear_hold_latched": clear_hold_latched,
                "recover_robot_faults": clear_hold_latched,
            },
            separators=(",", ":"),
        )
        node.home_status = None
        for _ in range(3):
            authorization_publisher.publish(outgoing)
            rclpy.spin_once(node, timeout_sec=0.05)
        request_publisher.publish(Empty())

        _spin_until(
            node,
            lambda: node.home_status is not None
            and str(node.home_status.get("state", "")) in terminal,
            timeout_s + 10.0,
        )
        assert node.home_status is not None
        if node.home_status.get("state") == "complete":
            return node.home_status
        reason = str(node.home_status.get("reason", "Home did not complete"))
        stale_lease = "authorization is missing, expired or consumed" in reason
        if attempt == 0 and stale_lease:
            print("Home authorization lease was stale; retrying once", flush=True)
            continue
        raise RuntimeError(
            reason or str(node.home_status.get("state", "failed"))
        )
    raise RuntimeError("Home authorization retry exhausted")


def _cycle_inspire_hands(
    node: _Preflight, timeout_s: float = 20.0
) -> dict[str, object]:
    """Request the DFTP-owned open-close-open Reset motion."""

    if not node.hand_reset.wait_for_service(timeout_sec=5.0):
        raise RuntimeError("Inspire hand Reset service is unavailable")
    future = node.hand_reset.call_async(Trigger.Request())
    _spin_until(node, future.done, timeout_s)
    result = future.result()
    if result is None or not result.success:
        reason = "no response" if result is None else str(result.message)
        raise RuntimeError("Inspire hand Reset failed: " + reason)
    return {
        "state": "complete",
        "sequence": "open-close-open",
        "final_target": "open",
        "message": str(result.message),
    }


def _ft_zero_mode(state_name: str, ft_zeroed_for_session: bool) -> str:
    """Choose an idempotent Reset path from the current control state."""

    normalized = str(state_name).upper()
    if normalized == "MAINTENANCE":
        return "execute"
    # An armed source still has a valid F/T-zero generation; guarded Home
    # already performs hold -> release source ownership -> READY before moving.
    # Requiring the operator to disarm separately only makes the one-shot Reset
    # fail before it reaches that existing transition.
    if ft_zeroed_for_session and normalized in {
        "READY",
        "TELEOP_ARMED",
        "POLICY_ARMED",
        "REPLAY_ARMED",
        "HOLD_LATCHED",
        "FAULT",
    }:
        return "reuse"
    raise RuntimeError(
        "Reset requires MAINTENANCE, or READY/ARMED/HOLD_LATCHED/FAULT with a valid "
        "session F/T zero; "
        f"got {normalized or 'UNKNOWN'}"
    )


def _wait_for_ready_before_home(zero_mode: str) -> bool:
    """A reused ARMED zero reaches READY inside guarded Home, not before it."""

    if zero_mode not in {"execute", "reuse"}:
        raise ValueError(f"unsupported F/T-zero mode: {zero_mode}")
    return zero_mode == "execute"


def _skip_hand_preview(
    zero_mode: str,
    *,
    requested: bool,
    execution_confirmed: bool,
) -> bool:
    """Skip only the preview that cannot affect an already-valid F/T zero."""

    if zero_mode not in {"execute", "reuse"}:
        raise ValueError(f"unsupported F/T-zero mode: {zero_mode}")
    return requested and execution_confirmed and zero_mode == "reuse"


def _execute_ft_zero(
    node: _Preflight,
    *,
    rdk_socket: Path,
    digest: str,
    session_id: str,
    preview: dict[str, dict[str, object]],
) -> bool:
    client = RDKIPCClient(rdk_socket)
    try:
        kind, authorization = client.request(
            "authorize_zero_ft",
            {
                "session_id": session_id,
                "operator_confirmation": "FLEXIV-FT-UNLOADED",
                "tool_payload_config_hash": digest,
            },
        )
    finally:
        client.close()
    if kind != "authorize_zero_ft_result" or not authorization.get(
        "authorized", False
    ):
        raise RuntimeError(authorization.get("reason", kind))
    if not node.action.wait_for_server(timeout_sec=5.0):
        raise RuntimeError("zero_ft Action server is unavailable")
    goal = ZeroFTSensors.Goal()
    goal.session_id = session_id
    goal.operator_confirmation = "FLEXIV-FT-UNLOADED"
    goal.local_console = True
    goal.tool_payload_config_hash = digest
    goal.local_authorization_token = authorization["one_time_token"]
    goal.left_hand_position = preview["left"]["latest"]
    goal.right_hand_position = preview["right"]["latest"]
    future = node.action.send_goal_async(
        goal,
        feedback_callback=lambda item: print(
            f"{item.feedback.phase}: {item.feedback.status}"
        ),
    )
    _spin_until(node, future.done, 5.0)
    handle = future.result()
    if not handle.accepted:
        raise RuntimeError("zero_ft Action goal was rejected")
    result_future = handle.get_result_async()
    _spin_until(node, result_future.done, 180.0)
    result = result_future.result().result
    print(
        json.dumps(
            {
                "success": result.success,
                "final_state": result.final_state,
                "event_id": result.event_id,
                "failure_reason": result.failure_reason,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return bool(result.success)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rdk-socket", type=Path, required=True)
    parser.add_argument("--tool-payload-config", type=Path, required=True)
    parser.add_argument("--preview-seconds", type=float, default=2.0)
    parser.add_argument("--hand-warmup-seconds", type=float, default=1.0)
    parser.add_argument(
        "--skip-hand-preview",
        action="store_true",
        help="use the latest valid hand state without a stationary preview window",
    )
    parser.add_argument("--skip-preview-if-ft-zeroed", action="store_true")
    parser.add_argument(
        "--max-hand-delta",
        type=float,
        default=5.0,
        help="maximum stationary Inspire angle variation in 0..1000 raw counts",
    )
    parser.add_argument("--confirm-ft-unloaded", default="")
    parser.add_argument("--home-after-zero", action="store_true")
    parser.add_argument("--cycle-hands-after-home", action="store_true")
    parser.add_argument("--home-timeout", type=float, default=20.0)
    parser.add_argument("--clear-home-hold-latched", action="store_true")
    args = parser.parse_args(argv)
    if not sys.stdin.isatty():
        raise SystemExit("zero_ft_local requires a same-host interactive TTY")
    digest = _canonical_hash(args.tool_payload_config.resolve(strict=True))
    rclpy.init(args=[])
    node = _Preflight()
    try:
        try:
            _spin_until(
                node,
                lambda: node.control_state is not None
                and all(side in node.latest for side in ("left", "right")),
                15.0,
            )
        except TimeoutError as exc:
            raise RuntimeError(
                "Reset 等待状态超时：" + _initial_state_problem(node)
            ) from exc
        assert node.control_state is not None
        execution_confirmed = (
            args.confirm_ft_unloaded == "FLEXIV-FT-UNLOADED"
        )
        zero_mode: str | None = None
        try:
            zero_mode = _ft_zero_mode(
                node.control_state.state_name,
                bool(node.control_state.ft_zeroed_for_session),
            )
        except RuntimeError:
            # Keep the read-only preflight useful for reporting an unavailable
            # state; a confirmed Reset still fails below before any write.
            if execution_confirmed:
                raise
        skip_preview = bool(args.skip_hand_preview) or (
            zero_mode is not None
            and _skip_hand_preview(
                zero_mode,
                requested=bool(args.skip_preview_if_ft_zeroed),
                execution_confirmed=execution_confirmed,
            )
        )
        preview: dict[str, dict[str, object]] = {}
        if skip_preview:
            for side in ("left", "right"):
                latest = np.asarray(node.latest[side].angle, dtype=np.float64)
                preview[side] = {
                    "sample_count": 1,
                    "latest": latest.tolist(),
                    "preview_skipped": True,
                }
        else:
            warmup_deadline = time.monotonic() + max(
                0.0, args.hand_warmup_seconds
            )
            while time.monotonic() < warmup_deadline:
                rclpy.spin_once(node, timeout_sec=0.02)
            node.samples = {"left": [], "right": []}
            deadline = time.monotonic() + args.preview_seconds
            while time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.02)
            for side in ("left", "right"):
                if len(node.samples[side]) < max(
                    2, int(args.preview_seconds * 20)
                ):
                    raise RuntimeError(
                        f"{side} hand preview has too few fresh samples"
                    )
                matrix = np.stack(node.samples[side])
                delta = float(np.max(np.ptp(matrix, axis=0)))
                if delta > args.max_hand_delta:
                    raise RuntimeError(
                        f"{side} hand moved during preview: {delta}"
                    )
                preview[side] = {
                    "sample_count": int(matrix.shape[0]),
                    "mean": np.mean(matrix, axis=0).tolist(),
                    "standard_deviation": np.std(matrix, axis=0).tolist(),
                    "max_delta": delta,
                    "latest": matrix[-1].tolist(),
                    "preview_skipped": False,
                }
        planned_ft_zero = (
            "reuse-current-session"
            if zero_mode == "reuse"
            else zero_mode or "unavailable"
        )
        report = {
            "tool_payload_config": str(args.tool_payload_config.resolve()),
            "canonical_sha256": digest,
            "session_id": node.control_state.session_id,
            "control_state": node.control_state.state_name,
            "ft_zero_action": planned_ft_zero,
            "hand_preview_skipped": skip_preview,
            "hands": preview,
            "will_execute": execution_confirmed,
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        if not execution_confirmed:
            print(
                "Preflight only. No F/T zero requested. Re-run with "
                "--confirm-ft-unloaded FLEXIV-FT-UNLOADED after local inspection."
            )
            return 0
        assert zero_mode is not None
        if zero_mode == "execute":
            if not _execute_ft_zero(
                node,
                rdk_socket=args.rdk_socket,
                digest=digest,
                session_id=node.control_state.session_id,
                preview=preview,
            ):
                return 2
        else:
            print(
                json.dumps(
                    {
                        "success": True,
                        "final_state": "READY",
                        "event_id": "",
                        "failure_reason": "",
                        "ft_zero": "reused-current-session",
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        if args.cycle_hands_after_home and not args.home_after_zero:
            raise RuntimeError("hand Reset requires --home-after-zero")
        if args.home_after_zero:
            # A new F/T-zero transaction publishes READY asynchronously, so
            # wait for that transition.  When reusing a valid zero from an
            # ARMED state, guarded Home itself performs hold -> READY; waiting
            # here would deadlock before the Home request that causes it.
            if _wait_for_ready_before_home(zero_mode):
                _spin_until(
                    node,
                    lambda: node.control_state is not None
                    and node.control_state.state_name == "READY",
                    5.0,
                )
            home = _publish_home_authorization_and_request(
                node,
                rdk_socket=args.rdk_socket,
                session_id=node.control_state.session_id,
                timeout_s=args.home_timeout,
                clear_hold_latched=args.clear_home_hold_latched,
            )
            hands = None
            if args.cycle_hands_after_home:
                print(
                    "Home: 双臂已完成；正在执行双手张开/闭合/张开（最长 20 秒）",
                    flush=True,
                )
                hands = _cycle_inspire_hands(node)
                print("Hands: 双手 Reset 已完成，最终目标为张开", flush=True)
            reset_report = {
                "reset_success": True,
                "ft_zero": (
                    "complete" if zero_mode == "execute" else "reused"
                ),
                "home": home,
            }
            if hands is not None:
                reset_report["hands"] = hands
            print(
                json.dumps(
                    reset_report,
                    indent=2,
                    sort_keys=True,
                )
            )
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
