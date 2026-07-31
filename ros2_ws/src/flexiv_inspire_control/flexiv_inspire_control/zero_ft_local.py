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
        self.control_state: ControlState | None = None
        for side in ("left", "right"):
            self.create_subscription(
                HandState,
                f"/robot/{side}_hand/state",
                lambda message, selected=side: self._hand(selected, message),
                20,
            )
        self.create_subscription(ControlState, "/control/state", self._state, 10)
        self.action = ActionClient(
            self, ZeroFTSensors, "/maintenance/zero_ft_sensors"
        )

    def _hand(self, side: str, message: HandState) -> None:
        if message.connected and not message.fault:
            values = np.asarray(message.angle, dtype=np.float64)
            if values.shape == (6,) and np.all(np.isfinite(values)):
                self.latest[side] = message
                self.samples[side].append(values)

    def _state(self, message: ControlState) -> None:
        self.control_state = message


def _spin_until(node: Node, predicate, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out waiting for local ROS state")
        rclpy.spin_once(node, timeout_sec=0.05)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rdk-socket", type=Path, required=True)
    parser.add_argument("--tool-payload-config", type=Path, required=True)
    parser.add_argument("--preview-seconds", type=float, default=2.0)
    parser.add_argument("--max-hand-delta", type=float, default=1.0)
    parser.add_argument("--confirm-ft-unloaded", default="")
    args = parser.parse_args(argv)
    if not sys.stdin.isatty():
        raise SystemExit("zero_ft_local requires a same-host interactive TTY")
    digest = _canonical_hash(args.tool_payload_config.resolve(strict=True))
    rclpy.init(args=None)
    node = _Preflight()
    try:
        _spin_until(
            node,
            lambda: node.control_state is not None
            and all(side in node.latest for side in ("left", "right")),
            5.0,
        )
        node.samples = {"left": [], "right": []}
        deadline = time.monotonic() + args.preview_seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.02)
        preview: dict[str, dict[str, object]] = {}
        for side in ("left", "right"):
            if len(node.samples[side]) < max(2, int(args.preview_seconds * 20)):
                raise RuntimeError(f"{side} hand preview has too few fresh samples")
            matrix = np.stack(node.samples[side])
            delta = float(np.max(np.ptp(matrix, axis=0)))
            if delta > args.max_hand_delta:
                raise RuntimeError(f"{side} hand moved during preview: {delta}")
            preview[side] = {
                "sample_count": int(matrix.shape[0]),
                "mean": np.mean(matrix, axis=0).tolist(),
                "standard_deviation": np.std(matrix, axis=0).tolist(),
                "max_delta": delta,
                "latest": matrix[-1].tolist(),
            }
        assert node.control_state is not None
        report = {
            "tool_payload_config": str(args.tool_payload_config.resolve()),
            "canonical_sha256": digest,
            "session_id": node.control_state.session_id,
            "control_state": node.control_state.state_name,
            "hands": preview,
            "will_execute": args.confirm_ft_unloaded == "FLEXIV-FT-UNLOADED",
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        if args.confirm_ft_unloaded != "FLEXIV-FT-UNLOADED":
            print(
                "Preflight only. No F/T zero requested. Re-run with "
                "--confirm-ft-unloaded FLEXIV-FT-UNLOADED after local inspection."
            )
            return 0
        if node.control_state.state_name != "MAINTENANCE":
            raise RuntimeError("control bridge must be in MAINTENANCE")
        client = RDKIPCClient(args.rdk_socket)
        try:
            kind, authorization = client.request(
                "authorize_zero_ft",
                {
                    "session_id": node.control_state.session_id,
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
        goal.session_id = node.control_state.session_id
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
        return 0 if result.success else 2
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
