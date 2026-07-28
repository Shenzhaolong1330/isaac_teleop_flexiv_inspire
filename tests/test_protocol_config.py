from __future__ import annotations

import copy

import pytest

from flexiv_inspire_isaac.config import BridgeConfig
from flexiv_inspire_isaac.mapping import action_from_vectors
from flexiv_inspire_isaac.protocol import (
    decode_packet,
    encode_packet,
    validate_action,
)

from conftest import make_config


def test_protocol_round_trip_and_action_validation() -> None:
    cfg = make_config()
    action = action_from_vectors(
        [0.001, 0.0, 0.0],
        [0.0, 0.002, 0.0],
        [-0.001, 0.0, 0.0],
        [0.0, -0.002, 0.0],
    )
    packet = {
        "kind": "action",
        "session_id": "test",
        "sequence": 1,
        "epoch": 1,
        "sent_monotonic_ns": 123,
        "deadman": True,
        "action": action,
    }
    decoded = decode_packet(encode_packet(packet))
    clean = validate_action(
        decoded["action"],
        max_translation_step_m=cfg.safety.max_translation_step_m,
        max_rotation_step_rad=cfg.safety.max_rotation_step_rad,
    )
    assert clean == action


def test_partial_hand_or_arm_schema_is_not_synthesized() -> None:
    cfg = make_config()
    action = action_from_vectors([0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0])
    del action["right_delta_ee_pose.rz"]
    with pytest.raises(ValueError, match="missing"):
        validate_action(
            action,
            max_translation_step_m=cfg.safety.max_translation_step_m,
            max_rotation_step_rad=cfg.safety.max_rotation_step_rad,
        )


def test_reflection_axis_mapping_is_rejected() -> None:
    raw = {
        "mapping": {
            "left": {
                "pose_index": 0,
                "axis_rotation": [[-1, 0, 0], [0, 1, 0], [0, 0, 1]],
            },
            "right": {"pose_index": 1},
        },
        "safety": {},
        "deadman": {},
        "ros": {},
        "ipc": {},
        "existing_stack": {},
    }
    with pytest.raises(ValueError, match="proper rotation"):
        BridgeConfig.from_dict(raw)


def test_bridge_step_cannot_exceed_existing_wrapper_limit() -> None:
    raw = {
        "mapping": {
            "left": {"pose_index": 0},
            "right": {"pose_index": 1},
        },
        "safety": {"max_translation_step_m": 0.021},
        "deadman": {},
        "ros": {},
        "ipc": {},
        "existing_stack": {},
    }
    with pytest.raises(ValueError, match="0.02"):
        BridgeConfig.from_dict(raw)

