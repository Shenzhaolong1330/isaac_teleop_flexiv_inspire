from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from flexiv_inspire_isaac.policy_api.models import capabilities_v1
from policy_contracts import (
    ActionMappingRegistry,
    ActionSchema,
    ChannelDescriptor,
    DUAL_ARM_LEROBOT_V1_PROFILE,
    FeatureContract,
    FeatureContractError,
    JOINT_MINIMAL_PROFILE,
    SystemSchema,
    TensorDescriptor,
    cartesian_minimal_state,
    flexiv_inspire_action_mappings,
    joint_minimal_state,
    legacy_state38,
    matrix_to_rotvec,
    native_action30_to_policy24,
    policy_action24_to_native30,
    quaternion_xyzw_to_matrix,
    rotation6d_to_matrix,
    rotvec_to_matrix,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "policy_contracts"
IDENTITY_ROT6D = np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])


def _native_action() -> np.ndarray:
    action = np.zeros(30, dtype=np.float64)
    action[0:3] = [0.01, -0.02, 0.03]
    action[3:9] = [1.0, 0.0, 0.0, 0.0, np.cos(0.2), np.sin(0.2)]
    action[9:12] = [-0.03, 0.02, -0.01]
    action[12:18] = [np.cos(0.1), np.sin(0.1), 0.0, -np.sin(0.1), np.cos(0.1), 0.0]
    action[18:30] = np.linspace(0.0, 1000.0, 12)
    return action


def test_legacy_feature_fixture_is_exact_and_loadable() -> None:
    contract = FeatureContract.from_info(FIXTURES / "dual_arm_legacy_info.json")

    contract.validate_profile(DUAL_ARM_LEROBOT_V1_PROFILE)
    assert contract.fps == 30.0
    assert contract.robot_type == "flexiv_dual_arm"
    assert contract.feature("observation.state").shape == (38,)
    assert contract.feature("action").shape == (24,)


def test_native_fixture_locks_current_30d_wire_order() -> None:
    contract = FeatureContract.from_info(FIXTURES / "native_sent_command_info.json")
    capabilities = capabilities_v1()

    assert contract.feature("action").shape == (30,)
    assert contract.feature("action").names[3:9] == (
        "left_dR00", "left_dR10", "left_dR20",
        "left_dR01", "left_dR11", "left_dR21",
    )
    assert capabilities["default_action_dimension"] == 30
    assert capabilities["rotation_element_order"] == [
        "R00", "R10", "R20", "R01", "R11", "R21"
    ]


def test_full_lerobot_conversion_does_not_enable_new_profile() -> None:
    config = yaml.safe_load(
        (ROOT / "config" / "conversion_lerobot_full.yaml").read_text()
    )
    export = config["lerobot_export"]

    assert "profile" not in export
    assert export["action"]["view"] == "sent_command"
    assert "observation.arm_pose" in export["fields"]


def test_default_conversion_targets_rl100_zarr() -> None:
    config = yaml.safe_load((ROOT / "config" / "conversion.yaml").read_text())

    assert config["output"]["format"] == "rl100_zarr"
    assert config["rl100_zarr_export"]["profile"] == "joint_proprio_cartesian_v1"
    assert config["rl100_zarr_export"]["timeline"]["fps"] == 30.0


@pytest.mark.parametrize(
    "rotvec",
    (
        [0.0, 0.0, 0.0],
        [0.1, -0.2, 0.3],
        [-0.7, 0.2, 1.1],
        [np.pi - 1e-5, 0.0, 0.0],
    ),
)
def test_so3_rotvec_roundtrip(rotvec) -> None:
    matrix = rotvec_to_matrix(rotvec)
    recovered = matrix_to_rotvec(matrix)

    assert np.allclose(rotvec_to_matrix(recovered), matrix, atol=1e-7)


def test_xyzw_quaternion_is_normalized_and_converted_to_so3() -> None:
    matrix = quaternion_xyzw_to_matrix([0.0, 0.0, 2.0, 2.0])

    assert np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-8)
    assert np.isclose(np.linalg.det(matrix), 1.0)
    assert np.allclose(matrix @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])


def test_native_and_policy_action_roundtrip_preserves_se3_and_hands() -> None:
    native = _native_action()
    policy = native_action30_to_policy24(native)
    recovered = policy_action24_to_native30(policy)

    assert policy.shape == (24,)
    assert recovered.shape == (30,)
    assert np.allclose(recovered[0:3], native[0:3])
    assert np.allclose(recovered[9:12], native[9:12])
    assert np.allclose(
        rotation6d_to_matrix(recovered[3:9]),
        rotation6d_to_matrix(native[3:9]),
        atol=1e-7,
    )
    assert np.allclose(
        rotation6d_to_matrix(recovered[12:18]),
        rotation6d_to_matrix(native[12:18]),
        atol=1e-7,
    )
    assert np.allclose(recovered[18:30], native[18:30])


def test_action_mapping_registry_supports_multiple_contracts_without_transport() -> None:
    registry = flexiv_inspire_action_mappings()
    policy = native_action30_to_policy24(_native_action())

    mapped_policy = registry.map("cartesian_delta_rotvec_v1", policy)
    mapped_native = registry.map("flexiv_inspire_native_rot6d_v1", _native_action())

    assert registry.schema_ids == (
        "cartesian_delta_rotvec_v1",
        "flexiv_inspire_native_rot6d_v1",
    )
    assert np.allclose(mapped_policy, policy_action24_to_native30(policy))
    assert np.allclose(mapped_native, _native_action())


def test_custom_action_mapping_registry_is_dimension_checked() -> None:
    registry = ActionMappingRegistry(canonical_dimension=3).register(
        "test_delta_v1",
        input_dimension=2,
        transform=lambda value: (value[0], value[1], 0.0),
    )

    assert np.allclose(registry.map("test_delta_v1", [1.0, 2.0]), [1.0, 2.0, 0.0])
    with pytest.raises(ValueError, match="2 finite"):
        registry.map("test_delta_v1", [1.0])
    with pytest.raises(ValueError, match="no canonical mapping"):
        registry.map("unknown", [1.0, 2.0])


def test_state_profiles_do_not_duplicate_pose_representations() -> None:
    q = np.arange(14, dtype=np.float64) / 10.0
    pose = np.concatenate(
        ([0.1, 0.2, 0.3], IDENTITY_ROT6D, [-0.1, -0.2, 0.4], IDENTITY_ROT6D)
    )
    hands = np.linspace(0.0, 1000.0, 12)

    assert legacy_state38(q, pose, hands).shape == (38,)
    assert joint_minimal_state(q, hands).shape == (26,)
    assert cartesian_minimal_state(pose, hands).shape == (24,)
    assert all("ee_pose" not in name for name in JOINT_MINIMAL_PROFILE.state_names)
    assert JOINT_MINIMAL_PROFILE.state_dimension == 26
    assert JOINT_MINIMAL_PROFILE.action_dimension == 24


def test_feature_contract_rejects_unrequested_observation_fields() -> None:
    info = yaml.safe_load((FIXTURES / "dual_arm_legacy_info.json").read_text())
    info["features"]["observation.arm_pose"] = {
        "dtype": "float32", "shape": [18], "names": None
    }

    with pytest.raises(FeatureContractError, match="extras"):
        FeatureContract.from_info(info).validate_profile(DUAL_ARM_LEROBOT_V1_PROFILE)


def test_system_schema_hash_is_order_stable_and_content_sensitive() -> None:
    channel = ChannelDescriptor(
        channel_id="arm.left.q",
        semantic="joint_position",
        tensor=TensorDescriptor("float64", (7,), tuple(f"j{i}" for i in range(7)), "rad"),
        native_rate_hz=200.0,
        frame_id="left_base",
    )
    action = ActionSchema(
        schema_id="cartesian_delta_rotvec_v1",
        tensor=TensorDescriptor("float32", (24,), JOINT_MINIMAL_PROFILE.action_names),
        frame_id="world",
        representation="delta_xyz_rotvec_hand",
        rate_hz=30.0,
        relative=True,
    )
    first = SystemSchema(2, "station", "flexiv_inspire", (channel,), (action,), {"b": "2", "a": "1"})
    second = SystemSchema(2, "station", "flexiv_inspire", (channel,), (action,), {"a": "1", "b": "2"})

    assert first.schema_hash == second.schema_hash
    changed = SystemSchema(2, "station", "flexiv_inspire", (channel,), (), {"a": "1", "b": "2"})
    assert changed.schema_hash != first.schema_hash
