from __future__ import annotations

import json

import numpy as np
import pytest

from flexiv_inspire_isaac.policy_api.data_server import system_description_message
from flexiv_inspire_isaac.policy_api.generated import policy_data_v2_pb2 as pb
from flexiv_inspire_isaac.policy_api.lease import LocalControlState
from flexiv_inspire_isaac.policy_api.profile_runtime import (
    ProfileRuntimeError,
    ProfileSnapshotMapper,
)
from flexiv_inspire_isaac.policy_api.system_schema import build_system_schema
from policy_contracts import get_profile


def _state() -> LocalControlState:
    return LocalControlState(
        session_id="runtime-session",
        ft_zeroed=True,
        local_policy_authorized=False,
        pedal_valid=False,
        arms_online=True,
        hands_online=True,
        state="READY",
    )


def _description():
    return system_description_message(
        build_system_schema(image_shape=(2, 3, 3)), _state()
    )


def _tensor(channel_id: str, value) -> pb.SampleEnvelope:
    array = np.asarray(value, dtype="<f8")
    return pb.SampleEnvelope(
        schema_version=2,
        schema_hash=_description().schema_hash,
        channel_id=channel_id,
        timing=pb.SampleMetadata(valid=True, timing_valid=True, sequence=1),
        tensor=pb.TensorPayload(
            dtype=pb.FLOAT64,
            shape=array.shape,
            data=array.tobytes(),
        ),
    )


def _image(channel_id: str, value: int) -> pb.SampleEnvelope:
    image = np.full((2, 3, 3), value, dtype=np.uint8)
    return pb.SampleEnvelope(
        schema_version=2,
        schema_hash=_description().schema_hash,
        channel_id=channel_id,
        timing=pb.SampleMetadata(valid=True, timing_valid=True, sequence=1),
        image=pb.ImagePayload(
            encoding="rgb8",
            data=image.tobytes(),
            width=3,
            height=2,
            channels=3,
        ),
    )


def _snapshot(mapper: ProfileSnapshotMapper) -> pb.Snapshot:
    values = {
        "arm.left.q": np.arange(1, 8),
        "arm.right.q": np.arange(8, 15),
        "arm.left.tcp_pose": (0.1, 0.2, 0.3, 0, 0, 0, 1),
        "arm.right.tcp_pose": (-0.1, -0.2, -0.3, 0, 0, 0, 1),
        "hand.left.angle": np.full(6, 250),
        "hand.right.angle": np.full(6, 750),
    }
    snapshot = pb.Snapshot(
        schema_version=2,
        schema_hash=mapper.schema_hash,
        session_id=mapper.session_id,
        complete=True,
    )
    for channel_id in mapper.required_channels:
        sample = (
            _image(channel_id, 17)
            if channel_id.startswith("camera.")
            else _tensor(channel_id, values[channel_id])
        )
        sample.schema_hash = mapper.schema_hash
        snapshot.samples.add().CopyFrom(sample)
    return snapshot


def test_runtime_mapper_produces_exact_legacy_policy_frame(tmp_path):
    profile = get_profile("dual_arm_lerobot_v1")
    info = tmp_path / "info.json"
    info.write_text(
        json.dumps(
            {
                "fps": 15,
                "robot_type": "flexiv_dual_arm",
                "features": {
                    "observation.state": {
                        "dtype": "float32",
                        "shape": [38],
                        "names": list(profile.state_names),
                    },
                    "action": {
                        "dtype": "float32",
                        "shape": [24],
                        "names": list(profile.action_names),
                    },
                    **{
                        key: {
                            "dtype": "video",
                            "shape": [2, 3, 3],
                            "names": ["height", "width", "channel"],
                        }
                        for key in profile.image_keys
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    mapper = ProfileSnapshotMapper(
        _description(),
        profile_id=profile.profile_id,
        feature_contract_info=info,
    )

    frame = mapper.frame_from_snapshot(_snapshot(mapper))

    assert set(frame) == {"observation.state", *profile.image_keys}
    state = frame["observation.state"]
    assert state.dtype == np.float32
    assert state.shape == (38,)
    assert state[:7].tolist() == list(range(1, 8))
    assert np.allclose(state[13:19], 0.25)
    assert state[19:26].tolist() == list(range(8, 15))
    assert np.allclose(state[32:38], 0.75)
    assert frame["observation.images.head_image"].shape == (2, 3, 3)


def test_runtime_mapper_minimal_state_does_not_request_tcp_pose():
    mapper = ProfileSnapshotMapper(
        _description(), profile_id="joint_proprio_cartesian_v1"
    )

    frame = mapper.frame_from_snapshot(_snapshot(mapper))

    assert frame["observation.state"].shape == (26,)
    assert not any("tcp_pose" in item for item in mapper.required_channels)


def test_runtime_mapper_right_profile_requests_only_right_and_two_rgb():
    mapper = ProfileSnapshotMapper(
        _description(), profile_id="right_joint_proprio_cartesian_v1"
    )

    frame = mapper.frame_from_snapshot(_snapshot(mapper))

    assert frame["observation.state"].shape == (13,)
    assert set(frame) == {
        "observation.state",
        "observation.images.head_image",
        "observation.images.right_wrist_image",
    }
    assert not any("left" in item for item in mapper.required_channels)


def test_runtime_mapper_fails_closed_on_incomplete_snapshot():
    mapper = ProfileSnapshotMapper(
        _description(), profile_id="joint_proprio_cartesian_v1"
    )
    snapshot = _snapshot(mapper)
    snapshot.complete = False
    snapshot.errors.append("camera.head.rgb:stale")

    with pytest.raises(ProfileRuntimeError, match="stale"):
        mapper.frame_from_snapshot(snapshot)


def test_runtime_mapper_rejects_checkpoint_image_shape_mismatch(tmp_path):
    profile = get_profile("joint_proprio_cartesian_v1")
    info = tmp_path / "info.json"
    info.write_text(
        json.dumps(
            {
                "fps": 15,
                "robot_type": "test",
                "features": {
                    "observation.state": {
                        "dtype": "float32",
                        "shape": [26],
                        "names": list(profile.state_names),
                    },
                    "action": {
                        "dtype": "float32",
                        "shape": [24],
                        "names": list(profile.action_names),
                    },
                    **{
                        key: {
                            "dtype": "video",
                            "shape": [4, 5, 3],
                            "names": ["height", "width", "channel"],
                        }
                        for key in profile.image_keys
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ProfileRuntimeError, match="differs from checkpoint"):
        ProfileSnapshotMapper(
            _description(),
            profile_id=profile.profile_id,
            feature_contract_info=info,
        )
