"""Synchronous, read-only runtime client for strict policy profiles.

LeRobot robot implementations are synchronous, while the service also offers
an async streaming API.  This small client deliberately uses ``GetSnapshot``
so one policy step receives one causal, schema-checked observation.  It does
not acquire a lease or expose an action RPC; P3 is shadow inference only.
"""

from __future__ import annotations

from io import BytesIO
import ipaddress
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image

from policy_contracts import (
    FeatureContract,
    cartesian_minimal_state,
    get_profile,
    joint_minimal_state,
    legacy_state38,
    matrix_to_rotation6d,
    quaternion_xyzw_to_matrix,
)

from .data_client import decode_tensor
from .generated import policy_data_v2_pb2 as pb
from .generated import policy_data_v2_pb2_grpc as pb_grpc


class ProfileRuntimeError(RuntimeError):
    pass


PROFILE_CHANNELS = {
    "dual_arm_lerobot_v1": (
        "arm.left.q",
        "arm.right.q",
        "arm.left.tcp_pose",
        "arm.right.tcp_pose",
        "hand.left.angle",
        "hand.right.angle",
        "camera.left_wrist.rgb",
        "camera.right_wrist.rgb",
        "camera.head.rgb",
    ),
    "joint_proprio_cartesian_v1": (
        "arm.left.q",
        "arm.right.q",
        "hand.left.angle",
        "hand.right.angle",
        "camera.left_wrist.rgb",
        "camera.right_wrist.rgb",
        "camera.head.rgb",
    ),
    "cartesian_proprio_v1": (
        "arm.left.tcp_pose",
        "arm.right.tcp_pose",
        "hand.left.angle",
        "hand.right.angle",
        "camera.left_wrist.rgb",
        "camera.right_wrist.rgb",
        "camera.head.rgb",
    ),
}

PROFILE_IMAGE_CHANNELS = {
    "observation.images.left_wrist_image": "camera.left_wrist.rgb",
    "observation.images.right_wrist_image": "camera.right_wrist.rgb",
    "observation.images.head_image": "camera.head.rgb",
}


def _descriptor_map(description) -> dict[str, object]:
    result = {item.channel_id: item for item in description.channels}
    if len(result) != len(description.channels):
        raise ProfileRuntimeError("system description contains duplicate channels")
    return result


def _require_descriptor(
    descriptors: Mapping[str, object],
    channel_id: str,
    *,
    shape: tuple[int, ...] | None,
    semantic: str,
) -> object:
    try:
        descriptor = descriptors[channel_id]
    except KeyError as exc:
        raise ProfileRuntimeError(f"RPC channel is missing: {channel_id}") from exc
    if shape is not None and tuple(descriptor.tensor.shape) != shape:
        raise ProfileRuntimeError(
            f"RPC channel {channel_id} shape changed: "
            f"{tuple(descriptor.tensor.shape)} != {shape}"
        )
    if descriptor.semantic != semantic:
        raise ProfileRuntimeError(
            f"RPC channel {channel_id} semantic changed: {descriptor.semantic}"
        )
    return descriptor


def _decode_image(payload, expected_shape: tuple[int, int, int]) -> np.ndarray:
    height, width, channels = expected_shape
    actual = (int(payload.height), int(payload.width), int(payload.channels))
    if actual != expected_shape:
        raise ProfileRuntimeError(
            f"RPC image dimensions changed: {actual} != {expected_shape}"
        )
    encoding = str(payload.encoding).lower()
    if encoding in {"jpeg", "jpg"}:
        image = np.asarray(Image.open(BytesIO(payload.data)).convert("RGB"))
    elif encoding in {"rgb", "rgb8"}:
        expected_bytes = height * width * channels
        if len(payload.data) != expected_bytes:
            raise ProfileRuntimeError(
                f"RPC RGB payload has {len(payload.data)} bytes, "
                f"expected {expected_bytes}"
            )
        image = np.frombuffer(payload.data, dtype=np.uint8).reshape(expected_shape)
    else:
        raise ProfileRuntimeError(f"unsupported RPC image encoding: {encoding}")
    if image.shape != expected_shape or image.dtype != np.uint8:
        raise ProfileRuntimeError("decoded RPC image shape/dtype is invalid")
    result = np.asarray(image)
    result.setflags(write=False)
    return result


def _pose7_to_pose9(value: np.ndarray) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64).reshape(-1)
    if pose.shape != (7,) or not np.all(np.isfinite(pose)):
        raise ProfileRuntimeError("RPC TCP pose must contain 7 finite XYZ+XYZW values")
    rotation6d = matrix_to_rotation6d(quaternion_xyzw_to_matrix(pose[3:7]))
    return np.concatenate((pose[:3], rotation6d))


class ProfileSnapshotMapper:
    """Compile a v2 system description into one registered policy profile."""

    def __init__(
        self,
        description,
        *,
        profile_id: str,
        feature_contract_info: str | Path | None = None,
    ) -> None:
        self.profile = get_profile(profile_id)
        self.profile_id = self.profile.profile_id
        if int(description.schema_version) != 2:
            raise ProfileRuntimeError("PolicyDataService schema_version must be 2")
        if not description.schema_hash or not description.session_id:
            raise ProfileRuntimeError("RPC description lacks schema/session identity")
        self.schema_hash = str(description.schema_hash)
        self.session_id = str(description.session_id)
        self.required_channels = PROFILE_CHANNELS[self.profile_id]
        descriptors = _descriptor_map(description)
        self.image_shapes: dict[str, tuple[int, int, int]] = {}

        if any(channel.endswith(".q") for channel in self.required_channels):
            for side in ("left", "right"):
                _require_descriptor(
                    descriptors,
                    f"arm.{side}.q",
                    shape=(7,),
                    semantic="joint_position",
                )
        if any(channel.endswith(".tcp_pose") for channel in self.required_channels):
            for side in ("left", "right"):
                descriptor = _require_descriptor(
                    descriptors,
                    f"arm.{side}.tcp_pose",
                    shape=(7,),
                    semantic="cartesian_pose_xyzw",
                )
                if descriptor.frame_id != "world":
                    raise ProfileRuntimeError("RPC TCP pose must be expressed in world")
        for side in ("left", "right"):
            _require_descriptor(
                descriptors,
                f"hand.{side}.angle",
                shape=(6,),
                semantic="hand_actuator_angle",
            )
        for feature_key, channel_id in PROFILE_IMAGE_CHANNELS.items():
            descriptor = _require_descriptor(
                descriptors,
                channel_id,
                shape=None,
                semantic="rgb_image",
            )
            shape = tuple(int(item) for item in descriptor.tensor.shape)
            if len(shape) != 3 or shape[2] != 3:
                raise ProfileRuntimeError(f"RPC image shape is invalid: {channel_id}")
            self.image_shapes[feature_key] = shape

        actions = {item.schema_id: item for item in description.action_schemas}
        action = actions.get("cartesian_delta_rotvec_v1")
        if action is None or tuple(action.tensor.shape) != (
            self.profile.action_dimension,
        ):
            raise ProfileRuntimeError(
                "RPC does not provide the registered 24D Cartesian action schema"
            )
        if action.frame_id != "world" or not action.relative:
            raise ProfileRuntimeError("RPC Cartesian action semantics changed")

        self.contract = None
        if feature_contract_info:
            self.contract = FeatureContract.from_info(feature_contract_info)
            self.contract.validate_profile(self.profile)
            for key, shape in self.image_shapes.items():
                if self.contract.feature(key).shape != shape:
                    raise ProfileRuntimeError(
                        f"live image {key} shape {shape} differs from checkpoint "
                        f"{self.contract.feature(key).shape}"
                    )

    def frame_from_snapshot(self, snapshot) -> dict[str, np.ndarray]:
        if snapshot.schema_hash != self.schema_hash:
            raise ProfileRuntimeError("RPC system schema changed during inference")
        if snapshot.session_id != self.session_id:
            raise ProfileRuntimeError("RPC hardware session changed during inference")
        if not snapshot.complete:
            raise ProfileRuntimeError(
                "RPC snapshot is incomplete: " + "; ".join(snapshot.errors)
            )
        samples = {item.channel_id: item for item in snapshot.samples}
        if len(samples) != len(snapshot.samples):
            raise ProfileRuntimeError("RPC snapshot contains duplicate channels")
        missing = sorted(set(self.required_channels).difference(samples))
        if missing:
            raise ProfileRuntimeError("RPC snapshot is missing: " + ", ".join(missing))
        for channel_id in self.required_channels:
            sample = samples[channel_id]
            if not sample.timing.valid or not sample.timing.timing_valid:
                raise ProfileRuntimeError(
                    f"RPC channel {channel_id} is invalid: "
                    f"{sample.timing.invalid_reason}"
                )

        hands = np.concatenate(
            [
                decode_tensor(samples[f"hand.{side}.angle"].tensor)
                for side in ("left", "right")
            ]
        )
        if self.profile_id == "joint_proprio_cartesian_v1":
            joints = np.concatenate(
                [
                    decode_tensor(samples[f"arm.{side}.q"].tensor)
                    for side in ("left", "right")
                ]
            )
            state = joint_minimal_state(joints, hands)
        else:
            poses = np.concatenate(
                [
                    _pose7_to_pose9(
                        decode_tensor(samples[f"arm.{side}.tcp_pose"].tensor)
                    )
                    for side in ("left", "right")
                ]
            )
            if self.profile_id == "cartesian_proprio_v1":
                state = cartesian_minimal_state(poses, hands)
            else:
                joints = np.concatenate(
                    [
                        decode_tensor(samples[f"arm.{side}.q"].tensor)
                        for side in ("left", "right")
                    ]
                )
                state = legacy_state38(joints, poses, hands)

        frame = {
            "observation.state": state.astype(np.float32, copy=False),
        }
        for feature_key, channel_id in PROFILE_IMAGE_CHANNELS.items():
            frame[feature_key] = _decode_image(
                samples[channel_id].image, self.image_shapes[feature_key]
            )
        return frame


def _target_host(target: str) -> str:
    return str(target).rsplit(":", 1)[0].strip("[]")


def _is_loopback_target(target: str) -> bool:
    host = _target_host(target)
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class SyncPolicyProfileClient:
    """Blocking client suitable for LeRobot's synchronous Robot API."""

    def __init__(
        self,
        *,
        target: str,
        profile_id: str,
        insecure_loopback: bool = False,
        server_ca: str | Path | None = None,
        client_cert: str | Path | None = None,
        client_key: str | Path | None = None,
        feature_contract_info: str | Path | None = None,
        connect_timeout_s: float = 5.0,
        request_timeout_s: float = 0.5,
        state_max_age_ms: float = 50.0,
        hand_max_age_ms: float = 150.0,
        image_max_age_ms: float = 150.0,
    ) -> None:
        self.target = str(target)
        self.profile_id = str(profile_id)
        self.insecure_loopback = bool(insecure_loopback)
        self.server_ca = None if not server_ca else Path(server_ca).expanduser()
        self.client_cert = None if not client_cert else Path(client_cert).expanduser()
        self.client_key = None if not client_key else Path(client_key).expanduser()
        self.feature_contract_info = feature_contract_info
        self.connect_timeout_s = float(connect_timeout_s)
        self.request_timeout_s = float(request_timeout_s)
        self.state_max_age_ns = int(float(state_max_age_ms) * 1e6)
        self.hand_max_age_ns = int(float(hand_max_age_ms) * 1e6)
        self.image_max_age_ns = int(float(image_max_age_ms) * 1e6)
        self.channel = None
        self.stub = None
        self.mapper: ProfileSnapshotMapper | None = None

    def connect(self) -> ProfileSnapshotMapper:
        import grpc

        if self.channel is not None:
            raise ProfileRuntimeError("RPC profile client is already connected")
        if self.insecure_loopback:
            if not _is_loopback_target(self.target):
                raise ProfileRuntimeError(
                    "insecure PolicyDataService transport is loopback-only"
                )
            channel = grpc.insecure_channel(self.target)
        else:
            if self.server_ca is None:
                raise ProfileRuntimeError("server_ca is required for TLS")
            cert = None if self.client_cert is None else self.client_cert.read_bytes()
            key = None if self.client_key is None else self.client_key.read_bytes()
            if (cert is None) != (key is None):
                raise ProfileRuntimeError(
                    "client_cert and client_key must be provided together"
                )
            credentials = grpc.ssl_channel_credentials(
                root_certificates=self.server_ca.read_bytes(),
                private_key=key,
                certificate_chain=cert,
            )
            channel = grpc.secure_channel(self.target, credentials)
        try:
            grpc.channel_ready_future(channel).result(timeout=self.connect_timeout_s)
            stub = pb_grpc.PolicyDataServiceStub(channel)
            description = stub.DescribeSystem(
                pb.DescribeSystemRequest(), timeout=self.request_timeout_s
            )
            mapper = ProfileSnapshotMapper(
                description,
                profile_id=self.profile_id,
                feature_contract_info=self.feature_contract_info,
            )
        except Exception:
            channel.close()
            raise
        self.channel = channel
        self.stub = stub
        self.mapper = mapper
        return mapper

    def get_frame(self) -> dict[str, np.ndarray]:
        if self.stub is None or self.mapper is None:
            raise ProfileRuntimeError("RPC profile client is not connected")
        channels = []
        for channel_id in self.mapper.required_channels:
            if channel_id.startswith("camera."):
                max_age_ns = self.image_max_age_ns
            elif channel_id.startswith("hand."):
                max_age_ns = self.hand_max_age_ns
            else:
                max_age_ns = self.state_max_age_ns
            channels.append(
                pb.SnapshotChannelRequest(
                    channel_id=channel_id,
                    alignment=pb.LATEST_CAUSAL,
                    tolerance_ns=max_age_ns,
                    max_age_ns=max_age_ns,
                )
            )
        snapshot = self.stub.GetSnapshot(
            pb.SnapshotRequest(
                expected_schema_hash=self.mapper.schema_hash,
                expected_session_id=self.mapper.session_id,
                target_monotonic_ns=0,
                channels=channels,
            ),
            timeout=self.request_timeout_s,
        )
        return self.mapper.frame_from_snapshot(snapshot)

    def close(self) -> None:
        channel, self.channel = self.channel, None
        self.stub = None
        self.mapper = None
        if channel is not None:
            channel.close()
