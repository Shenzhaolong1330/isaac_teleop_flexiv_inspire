from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import yaml


_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "apps" / "policy_client" / "src"))

from flexiv_policy_client import MultiRatePolicyClient  # noqa: E402
from flexiv_policy_client.client import decode_sample  # noqa: E402
from flexiv_policy_client.generated import policy_data_v2_pb2 as data_pb  # noqa: E402


def test_copyable_client_example_subscribes_force_torque_at_200_hz() -> None:
    document = yaml.safe_load(
        (_ROOT / "apps" / "policy_client" / "config.example.yaml").read_text(
            encoding="utf-8"
        )
    )
    channels = document["subscriptions"]["force_torque"]
    expected = {
        f"arm.{side}.{field}"
        for side in ("left", "right")
        for field in (
            "tau", "tau_des", "tau_ext", "tau_interact", "raw_ft", "tcp_wrench"
        )
    } | {f"hand.{side}.actual_force" for side in ("left", "right")}

    assert set(channels) == expected
    assert all(float(channels[channel]) == 200.0 for channel in expected)


def test_copyable_client_decodes_read_only_tensor_sample() -> None:
    values = np.arange(7, dtype="<f8")
    message = data_pb.SampleEnvelope(
        schema_version=2,
        schema_hash="schema",
        channel_id="arm.left.q",
        timing=data_pb.SampleMetadata(
            source_time_ns=10,
            host_receive_time_ns=20,
            mapped_host_time_ns=15,
            sequence=3,
            valid=True,
            age_ns=5,
            timing_valid=True,
        ),
        tensor=data_pb.TensorPayload(
            dtype=data_pb.FLOAT64,
            shape=(7,),
            data=values.tobytes(),
        ),
    )

    sample = decode_sample(message)

    assert sample.channel_id == "arm.left.q"
    assert np.array_equal(sample.value, values)
    assert sample.value.flags.writeable is False
    assert sample.sequence == 3


def test_copyable_client_subscriptions_keep_latest_sample() -> None:
    messages = []
    for sequence in (1, 2):
        value = np.asarray([sequence] * 6, dtype="<f8")
        messages.append(
            data_pb.SampleEnvelope(
                schema_version=2,
                schema_hash="schema",
                channel_id="hand.left.angle",
                timing=data_pb.SampleMetadata(
                    sequence=sequence,
                    valid=True,
                    timing_valid=True,
                ),
                tensor=data_pb.TensorPayload(
                    dtype=data_pb.FLOAT64,
                    shape=(6,),
                    data=value.tobytes(),
                ),
            )
        )

    class Call:
        def __iter__(self):
            return iter(messages)

        def cancel(self):
            pass

    class DataStub:
        def SubscribeSamples(self, request):
            assert request.channels[0].max_rate_hz == 200.0
            return Call()

    client = MultiRatePolicyClient(target="127.0.0.1:50051")
    client.description = data_pb.SystemDescription(
        schema_version=2,
        schema_hash="schema",
        session_id="session",
        channels=(
            data_pb.ChannelDescriptor(
                channel_id="hand.left.angle", native_rate_hz=200.0
            ),
        ),
    )
    client.data_stub = DataStub()

    client.subscribe("fast", {"hand.left.angle": 200.0})
    result = client.wait_for_channels(
        ("hand.left.angle",), timeout_s=1.0
    )

    assert result["hand.left.angle"].sequence == 2
    assert np.array_equal(result["hand.left.angle"].value, [2] * 6)
    client.close()


def test_copyable_client_sends_normalized_24d_action() -> None:
    captured = []

    class ControlStub:
        def AcquireControlLease(self, request, timeout):
            assert request.client_id == "test-policy"
            return SimpleNamespace(granted=True, lease_id="lease", reason="")

        def Stop(self, request, timeout):
            return SimpleNamespace()

    class DataStub:
        def StreamActions(self, requests):
            for request in requests:
                captured.append(request)
                yield data_pb.PolicyActionResult(
                    sequence=request.sequence,
                    accepted=True,
                    control_state="ACTIVE",
                )

    client = MultiRatePolicyClient(
        target="127.0.0.1:50051",
        action_client_id="test-policy",
    )
    client.description = data_pb.SystemDescription(
        schema_version=2,
        schema_hash="schema",
        session_id="session",
    )
    client.data_stub = DataStub()
    client.control_stub = ControlStub()
    action = np.zeros(24, dtype=np.float32)
    action[12:24] = 0.5

    response = client.send_action(action)

    assert response.accepted is True
    assert len(captured) == 1
    assert captured[0].action_schema_id == "cartesian_delta_rotvec_v1"
    assert captured[0].points[0].action.shape == [24]
    client.close()
