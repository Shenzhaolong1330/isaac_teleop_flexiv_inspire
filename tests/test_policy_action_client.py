from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from policy_runtime_client import (
    PolicyActionError,
    SyncPolicyActionClient,
    SyncPolicyProfileClient,
)
from policy_runtime_client import control as control_module

from flexiv_inspire_isaac.policy_api.broker import (
    LatestActionBuffer,
    PolicyStreamLiveness,
)
from flexiv_inspire_isaac.policy_api.channel_broker import ChannelBroker
from flexiv_inspire_isaac.policy_api.data_server import PolicyDataServicer
from flexiv_inspire_isaac.policy_api.generated import policy_data_v2_pb2_grpc
from flexiv_inspire_isaac.policy_api.generated import policy_service_v1_pb2 as v1_pb
from flexiv_inspire_isaac.policy_api.generated import policy_service_v1_pb2_grpc as v1_grpc
from flexiv_inspire_isaac.policy_api.lease import (
    ControlLeaseManager,
    LocalControlState,
)
from flexiv_inspire_isaac.policy_api.system_schema import build_system_schema


class _FakeCall:
    def __init__(self, requests, received, *, accept=True):
        self.requests = requests
        self.received = received
        self.accept = accept
        self.cancelled = False

    def __iter__(self):
        for request in self.requests:
            self.received.append(request)
            yield SimpleNamespace(
                sequence=request.sequence,
                accepted=self.accept,
                reason="" if self.accept else "local policy authorization lost",
                control_state="POLICY_ARMED",
            )

    def cancel(self):
        self.cancelled = True


class _ControlStub:
    def __init__(self, channel):
        self.channel = channel
        self.acquires = []
        self.stops = []
        self.releases = []

    def AcquireControlLease(self, request, timeout):
        self.acquires.append((request, timeout))
        return SimpleNamespace(granted=True, lease_id="lease", reason="")

    def Stop(self, request, timeout):
        self.stops.append((request, timeout))
        return SimpleNamespace()

    def ReleaseControlLease(self, request, timeout):
        self.releases.append((request, timeout))
        return SimpleNamespace(success=True)


class _DataStub:
    accept = True

    def __init__(self, channel):
        self.channel = channel
        self.received = []
        self.calls = []

    def StreamActions(self, requests):
        call = _FakeCall(requests, self.received, accept=self.accept)
        self.calls.append(call)
        return call


def _profile_client(profile_id="joint_proprio_cartesian_v1"):
    return SimpleNamespace(
        channel=object(),
        stub=object(),
        mapper=SimpleNamespace(
            schema_hash="schema", session_id="session", profile_id=profile_id
        ),
        request_timeout_s=0.5,
    )


def _install_fakes(monkeypatch, *, accept=True):
    _DataStub.accept = accept
    controls = []
    data = []

    def control_factory(channel):
        result = _ControlStub(channel)
        controls.append(result)
        return result

    def data_factory(channel):
        result = _DataStub(channel)
        data.append(result)
        return result

    monkeypatch.setattr(control_module.control_pb_grpc, "PolicyServiceStub", control_factory)
    monkeypatch.setattr(control_module.data_pb_grpc, "PolicyDataServiceStub", data_factory)
    return controls, data


def test_action_client_lazily_acquires_and_keeps_one_stream(monkeypatch):
    controls, data = _install_fakes(monkeypatch)
    client = SyncPolicyActionClient(
        _profile_client(), client_id="test", action_ttl_ms=250
    )
    action = np.concatenate((np.zeros(12), np.full(12, 0.5)))

    first = client.send(action)
    second = client.send(action)
    client.close()

    assert first.accepted and second.accepted
    assert len(controls[0].acquires) == 1
    assert len(data[0].calls) == 1
    assert [item.sequence for item in data[0].received] == [1, 2]
    wire = data[0].received[0]
    assert wire.schema_hash == "schema"
    assert wire.action_schema_id == "cartesian_delta_rotvec_v1"
    assert tuple(wire.points[0].action.shape) == (24,)
    assert np.allclose(
        np.frombuffer(wire.points[0].action.data, dtype="<f4"), action
    )
    assert len(controls[0].stops) == 1
    assert not controls[0].releases


def test_right_action_client_sends_12d_registered_schema(monkeypatch):
    controls, data = _install_fakes(monkeypatch)
    client = SyncPolicyActionClient(
        _profile_client("right_joint_proprio_cartesian_v1"), client_id="right"
    )
    action = np.concatenate((np.zeros(6), np.full(6, 0.5)))

    result = client.send(action)
    client.close()

    wire = data[0].received[0]
    assert result.accepted
    assert wire.action_schema_id == "right_cartesian_delta_rotvec_v1"
    assert tuple(wire.points[0].action.shape) == (12,)
    assert np.allclose(
        np.frombuffer(wire.points[0].action.data, dtype="<f4"), action
    )


def test_action_rejection_drops_lease_and_stream(monkeypatch):
    controls, data = _install_fakes(monkeypatch, accept=False)
    client = SyncPolicyActionClient(_profile_client(), client_id="test")
    action = np.concatenate((np.zeros(12), np.full(12, 0.5)))

    with pytest.raises(PolicyActionError, match="authorization"):
        client.send(action)

    assert len(controls[0].acquires) == 1
    assert data[0].calls[0].cancelled
    assert client._lease_id is None


@pytest.mark.parametrize(
    "action",
    (
        np.zeros(23),
        np.concatenate((np.zeros(12), np.full(12, 1.1))),
        np.concatenate((np.full(1, np.nan), np.zeros(23))),
    ),
)
def test_invalid_action_never_acquires_a_lease(monkeypatch, action):
    controls, _ = _install_fakes(monkeypatch)
    client = SyncPolicyActionClient(_profile_client(), client_id="test")

    with pytest.raises((PolicyActionError, ValueError)):
        client.send(action)

    assert not controls[0].acquires


def test_real_loopback_v1_lease_and_v2_action_share_safety_ingress():
    import asyncio
    import grpc

    state = LocalControlState(
        session_id="integration-session",
        ft_zeroed=True,
        local_policy_authorized=True,
        pedal_valid=True,
        arms_online=True,
        hands_online=True,
        state="POLICY_ARMED",
    )

    async def scenario():
        schema = build_system_schema(image_shape=(2, 3, 3))
        lease = ControlLeaseManager()
        actions = LatestActionBuffer()
        liveness = PolicyStreamLiveness()
        stops = []
        sequences = {}

        class LeaseServicer(v1_grpc.PolicyServiceServicer):
            async def AcquireControlLease(self, request, context):
                try:
                    granted = lease.acquire(
                        client_id=request.client_id,
                        peer=f"{context.peer()}|",
                        requested_ms=request.requested_duration_ms,
                        local_state=state,
                    )
                except PermissionError as exc:
                    return v1_pb.ControlLease(granted=False, reason=str(exc))
                return v1_pb.ControlLease(
                    granted=True,
                    lease_id=granted.token,
                    expires_monotonic_ns=granted.expires_ns,
                )

            async def ReleaseControlLease(self, request, context):
                return v1_pb.LeaseResult(
                    success=lease.release(request.lease_id, f"{context.peer()}|")
                )

            async def Stop(self, request, context):
                lease.validate(request.lease_id, f"{context.peer()}|", state.session_id)
                stops.append(request.reason)
                lease.invalidate()
                liveness.clear()
                sequences.pop(request.lease_id, None)
                return v1_pb.StopResult(
                    stop_published=True,
                    control_state=state.state,
                    status="stopped",
                )

        server = grpc.aio.server()
        v1_grpc.add_PolicyServiceServicer_to_server(LeaseServicer(), server)
        policy_data_v2_pb2_grpc.add_PolicyDataServiceServicer_to_server(
            PolicyDataServicer(
                schema=schema,
                broker=ChannelBroker(schema),
                local_state=lambda: state,
                lease_manager=lease,
                action_buffer=actions,
                stop_callback=stops.append,
                action_liveness=liveness,
                last_sequence_by_lease=sequences,
            ),
            server,
        )
        try:
            port = server.add_insecure_port("127.0.0.1:0")
        except RuntimeError as exc:
            pytest.skip(f"loopback sockets are unavailable in this sandbox: {exc}")
        await server.start()

        def client_round_trip():
            profile = SyncPolicyProfileClient(
                target=f"127.0.0.1:{port}",
                profile_id="joint_proprio_cartesian_v1",
                insecure_loopback=True,
            )
            profile.connect()
            control = SyncPolicyActionClient(profile, client_id="integration")
            result = control.send(
                np.concatenate((np.zeros(12), np.full(12, 0.5)))
            )
            control.close(reason="integration-client-close")
            profile.close()
            return result

        try:
            result = await asyncio.to_thread(client_round_trip)
            assert result.accepted
            chunk = actions.take()
            assert chunk is not None and chunk.sequence == 1
            assert np.allclose(chunk.points[0].values[18:30], 500.0)
            assert stops == ["integration-client-close"]
        finally:
            await server.stop(grace=0)

    asyncio.run(scenario())
