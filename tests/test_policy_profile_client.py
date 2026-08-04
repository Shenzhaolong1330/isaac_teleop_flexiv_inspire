from __future__ import annotations

from types import SimpleNamespace

import pytest

from policy_runtime_client import ProfileRuntimeError, SyncPolicyProfileClient


class _Stub:
    def __init__(self, snapshots) -> None:
        self.snapshots = list(snapshots)
        self.calls = 0

    def GetSnapshot(self, request, *, timeout):
        del request, timeout
        selected = self.snapshots[min(self.calls, len(self.snapshots) - 1)]
        self.calls += 1
        return selected


class _Mapper:
    required_channels = ("arm.left.q",)
    schema_hash = "schema"
    session_id = "session"

    @staticmethod
    def frame_from_snapshot(snapshot):
        if not snapshot.complete:
            raise ProfileRuntimeError(
                "RPC snapshot is incomplete: arm.left.q:source-clock-unmapped"
            )
        return {"observation.state": snapshot.value}


def _client(*, timeout_s: float) -> SyncPolicyProfileClient:
    client = SyncPolicyProfileClient(
        target="127.0.0.1:50051",
        profile_id="dual_arm_lerobot_v1",
        insecure_loopback=True,
        snapshot_ready_timeout_s=timeout_s,
        snapshot_retry_interval_s=0.001,
    )
    client.mapper = _Mapper()
    return client


def test_profile_client_waits_for_initial_clock_mapping(caplog) -> None:
    caplog.set_level("INFO")
    client = _client(timeout_s=0.1)
    client.stub = _Stub(
        [
            SimpleNamespace(complete=False),
            SimpleNamespace(complete=False),
            SimpleNamespace(complete=True, value="ready"),
        ]
    )

    assert client.get_frame() == {"observation.state": "ready"}
    assert client.stub.calls == 3
    assert "waiting for complete clock-mapped" in caplog.text


def test_profile_client_still_fails_closed_without_readiness_window() -> None:
    client = _client(timeout_s=0.0)
    client.stub = _Stub([SimpleNamespace(complete=False)])

    with pytest.raises(ProfileRuntimeError, match="source-clock-unmapped"):
        client.get_frame()

    assert client.stub.calls == 1


def test_profile_client_does_not_retry_schema_error() -> None:
    client = _client(timeout_s=1.0)
    client.stub = _Stub([SimpleNamespace(complete=True, value="ignored")])
    client.mapper.frame_from_snapshot = lambda snapshot: (_ for _ in ()).throw(
        ProfileRuntimeError("RPC system schema changed during inference")
    )

    with pytest.raises(ProfileRuntimeError, match="schema changed"):
        client.get_frame()

    assert client.stub.calls == 1
