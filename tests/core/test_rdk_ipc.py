from __future__ import annotations

import os
from pathlib import Path
import socket
import stat
import threading
import time

import pytest

from flexiv_rdk_daemon.ipc import (
    SeqpacketServer,
    StructEnvelopeCodec,
)
from flexiv_rdk_daemon.typed_ipc import TypedEnvelopeCodec
from flexiv_rdk_daemon.mock_backend import MockBackend
from flexiv_inspire_control.ipc_client import RDKIPCClient


def test_struct_codec_is_explicitly_development_only_but_well_framed() -> None:
    codec = StructEnvelopeCodec()
    data = codec.encode("hello", 7, {"client_name": "test"})
    kind, sequence, payload = codec.decode(data)
    assert (kind, sequence) == ("hello", 7)
    assert payload["client_name"] == "test"


def test_typed_codec_preserves_uint64_beyond_double_precision() -> None:
    try:
        codec = TypedEnvelopeCodec.load()
    except RuntimeError:
        pytest.skip("generated pb2 not built yet")
    large = (1 << 63) + 12345
    packet = codec.encode(
        "dual_arm_state",
        large,
        {
            "left": {
                "side": "left",
                "connected": True,
                "robot_time_ns": str(large),
                "host_receive_monotonic_ns": str(large - 1),
                "connection_generation": "1",
            },
            "right": {
                "side": "right",
                "connected": True,
                "robot_time_ns": str(large),
                "host_receive_monotonic_ns": str(large - 1),
                "connection_generation": "1",
            },
        },
    )
    kind, sequence, payload = codec.decode(packet)
    assert kind == "dual_arm_state"
    assert sequence == large
    assert payload["left"]["robot_time_ns"] == str(large)
    assert payload["left"]["host_receive_monotonic_ns"] == str(large - 1)


def test_seqpacket_socket_is_0600_and_accepts_concurrent_peers(tmp_path: Path) -> None:
    path = tmp_path / "rdk.sock"
    codec = StructEnvelopeCodec()

    def handler(kind, sequence, payload, credentials):
        return "command_ack", {
            "accepted": True,
            "reason": payload.get("name", ""),
        }

    server = SeqpacketServer(path, handler, codec=codec)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    for _ in range(100):
        if path.exists():
            break
        time.sleep(0.001)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600

    first = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    first.connect(str(path))
    # Keep first persistent and idle; second must still receive a response.
    with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as second:
        second.settimeout(0.5)
        second.connect(str(path))
        second.sendall(codec.encode("hello", 1, {"name": "second"}))
        kind, _, payload = codec.decode(second.recv(65536))
        assert kind == "command_ack"
        assert payload["reason"] == "second"
    first.close()
    server.close()
    thread.join(timeout=1.0)


def test_server_refuses_to_replace_regular_file(tmp_path: Path) -> None:
    path = tmp_path / "rdk.sock"
    path.write_text("do not delete", encoding="utf-8")
    server = SeqpacketServer(path, lambda *args: ("error", {}), codec=StructEnvelopeCodec())
    with pytest.raises(RuntimeError):
        server.open()
    assert path.read_text(encoding="utf-8") == "do not delete"

def test_typed_codec_preserves_empty_observe_oneof_presence() -> None:
    codec = TypedEnvelopeCodec.load()
    packet = codec.encode("observe", 1, {})
    kind, sequence, payload = codec.decode(packet)
    assert kind == "observe"
    assert sequence == 1
    assert payload == {}


def test_four_persistent_typed_clients_can_send_empty_observe(
    tmp_path: Path,
) -> None:
    path = tmp_path / "typed-rdk.sock"
    codec = TypedEnvelopeCodec.load()

    backend = MockBackend()

    def observe_handler(kind, sequence, payload, credentials):
        assert kind == "observe"
        assert payload == {}
        observation = backend.observe_both().to_wire()
        observation["daemon_instance_id"] = f"mock-{credentials[0]}"
        return "dual_arm_state", observation

    server = SeqpacketServer(path, observe_handler, codec=codec)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    for _ in range(100):
        if path.exists():
            break
        time.sleep(0.001)
    clients = [RDKIPCClient(path, timeout_s=0.5) for _ in range(4)]
    try:
        for _round in range(2):
            for client in clients:
                kind, payload = client.request("observe", {})
                assert kind == "dual_arm_state"
                assert payload["left"]["side"] == "left"
                assert payload["right"]["side"] == "right"
                assert len(payload["left"]["tcp_pose_rdk_xyz_wxyz"]) == 7
                assert len(payload["right"]["tcp_pose_rdk_xyz_wxyz"]) == 7
    finally:
        for client in clients:
            client.close()
        server.close()
        thread.join(timeout=1.0)

ARM_STATE_WIRE_KEYS = {
    "side",
    "connected",
    "robot_time_ns",
    "host_receive_monotonic_ns",
    "q",
    "dq",
    "tau",
    "tau_des",
    "tau_ext",
    "tau_interact",
    "tcp_pose_rdk_xyz_wxyz",
    "tcp_velocity",
    "raw_ft",
    "external_wrench",
    "temperature",
    "fault",
    "connection_generation",
    "robot_time_sec",
    "robot_time_nsec",
    "clock_domain",
    "host_receive_unix_ns",
}


def test_mock_observation_roundtrip_covers_every_arm_state_field() -> None:
    codec = TypedEnvelopeCodec.load()
    payload = MockBackend().observe_both().to_wire()
    payload["daemon_instance_id"] = "mock-instance"
    kind, sequence, decoded = codec.decode(
        codec.encode("dual_arm_state", 17, payload)
    )
    assert (kind, sequence) == ("dual_arm_state", 17)
    assert decoded["daemon_instance_id"] == "mock-instance"
    for side in ("left", "right"):
        arm = decoded[side]
        assert set(arm) == ARM_STATE_WIRE_KEYS
        assert len(arm["q"]) == len(arm["dq"]) == 7
        assert len(arm["tau"]) == len(arm["tau_des"]) == 7
        assert len(arm["tau_ext"]) == len(arm["tau_interact"]) == 7
        assert arm["tcp_pose_rdk_xyz_wxyz"] == [
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
        ]
        assert len(arm["tcp_velocity"]) == 6
        assert len(arm["raw_ft"]) == len(arm["external_wrench"]) == 6
        assert len(arm["temperature"]) == 7


def test_response_encoding_failure_returns_typed_error_and_keeps_peer_alive(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bad-response.sock"
    codec = TypedEnvelopeCodec.load()

    def bad_handler(kind, sequence, payload, credentials):
        return "dual_arm_state", {"left": {"unknown_field": True}}

    server = SeqpacketServer(path, bad_handler, codec=codec)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    for _ in range(100):
        if path.exists():
            break
        time.sleep(0.001)
    client = RDKIPCClient(path, timeout_s=0.5)
    try:
        for _ in range(2):
            with pytest.raises(RuntimeError, match="response_encode_error"):
                client.request("observe", {})
    finally:
        client.close()
        server.close()
        thread.join(timeout=1.0)


def test_client_reports_closed_connection_before_protobuf_decode(
    tmp_path: Path,
) -> None:
    path = tmp_path / "close-without-response.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    listener.bind(str(path))
    listener.listen(1)

    def close_without_response() -> None:
        connection, _ = listener.accept()
        with connection:
            connection.recv(65536)

    thread = threading.Thread(target=close_without_response, daemon=True)
    thread.start()
    client = RDKIPCClient(path, timeout_s=0.5)
    try:
        with pytest.raises(
            ConnectionError,
            match="closed IPC connection without a response",
        ):
            client.request("observe", {})
    finally:
        client.close()
        listener.close()
        thread.join(timeout=1.0)

@pytest.mark.parametrize(
    "kind",
    (
        "hello",
        "observe",
        "dual_arm_state",
        "hand_observation",
        "cartesian_command",
        "command_ack",
        "authorize_zero_ft",
        "authorize_zero_ft_result",
        "authorize_control",
        "authorize_control_result",
        "zero_ft",
        "zero_ft_result",
        "hold",
        "error",
    ),
)
def test_typed_codec_sets_oneof_for_every_empty_payload(kind: str) -> None:
    codec = TypedEnvelopeCodec.load()
    packet = codec.encode(kind, 9, {})
    decoded_kind, sequence, _payload = codec.decode(packet)
    assert decoded_kind == kind
    assert sequence == 9
