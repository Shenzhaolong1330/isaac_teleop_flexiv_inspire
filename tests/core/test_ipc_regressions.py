from __future__ import annotations

from pathlib import Path
import socket
import threading
import time

import pytest

from flexiv_rdk_daemon.ipc import SeqpacketServer, StructEnvelopeCodec


def wait_for(path: Path) -> None:
    deadline = time.monotonic() + 1.0
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.001)
    assert path.exists()


def handler(kind, sequence, payload, credentials):
    return "command_ack", {"accepted": True, "reason": payload.get("name", "")}


def request(
    path: Path,
    codec: StructEnvelopeCodec,
    sock: socket.socket,
    sequence: int,
    name: str,
):
    sock.sendall(codec.encode("hello", sequence, {"name": name}))
    return codec.decode(sock.recv(65536))


def test_four_persistent_connections_from_same_pid_have_independent_sequences(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rdk.sock"
    codec = StructEnvelopeCodec()
    server = SeqpacketServer(path, handler, codec=codec)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    wait_for(path)
    clients = [socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) for _ in range(4)]
    try:
        for client in clients:
            client.settimeout(0.5)
            client.connect(str(path))
        for index, client in enumerate(clients):
            kind, _, payload = request(path, codec, client, 1, f"client-{index}")
            assert kind == "command_ack"
            assert payload["accepted"]
        kind, _, payload = request(path, codec, clients[0], 1, "duplicate")
        assert kind == "error"
        assert payload["code"] == "IPCProtocolError"
    finally:
        for client in clients:
            client.close()
        server.close()
        thread.join(timeout=1.0)


def test_disconnect_notification_waits_for_last_connection_from_process(
    tmp_path: Path,
) -> None:
    class TrackingHandler:
        def __init__(self) -> None:
            self.disconnected = []

        def __call__(self, kind, sequence, payload, credentials):
            return "command_ack", {"accepted": True}

        def peer_disconnected(self, credentials) -> None:
            self.disconnected.append(credentials)

    path = tmp_path / "rdk.sock"
    codec = StructEnvelopeCodec()
    handler = TrackingHandler()
    server = SeqpacketServer(path, handler, codec=codec)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    wait_for(path)
    clients = [socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) for _ in range(2)]
    try:
        for index, client in enumerate(clients):
            client.settimeout(0.5)
            client.connect(str(path))
            request(path, codec, client, 1, f"client-{index}")
        clients[0].close()
        time.sleep(0.05)
        assert handler.disconnected == []
        clients[1].close()
        deadline = time.monotonic() + 0.5
        while not handler.disconnected and time.monotonic() < deadline:
            time.sleep(0.005)
        assert len(handler.disconnected) == 1
    finally:
        for client in clients:
            client.close()
        server.close()
        thread.join(timeout=1.0)


def test_second_server_refuses_to_unlink_active_socket(tmp_path: Path) -> None:
    path = tmp_path / "rdk.sock"
    codec = StructEnvelopeCodec()
    first = SeqpacketServer(path, handler, codec=codec)
    thread = threading.Thread(target=first.serve_forever, daemon=True)
    thread.start()
    wait_for(path)
    second = SeqpacketServer(path, handler, codec=codec)
    with pytest.raises(RuntimeError, match="active RDK daemon"):
        second.open()
    assert path.exists()
    first.close()
    thread.join(timeout=1.0)


def test_stale_owned_socket_is_recovered(tmp_path: Path) -> None:
    path = tmp_path / "rdk.sock"
    stale = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    stale.bind(str(path))
    stale.close()
    server = SeqpacketServer(path, handler, codec=StructEnvelopeCodec())
    server.open()
    try:
        assert path.exists()
    finally:
        server.close()
