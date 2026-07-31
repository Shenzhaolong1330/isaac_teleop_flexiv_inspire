"""Protobuf framing over a permission-restricted Unix SOCK_SEQPACKET socket."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import errno
import os
from pathlib import Path
import socket
import stat
import struct
import threading
from typing import Any, Protocol

from google.protobuf.json_format import MessageToDict
from google.protobuf.struct_pb2 import Struct

SCHEMA_VERSION = 1
MAX_PACKET_BYTES = 65536
ALLOWED_KINDS = frozenset(
    {
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
        "authorize_home",
        "authorize_home_result",
        "home_command",
        "home_result",
        "zero_ft",
        "zero_ft_result",
        "hold",
        "error",
    }
)


class IPCProtocolError(ValueError):
    pass


class EnvelopeCodec(Protocol):
    def encode(self, kind: str, sequence: int, payload: Mapping[str, Any]) -> bytes: ...
    def decode(self, data: bytes) -> tuple[str, int, dict[str, Any]]: ...


class StructEnvelopeCodec:
    """Development/mock-only protobuf fallback.

    This generic Struct codec is intentionally forbidden by the hardware CLI
    because nested uint64 timestamps become IEEE doubles. Production always uses
    :class:`TypedEnvelopeCodec`.
    """

    @staticmethod
    def encode(kind: str, sequence: int, payload: Mapping[str, Any]) -> bytes:
        if kind not in ALLOWED_KINDS:
            raise IPCProtocolError(f"unsupported packet kind {kind!r}")
        if sequence < 0:
            raise IPCProtocolError("sequence cannot be negative")
        message = Struct()
        message.update(
            {
                "schema_version": str(SCHEMA_VERSION),
                "kind": kind,
                "sequence": str(sequence),
                "payload": dict(payload),
            }
        )
        data = message.SerializeToString(deterministic=True)
        if len(data) > MAX_PACKET_BYTES:
            raise IPCProtocolError("encoded packet exceeds 64 KiB")
        return data

    @staticmethod
    def decode(data: bytes) -> tuple[str, int, dict[str, Any]]:
        if len(data) > MAX_PACKET_BYTES:
            raise IPCProtocolError("packet exceeds 64 KiB")
        message = Struct()
        try:
            message.ParseFromString(data)
        except Exception as exc:
            raise IPCProtocolError("invalid protobuf packet") from exc
        decoded = MessageToDict(message, preserving_proto_field_name=True)
        if decoded.get("schema_version") != str(SCHEMA_VERSION):
            raise IPCProtocolError("unsupported IPC schema version")
        kind = decoded.get("kind")
        if not isinstance(kind, str) or kind not in ALLOWED_KINDS:
            raise IPCProtocolError("invalid packet kind")
        sequence_text = decoded.get("sequence")
        if not isinstance(sequence_text, str) or not sequence_text.isdecimal():
            raise IPCProtocolError("sequence must be a decimal string")
        payload = decoded.get("payload")
        if not isinstance(payload, dict):
            raise IPCProtocolError("payload must be an object")
        return kind, int(sequence_text), payload


def _safe_prepare_socket_path(path: Path, codec: EnvelopeCodec) -> None:
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    parent.chmod(0o700)
    try:
        current = path.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISSOCK(current.st_mode):
        raise RuntimeError(f"refusing to replace non-socket IPC path: {path}")
    if current.st_uid != os.getuid():
        raise RuntimeError(f"refusing to replace socket not owned by uid {os.getuid()}: {path}")
    # Never unlink an endpoint owned by a live daemon.
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    probe.settimeout(0.25)
    try:
        probe.connect(str(path))
        try:
            probe.sendall(codec.encode("hello", 0, {"client_name": "startup_probe"}))
            probe.recv(MAX_PACKET_BYTES)
        except (OSError, IPCProtocolError):
            pass
    except OSError as exc:
        if exc.errno not in {errno.ECONNREFUSED, errno.ENOENT}:
            raise RuntimeError(f"cannot safely probe existing IPC socket: {path}") from exc
    else:
        raise RuntimeError(f"an active RDK daemon already owns IPC socket: {path}")
    finally:
        probe.close()
    try:
        after = path.lstat()
    except FileNotFoundError:
        return
    if (
        after.st_dev != current.st_dev
        or after.st_ino != current.st_ino
        or not stat.S_ISSOCK(after.st_mode)
        or after.st_uid != os.getuid()
    ):
        raise RuntimeError(f"IPC socket changed during stale-socket probe: {path}")
    path.unlink()


def peer_credentials(connection: socket.socket) -> tuple[int, int, int]:
    raw = connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
    return struct.unpack("3i", raw)


def peer_has_local_tty(pid: int) -> bool:
    try:
        target = os.readlink(f"/proc/{pid}/fd/0")
    except OSError:
        return False
    return target.startswith("/dev/pts/") or target.startswith("/dev/tty")


Handler = Callable[[str, int, dict[str, Any], tuple[int, int, int]], tuple[str, dict[str, Any]]]


class SeqpacketServer:
    """Concurrent same-UID seqpacket server.

    Persistent bridge connections cannot starve a one-shot local maintenance
    authorization client because each accepted peer has its own worker thread.
    """

    def __init__(
        self,
        path: Path,
        handler: Handler,
        *,
        codec: EnvelopeCodec | None = None,
        max_packet_bytes: int = MAX_PACKET_BYTES,
    ) -> None:
        if codec is None:
            from .typed_ipc import TypedEnvelopeCodec

            codec = TypedEnvelopeCodec.load()
        self.path = path
        self._handler = handler
        self._codec = codec
        self._max_packet_bytes = max_packet_bytes
        self._socket: socket.socket | None = None
        self._stop = threading.Event()
        self._response_sequence = 0
        self._sequence_lock = threading.Lock()
        self._peer_lock = threading.Lock()
        self._peers: set[socket.socket] = set()
        self._threads: set[threading.Thread] = set()

    def open(self) -> None:
        _safe_prepare_socket_path(self.path, self._codec)
        server = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        server.bind(str(self.path))
        os.chmod(self.path, 0o600)
        server.listen(8)
        server.settimeout(0.25)
        self._socket = server

    def close(self) -> None:
        self._stop.set()
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        with self._peer_lock:
            peers = tuple(self._peers)
        for peer in peers:
            try:
                peer.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            peer.close()
        try:
            current = self.path.lstat()
            if stat.S_ISSOCK(current.st_mode) and current.st_uid == os.getuid():
                self.path.unlink()
        except FileNotFoundError:
            pass

    def serve_forever(self) -> None:
        if self._socket is None:
            self.open()
        assert self._socket is not None
        while not self._stop.is_set():
            try:
                connection, _ = self._socket.accept()
            except TimeoutError:
                continue
            except OSError:
                if self._stop.is_set():
                    break
                raise
            credentials = peer_credentials(connection)
            if credentials[1] != os.getuid():
                connection.close()
                continue
            with self._peer_lock:
                self._peers.add(connection)
            thread = threading.Thread(
                target=self._peer_worker,
                args=(connection, credentials),
                daemon=True,
                name=f"rdk-ipc-peer-{credentials[0]}",
            )
            with self._peer_lock:
                self._threads.add(thread)
            thread.start()

    def _peer_worker(
        self,
        connection: socket.socket,
        credentials: tuple[int, int, int],
    ) -> None:
        try:
            with connection:
                self._serve_connection(connection, credentials)
        finally:
            with self._peer_lock:
                self._peers.discard(connection)
                self._threads.discard(threading.current_thread())
            disconnected = getattr(self._handler, "peer_disconnected", None)
            if callable(disconnected):
                disconnected(credentials)

    def _serve_connection(
        self,
        connection: socket.socket,
        credentials: tuple[int, int, int],
    ) -> None:
        connection.settimeout(0.25)
        last_request_sequence = -1
        while not self._stop.is_set():
            try:
                data, _, flags, _ = connection.recvmsg(self._max_packet_bytes)
            except TimeoutError:
                continue
            except (ConnectionError, OSError):
                return
            if not data:
                return
            if flags & socket.MSG_TRUNC:
                response_kind, response_payload = "error", {"code": "packet_too_large"}
            else:
                try:
                    kind, sequence, payload = self._codec.decode(data)
                    if sequence <= last_request_sequence:
                        raise IPCProtocolError(
                            "request sequence must increase monotonically per connection"
                        )
                    last_request_sequence = sequence
                    response_kind, response_payload = self._handler(
                        kind, sequence, payload, credentials
                    )
                except Exception as exc:
                    response_kind, response_payload = (
                        "error",
                        {"code": type(exc).__name__, "message": str(exc)},
                    )
            with self._sequence_lock:
                self._response_sequence += 1
                response_sequence = self._response_sequence
            try:
                encoded = self._codec.encode(
                    response_kind, response_sequence, response_payload
                )
            except Exception as exc:
                # A malformed handler response must not crash the peer thread
                # and leave the client parsing an empty packet as schema zero.
                # Return a bounded typed error whenever the codec itself works.
                try:
                    encoded = self._codec.encode(
                        "error",
                        response_sequence,
                        {
                            "code": "response_encode_error",
                            "message": f"{type(exc).__name__}: {exc}"[:1024],
                        },
                    )
                except Exception:
                    return
            try:
                connection.sendall(encoded)
            except (ConnectionError, OSError):
                return
