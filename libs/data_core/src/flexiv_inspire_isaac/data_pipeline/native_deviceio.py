"""Secure local ingress for producer-native DeviceIO RecordEnvelopes."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import socket
import stat
import struct
import threading
from typing import Any, Callable, Mapping

from isaac_teleop_core.deviceio import DEFAULT_MAX_DATAGRAM, FRAME_HEADER


REQUIRED_FIELDS = {
    "schema_version",
    "producer",
    "topic",
    "source_time_ns",
    "host_receive_time_ns",
    "sequence",
    "valid",
    "source_clock_domain",
    "host_clock_domain",
    "mapped_host_time_ns",
    "timing_valid",
    "payload",
}


@dataclass(frozen=True)
class NativeIngressStats:
    received: int
    invalid: int
    oversized: int
    callback_failures: int


def validate_native_envelope(document: Any) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise ValueError("native DeviceIO frame must be a JSON object")
    missing = REQUIRED_FIELDS.difference(document)
    if missing:
        raise ValueError(f"native DeviceIO envelope missing {sorted(missing)}")
    if int(document["schema_version"]) != 1:
        raise ValueError("unsupported native DeviceIO schema")
    if not str(document["producer"]).strip():
        raise ValueError("native DeviceIO producer is empty")
    if not str(document["topic"]).startswith("/"):
        raise ValueError("native DeviceIO topic must be absolute")
    for field in ("source_time_ns", "host_receive_time_ns", "sequence"):
        if int(document[field]) < 0:
            raise ValueError(f"native DeviceIO {field} must be non-negative")
    mapped = document["mapped_host_time_ns"]
    if bool(document["timing_valid"]) and mapped is None:
        raise ValueError("timing_valid requires mapped_host_time_ns")
    return document


class NativeDeviceIOIngress:
    """Length-framed AF_UNIX stream receiver with a private endpoint."""

    def __init__(
        self,
        socket_path: str | Path,
        callback: Callable[[Mapping[str, Any]], None],
        *,
        max_datagram_bytes: int = DEFAULT_MAX_DATAGRAM,
    ) -> None:
        self.socket_path = Path(socket_path)
        self.callback = callback
        self.max_datagram_bytes = int(max_datagram_bytes)
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._peer_lock = threading.Lock()
        self._peers: set[socket.socket] = set()
        self._workers: set[threading.Thread] = set()
        self._received = 0
        self._invalid = 0
        self._oversized = 0
        self._callback_failures = 0
        self._fatal: Exception | None = None

    def start(self) -> None:
        if self._socket is not None:
            raise RuntimeError("native DeviceIO ingress already started")
        self.socket_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        parent_info = self.socket_path.parent.stat()
        if parent_info.st_uid != os.getuid():
            raise RuntimeError("DeviceIO socket directory is owned by another user")
        os.chmod(self.socket_path.parent, 0o700)
        if self.socket_path.exists() or self.socket_path.is_symlink():
            info = self.socket_path.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISSOCK(info.st_mode):
                raise RuntimeError(
                    f"refusing to replace non-socket DeviceIO path {self.socket_path}"
                )
            if info.st_uid != os.getuid():
                raise RuntimeError("existing DeviceIO socket is owned by another user")
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            probe.settimeout(0.2)
            try:
                probe.connect(str(self.socket_path))
            except ConnectionRefusedError:
                pass
            except OSError as exc:
                raise RuntimeError(
                    f"cannot safely replace existing DeviceIO socket {self.socket_path}: {exc}"
                ) from exc
            else:
                raise RuntimeError(
                    f"an active DeviceIO collector already owns {self.socket_path}"
                )
            finally:
                probe.close()
            after = self.socket_path.lstat()
            if after.st_dev != info.st_dev or after.st_ino != info.st_ino:
                raise RuntimeError("DeviceIO socket changed during stale-path probe")
            self.socket_path.unlink()
        receiver = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        receiver.bind(str(self.socket_path))
        os.chmod(self.socket_path, 0o600)
        receiver.listen(16)
        receiver.settimeout(0.1)
        self._socket = receiver
        self._thread = threading.Thread(
            target=self._run, name="native-deviceio-ingress", daemon=True
        )
        self._thread.start()

    def stats(self) -> NativeIngressStats:
        with self._lock:
            return NativeIngressStats(
                self._received,
                self._invalid,
                self._oversized,
                self._callback_failures,
            )

    def check_health(self) -> None:
        with self._lock:
            if self._fatal is not None:
                raise RuntimeError(f"native DeviceIO ingress callback failed: {self._fatal}")

    def close(self, timeout_s: float = 2.0) -> None:
        self._stop.set()
        if self._socket is not None:
            self._socket.close()
        with self._peer_lock:
            peers = tuple(self._peers)
        for peer in peers:
            try:
                peer.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            peer.close()
        if self._thread is not None:
            self._thread.join(timeout_s)
            if self._thread.is_alive():
                raise TimeoutError("native DeviceIO ingress did not stop")
        with self._peer_lock:
            deadline_workers = tuple(self._workers)
        for worker in deadline_workers:
            worker.join(timeout_s)
        if any(worker.is_alive() for worker in deadline_workers):
            raise TimeoutError("native DeviceIO peer worker did not stop")
        if self._socket is not None:
            self._socket = None
        if self.socket_path.exists() and not self.socket_path.is_symlink():
            info = self.socket_path.lstat()
            if stat.S_ISSOCK(info.st_mode) and info.st_uid == os.getuid():
                self.socket_path.unlink()
        if self._fatal is not None:
            raise RuntimeError(f"native DeviceIO ingress callback failed: {self._fatal}")

    def _run(self) -> None:
        assert self._socket is not None
        while not self._stop.is_set():
            try:
                connection, _ = self._socket.accept()
            except socket.timeout:
                continue
            except OSError:
                if self._stop.is_set():
                    return
                raise
            credentials = connection.getsockopt(
                socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")
            )
            _pid, uid, _gid = struct.unpack("3i", credentials)
            if uid != os.getuid():
                connection.close()
                continue
            connection.settimeout(0.1)
            worker = threading.Thread(
                target=self._serve_peer,
                args=(connection,),
                name="native-deviceio-peer",
                daemon=True,
            )
            with self._peer_lock:
                self._peers.add(connection)
                self._workers.add(worker)
            worker.start()

    def _serve_peer(self, connection: socket.socket) -> None:
        try:
            while not self._stop.is_set():
                header = self._receive_exact(connection, FRAME_HEADER.size)
                if header is None:
                    return
                length = FRAME_HEADER.unpack(header)[0]
                if length > self.max_datagram_bytes:
                    with self._lock:
                        self._oversized += 1
                    return
                payload = self._receive_exact(connection, length)
                if payload is None:
                    with self._lock:
                        self._invalid += 1
                    return
                self._accept_payload(payload)
        finally:
            with self._peer_lock:
                self._peers.discard(connection)
                self._workers.discard(threading.current_thread())
            connection.close()

    def _receive_exact(
        self, connection: socket.socket, size: int
    ) -> bytes | None:
        chunks = bytearray()
        while len(chunks) < size and not self._stop.is_set():
            try:
                block = connection.recv(size - len(chunks))
            except socket.timeout:
                continue
            except OSError:
                return None
            if not block:
                return None
            chunks.extend(block)
        return bytes(chunks) if len(chunks) == size else None

    def _accept_payload(self, payload: bytes) -> None:
        try:
            document = validate_native_envelope(json.loads(payload))
        except Exception:
            with self._lock:
                self._invalid += 1
            return
        try:
            self.callback(document)
        except Exception as exc:
            with self._lock:
                self._callback_failures += 1
                self._fatal = exc
            self._stop.set()
            return
        with self._lock:
            self._received += 1
