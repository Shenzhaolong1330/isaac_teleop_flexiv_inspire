"""Secure local ingress for producer-native DeviceIO RecordEnvelopes."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import socket
import stat
import threading
from typing import Any, Callable, Mapping

from isaac_teleop_core.deviceio import DEFAULT_MAX_DATAGRAM


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
        raise ValueError("native DeviceIO datagram must be a JSON object")
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
    """AF_UNIX/SOCK_DGRAM receiver with mode-0600 endpoint ownership."""

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
            self.socket_path.unlink()
        receiver = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        receiver.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
        receiver.bind(str(self.socket_path))
        os.chmod(self.socket_path, 0o600)
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
        if self._thread is not None:
            self._thread.join(timeout_s)
            if self._thread.is_alive():
                raise TimeoutError("native DeviceIO ingress did not stop")
        if self._socket is not None:
            self._socket.close()
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
                payload, _ = self._socket.recvfrom(self.max_datagram_bytes + 1)
            except socket.timeout:
                continue
            except OSError:
                if self._stop.is_set():
                    return
                raise
            if len(payload) > self.max_datagram_bytes:
                with self._lock:
                    self._oversized += 1
                continue
            try:
                document = validate_native_envelope(json.loads(payload))
            except Exception:
                with self._lock:
                    self._invalid += 1
                continue
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
