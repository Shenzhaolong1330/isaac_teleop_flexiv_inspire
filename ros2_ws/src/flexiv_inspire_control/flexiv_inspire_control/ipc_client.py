"""Thread-safe typed seqpacket client for the local RDK daemon."""

from __future__ import annotations

from pathlib import Path
import socket
import threading
from typing import Any

from flexiv_rdk_daemon.typed_ipc import TypedEnvelopeCodec


class RDKIPCClient:
    def __init__(self, path: Path, *, timeout_s: float = 0.05) -> None:
        self._path = path
        self._timeout_s = timeout_s
        self._codec = TypedEnvelopeCodec.load()
        self._socket: socket.socket | None = None
        self._sequence = 0
        self._lock = threading.Lock()

    def close(self) -> None:
        with self._lock:
            if self._socket is not None:
                self._socket.close()
                self._socket = None

    def request(
        self,
        kind: str,
        payload: dict[str, Any],
        *,
        timeout_s: float | None = None,
    ) -> tuple[str, dict[str, Any]]:
        with self._lock:
            if self._socket is None:
                client = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
                client.settimeout(self._timeout_s)
                client.connect(str(self._path))
                self._socket = client
            self._sequence += 1
            selected_timeout = self._timeout_s if timeout_s is None else timeout_s
            self._socket.settimeout(selected_timeout)
            try:
                self._socket.sendall(self._codec.encode(kind, self._sequence, payload))
                response = self._socket.recv(65536)
                if not response:
                    raise ConnectionError(
                        "RDK daemon closed IPC connection without a response"
                    )
                response_kind, _, response_payload = self._codec.decode(response)
            except Exception:
                self._socket.close()
                self._socket = None
                raise
            finally:
                if self._socket is not None:
                    self._socket.settimeout(self._timeout_s)
            if response_kind == "error":
                raise RuntimeError(
                    f"RDK daemon error {response_payload.get('code')}: "
                    f"{response_payload.get('message')}"
                )
            return response_kind, response_payload
