"""Small Modbus/TCP transport with an explicit read-only default.

The validation path constructs :class:`ReadOnlyModbusTcpClient`, which only
implements function 0x03.  Motion writes live in a distinct subclass and
require a process-local permit created by the safety supervisor.
"""

from __future__ import annotations

from dataclasses import dataclass
import secrets
import socket
import struct
import threading
from typing import Callable


class ModbusError(RuntimeError):
    pass


class ModbusConnectionError(ModbusError):
    pass


class ModbusProtocolError(ModbusError):
    pass


class WriteNotAuthorized(ModbusError):
    pass


SocketFactory = Callable[..., socket.socket]


class ReadOnlyModbusTcpClient:
    READ_HOLDING_REGISTERS = 0x03

    def __init__(
        self,
        host: str,
        port: int = 6000,
        *,
        unit_id: int = 0xFF,
        timeout_s: float = 0.25,
        socket_factory: SocketFactory = socket.socket,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.unit_id = int(unit_id)
        self.timeout_s = float(timeout_s)
        self._socket_factory = socket_factory
        self._socket: socket.socket | None = None
        self._transaction_id = 0
        self._lock = threading.Lock()

    def connect(self) -> None:
        if self._socket is not None:
            return
        try:
            sock = self._socket_factory(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout_s)
            sock.connect((self.host, self.port))
        except OSError as exc:
            raise ModbusConnectionError(
                f"failed to connect to {self.host}:{self.port}: {exc}"
            ) from exc
        self._socket = sock

    def close(self) -> None:
        sock, self._socket = self._socket, None
        if sock is not None:
            sock.close()

    def read_raw(self, byte_address: int, word_count: int) -> bytes:
        if not 0 <= byte_address <= 0xFFFF:
            raise ValueError("byte_address must fit uint16")
        if not 1 <= word_count <= 125:
            raise ValueError("word_count must be in 1..125")
        payload = struct.pack(">HH", byte_address, word_count)
        with self._lock:
            return self._request(self.READ_HOLDING_REGISTERS, payload, word_count * 2)

    def _request(self, function: int, payload: bytes, expected_bytes: int) -> bytes:
        if self._socket is None:
            raise ModbusConnectionError("client is not connected")
        self._transaction_id = (self._transaction_id % 0xFFFF) + 1
        pdu = bytes((function,)) + payload
        request = struct.pack(
            ">HHHB", self._transaction_id, 0, len(pdu) + 1, self.unit_id
        ) + pdu
        try:
            self._socket.sendall(request)
            header = self._recv_exact(7)
            tid, protocol, length, unit = struct.unpack(">HHHB", header)
            response = self._recv_exact(length - 1)
        except OSError as exc:
            self.close()
            raise ModbusConnectionError(str(exc)) from exc
        if tid != self._transaction_id or protocol != 0 or unit != self.unit_id:
            raise ModbusProtocolError("invalid MBAP response")
        if not response:
            raise ModbusProtocolError("empty PDU")
        response_function = response[0]
        if response_function == (function | 0x80):
            code = response[1] if len(response) > 1 else -1
            raise ModbusProtocolError(f"device exception {code}")
        if response_function != function:
            raise ModbusProtocolError("function code mismatch")
        body = response[1:]
        if function == self.READ_HOLDING_REGISTERS:
            if not body or body[0] != expected_bytes or len(body[1:]) != expected_bytes:
                raise ModbusProtocolError("invalid read byte count")
            return body[1:]
        return body

    def _recv_exact(self, size: int) -> bytes:
        assert self._socket is not None
        chunks = bytearray()
        while len(chunks) < size:
            part = self._socket.recv(size - len(chunks))
            if not part:
                raise ModbusConnectionError("connection closed")
            chunks.extend(part)
        return bytes(chunks)

    def __enter__(self) -> "ReadOnlyModbusTcpClient":
        self.connect()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


@dataclass(frozen=True)
class LocalWritePermit:
    """Unforgeable-in-configuration token issued by the local safety process."""

    session_id: str
    _nonce: str

    @classmethod
    def issue_after_local_authorization(
        cls, session_id: str, confirmation: str
    ) -> "LocalWritePermit":
        if confirmation != "DFTP-LOCAL-CONTROL-AUTHORIZED":
            raise WriteNotAuthorized("local DFTP authorization confirmation is missing")
        if not session_id:
            raise WriteNotAuthorized("session_id is required")
        return cls(session_id=session_id, _nonce=secrets.token_hex(16))


class CommandCapableModbusTcpClient(ReadOnlyModbusTcpClient):
    WRITE_SINGLE_REGISTER = 0x06
    WRITE_MULTIPLE_REGISTERS = 0x10

    def __init__(self, *args: object, permit: LocalWritePermit, **kwargs: object) -> None:
        if not isinstance(permit, LocalWritePermit) or not permit._nonce:
            raise WriteNotAuthorized("a valid local write permit is required")
        super().__init__(*args, **kwargs)
        self._permit = permit

    def write_single_u16(self, byte_address: int, value: int) -> None:
        if not 0 <= byte_address <= 0xFFFF:
            raise ValueError("byte_address must fit uint16")
        if not 0 <= value <= 0xFFFF:
            raise ValueError("value must fit uint16")
        request = struct.pack(">HH", byte_address, value)
        with self._lock:
            response = self._request(self.WRITE_SINGLE_REGISTER, request, 0)
        if response != request:
            raise ModbusProtocolError("write response did not echo address/value")

    def write_i16(self, byte_address: int, values: tuple[int, ...]) -> None:
        if not values or len(values) > 123:
            raise ValueError("values must contain 1..123 words")
        encoded = b"".join((value & 0xFFFF).to_bytes(2, "big") for value in values)
        request = struct.pack(">HHB", byte_address, len(values), len(encoded)) + encoded
        with self._lock:
            response = self._request(self.WRITE_MULTIPLE_REGISTERS, request, 0)
        expected = struct.pack(">HH", byte_address, len(values))
        if response != expected:
            raise ModbusProtocolError("write response did not echo address/count")
