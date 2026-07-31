"""Typed protobuf codec generated from the canonical RDK IPC schema."""

from __future__ import annotations

import importlib
import time
from typing import Any, Mapping

from google.protobuf.json_format import MessageToDict, ParseDict

from .ipc import IPCProtocolError, MAX_PACKET_BYTES, SCHEMA_VERSION

_KIND_TO_FIELD = {
    "hello": "hello",
    "observe": "observe_request",
    "dual_arm_state": "dual_arm_state",
    "hand_observation": "hand_observation",
    "cartesian_command": "cartesian_command",
    "command_ack": "command_ack",
    "authorize_zero_ft": "authorize_zero_ft_request",
    "authorize_zero_ft_result": "authorize_zero_ft_result",
    "authorize_control": "authorize_control_request",
    "authorize_control_result": "authorize_control_result",
    "authorize_home": "authorize_home_request",
    "authorize_home_result": "authorize_home_result",
    "home_command": "home_command",
    "home_result": "home_result",
    "zero_ft": "zero_ft_request",
    "zero_ft_result": "zero_ft_result",
    "hold": "hold_request",
    "error": "error",
}
_FIELD_TO_KIND = {field: kind for kind, field in _KIND_TO_FIELD.items()}


class TypedEnvelopeCodec:
    """Lossless codec for uint64 timestamps and strongly typed payloads."""

    def __init__(self, pb2: Any) -> None:
        self._pb2 = pb2

    @classmethod
    def load(cls) -> "TypedEnvelopeCodec":
        try:
            pb2 = importlib.import_module(
                "flexiv_rdk_daemon.generated.rdk_ipc_pb2"
            )
        except ImportError as exc:
            raise RuntimeError(
                "typed RDK IPC bindings are missing; run "
                "apps/flexiv_daemon/scripts/generate_proto.sh during the build"
            ) from exc
        return cls(pb2)

    def encode(self, kind: str, sequence: int, payload: Mapping[str, Any]) -> bytes:
        try:
            field = _KIND_TO_FIELD[kind]
        except KeyError as exc:
            raise IPCProtocolError(f"unsupported typed packet kind {kind!r}") from exc
        if sequence < 0:
            raise IPCProtocolError("sequence cannot be negative")
        envelope = self._pb2.Envelope(
            schema_version=SCHEMA_VERSION,
            sequence=sequence,
            monotonic_ns=time.monotonic_ns(),
        )
        message = getattr(envelope, field)
        try:
            ParseDict(dict(payload), message, ignore_unknown_fields=False)
            # ParseDict({}) does not mutate an empty submessage, so protobuf
            # otherwise leaves its enclosing oneof unset. SetInParent marks
            # presence for empty requests such as ObserveRequest.
            message.SetInParent()
        except Exception as exc:
            raise IPCProtocolError(f"payload does not match {field}") from exc
        data = envelope.SerializeToString(deterministic=True)
        if len(data) > MAX_PACKET_BYTES:
            raise IPCProtocolError("encoded packet exceeds 64 KiB")
        return data

    def decode(self, data: bytes) -> tuple[str, int, dict[str, Any]]:
        if len(data) > MAX_PACKET_BYTES:
            raise IPCProtocolError("packet exceeds 64 KiB")
        envelope = self._pb2.Envelope()
        try:
            envelope.ParseFromString(data)
        except Exception as exc:
            raise IPCProtocolError("invalid typed protobuf packet") from exc
        if envelope.schema_version != SCHEMA_VERSION:
            raise IPCProtocolError("unsupported IPC schema version")
        field = envelope.WhichOneof("payload")
        if field is None or field not in _FIELD_TO_KIND:
            raise IPCProtocolError("typed envelope has no recognized payload")
        payload = MessageToDict(
            getattr(envelope, field),
            preserving_proto_field_name=True,
            always_print_fields_with_no_presence=True,
        )
        return _FIELD_TO_KIND[field], int(envelope.sequence), payload
