"""Canonical descriptors used by RPC, dataset and policy adapters."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence


class SchemaError(ValueError):
    pass


_DTYPES = frozenset(
    {
        "bool",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "int8",
        "int16",
        "int32",
        "int64",
        "float16",
        "float32",
        "float64",
        "bytes",
    }
)


def _nonempty(value: str, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise SchemaError(f"{name} must be non-empty")
    return result


def _shape(value: Sequence[int]) -> tuple[int, ...]:
    result = tuple(int(item) for item in value)
    if not result or any(item <= 0 for item in result):
        raise SchemaError("tensor shape must contain positive dimensions")
    return result


@dataclass(frozen=True)
class TensorDescriptor:
    dtype: str
    shape: tuple[int, ...]
    element_names: tuple[str, ...] = ()
    unit: str = ""

    def __post_init__(self) -> None:
        dtype = _nonempty(self.dtype, "dtype").lower()
        if dtype not in _DTYPES:
            raise SchemaError(f"unsupported dtype: {dtype}")
        shape = _shape(self.shape)
        names = tuple(str(item).strip() for item in self.element_names)
        if any(not item for item in names):
            raise SchemaError("element names must be non-empty")
        if names and (len(shape) != 1 or len(names) != shape[0]):
            raise SchemaError("element names require a one-dimensional matching tensor")
        if len(names) != len(set(names)):
            raise SchemaError("element names must be unique")
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "element_names", names)
        object.__setattr__(self, "unit", str(self.unit).strip())


@dataclass(frozen=True)
class ChannelDescriptor:
    channel_id: str
    semantic: str
    tensor: TensorDescriptor
    native_rate_hz: float
    frame_id: str = ""
    clock_domain: str = "host_monotonic"
    encodings: tuple[str, ...] = ("raw",)

    def __post_init__(self) -> None:
        rate = float(self.native_rate_hz)
        if not math.isfinite(rate) or rate <= 0.0:
            raise SchemaError("native_rate_hz must be finite and positive")
        encodings = tuple(str(item).strip().lower() for item in self.encodings)
        if not encodings or any(not item for item in encodings):
            raise SchemaError("encodings must contain non-empty values")
        object.__setattr__(self, "channel_id", _nonempty(self.channel_id, "channel_id"))
        object.__setattr__(self, "semantic", _nonempty(self.semantic, "semantic"))
        object.__setattr__(self, "native_rate_hz", rate)
        object.__setattr__(self, "frame_id", str(self.frame_id).strip())
        object.__setattr__(self, "clock_domain", _nonempty(self.clock_domain, "clock_domain"))
        object.__setattr__(self, "encodings", encodings)


@dataclass(frozen=True)
class ActionSchema:
    schema_id: str
    tensor: TensorDescriptor
    frame_id: str
    representation: str
    rate_hz: float
    relative: bool

    def __post_init__(self) -> None:
        rate = float(self.rate_hz)
        if not math.isfinite(rate) or rate <= 0.0:
            raise SchemaError("action rate_hz must be finite and positive")
        object.__setattr__(self, "schema_id", _nonempty(self.schema_id, "schema_id"))
        object.__setattr__(self, "frame_id", _nonempty(self.frame_id, "frame_id"))
        object.__setattr__(
            self, "representation", _nonempty(self.representation, "representation")
        )
        object.__setattr__(self, "rate_hz", rate)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True)
class SystemSchema:
    schema_version: int
    system_id: str
    robot_type: str
    channels: tuple[ChannelDescriptor, ...]
    action_schemas: tuple[ActionSchema, ...]
    metadata: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if int(self.schema_version) <= 0:
            raise SchemaError("schema_version must be positive")
        channels = tuple(self.channels)
        actions = tuple(self.action_schemas)
        channel_ids = [item.channel_id for item in channels]
        action_ids = [item.schema_id for item in actions]
        if len(channel_ids) != len(set(channel_ids)):
            raise SchemaError("channel IDs must be unique")
        if len(action_ids) != len(set(action_ids)):
            raise SchemaError("action schema IDs must be unique")
        metadata = {
            str(key): str(value)
            for key, value in sorted(dict(self.metadata or {}).items())
        }
        object.__setattr__(self, "schema_version", int(self.schema_version))
        object.__setattr__(self, "system_id", _nonempty(self.system_id, "system_id"))
        object.__setattr__(self, "robot_type", _nonempty(self.robot_type, "robot_type"))
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "action_schemas", actions)
        object.__setattr__(self, "metadata", metadata)

    def canonical_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def schema_hash(self) -> str:
        return hashlib.sha256(_canonical_json(self.canonical_dict())).hexdigest()

    def channel(self, channel_id: str) -> ChannelDescriptor:
        for channel in self.channels:
            if channel.channel_id == channel_id:
                return channel
        raise SchemaError(f"unknown channel: {channel_id}")

    def action(self, schema_id: str) -> ActionSchema:
        for action in self.action_schemas:
            if action.schema_id == schema_id:
                return action
        raise SchemaError(f"unknown action schema: {schema_id}")
