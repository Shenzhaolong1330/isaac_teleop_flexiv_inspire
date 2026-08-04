"""Bounded multi-channel buffers with async subscriptions and time alignment."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
import math
from typing import Iterable, Mapping

import numpy as np

from policy_contracts import ChannelDescriptor, SystemSchema


class ChannelDataError(ValueError):
    pass


_NUMPY_DTYPES = {
    "bool": np.dtype("?"),
    "uint8": np.dtype("u1"),
    "uint16": np.dtype("<u2"),
    "uint32": np.dtype("<u4"),
    "uint64": np.dtype("<u8"),
    "int8": np.dtype("i1"),
    "int16": np.dtype("<i2"),
    "int32": np.dtype("<i4"),
    "int64": np.dtype("<i8"),
    "float16": np.dtype("<f2"),
    "float32": np.dtype("<f4"),
    "float64": np.dtype("<f8"),
}


@dataclass(frozen=True)
class SampleTiming:
    source_time_ns: int
    host_receive_time_ns: int
    mapped_host_time_ns: int
    acquisition_start_ns: int
    acquisition_end_ns: int
    sequence: int
    valid: bool
    invalid_reason: str = ""
    source_clock_domain: str = ""
    host_clock_domain: str = "host_monotonic"
    timing_valid: bool = True

    @property
    def alignment_time_ns(self) -> int:
        if self.timing_valid and self.mapped_host_time_ns > 0:
            return self.mapped_host_time_ns
        return self.host_receive_time_ns


@dataclass(frozen=True)
class ChannelSample:
    channel_id: str
    timing: SampleTiming
    tensor: np.ndarray | None = None
    image_encoding: str = ""
    image_data: bytes = b""
    image_width: int = 0
    image_height: int = 0
    image_channels: int = 0

    @property
    def is_image(self) -> bool:
        return bool(self.image_encoding)


@dataclass(frozen=True)
class StoredSample:
    broker_sequence: int
    sample: ChannelSample


def tensor_sample(
    channel_id: str,
    values,
    timing: SampleTiming,
    descriptor: ChannelDescriptor,
) -> ChannelSample:
    try:
        dtype = _NUMPY_DTYPES[descriptor.tensor.dtype]
    except KeyError as exc:
        raise ChannelDataError(f"channel {channel_id} is not a numeric tensor") from exc
    tensor = np.asarray(values, dtype=dtype)
    if tensor.shape != descriptor.tensor.shape:
        raise ChannelDataError(
            f"channel {channel_id} expected {descriptor.tensor.shape}, got {tensor.shape}"
        )
    if tensor.dtype.kind == "f" and not np.all(np.isfinite(tensor)):
        raise ChannelDataError(f"channel {channel_id} contains NaN/Inf")
    tensor = np.ascontiguousarray(tensor)
    tensor.setflags(write=False)
    return ChannelSample(channel_id=channel_id, timing=timing, tensor=tensor)


def image_sample(
    channel_id: str,
    data: bytes,
    *,
    encoding: str,
    width: int,
    height: int,
    channels: int,
    timing: SampleTiming,
) -> ChannelSample:
    if not data:
        raise ChannelDataError(f"channel {channel_id} image is empty")
    if width <= 0 or height <= 0 or channels <= 0:
        raise ChannelDataError(f"channel {channel_id} image dimensions are invalid")
    return ChannelSample(
        channel_id=channel_id,
        timing=timing,
        image_encoding=str(encoding).lower(),
        image_data=bytes(data),
        image_width=int(width),
        image_height=int(height),
        image_channels=int(channels),
    )


class ChannelBroker:
    def __init__(self, schema: SystemSchema, *, buffer_seconds: float = 2.0) -> None:
        if not math.isfinite(buffer_seconds) or buffer_seconds <= 0.0:
            raise ValueError("buffer_seconds must be finite and positive")
        self.schema = schema
        self._descriptors = {item.channel_id: item for item in schema.channels}
        self._buffers = {
            item.channel_id: deque(
                maxlen=max(4, int(math.ceil(item.native_rate_hz * buffer_seconds)) + 2)
            )
            for item in schema.channels
        }
        self._channel_sequences = {item.channel_id: 0 for item in schema.channels}
        self._condition = asyncio.Condition()
        self.dropped_by_channel = {item.channel_id: 0 for item in schema.channels}

    def descriptor(self, channel_id: str) -> ChannelDescriptor:
        try:
            return self._descriptors[channel_id]
        except KeyError as exc:
            raise ChannelDataError(f"unknown channel: {channel_id}") from exc

    def _validate(self, sample: ChannelSample) -> None:
        descriptor = self.descriptor(sample.channel_id)
        if sample.is_image:
            if sample.image_encoding not in descriptor.encodings:
                raise ChannelDataError(
                    f"channel {sample.channel_id} encoding {sample.image_encoding} is unsupported"
                )
            expected_height, expected_width, expected_channels = descriptor.tensor.shape
            if (
                sample.image_width,
                sample.image_height,
                sample.image_channels,
            ) != (expected_width, expected_height, expected_channels):
                raise ChannelDataError(f"channel {sample.channel_id} image shape changed")
            return
        if sample.tensor is None:
            raise ChannelDataError(f"channel {sample.channel_id} has no payload")
        expected_dtype = _NUMPY_DTYPES.get(descriptor.tensor.dtype)
        if expected_dtype is None or sample.tensor.dtype != expected_dtype:
            raise ChannelDataError(f"channel {sample.channel_id} dtype mismatch")
        if sample.tensor.shape != descriptor.tensor.shape:
            raise ChannelDataError(f"channel {sample.channel_id} shape mismatch")

    async def publish(self, sample: ChannelSample) -> int:
        return (await self.publish_many((sample,)))[sample.channel_id]

    async def publish_many(self, samples: Iterable[ChannelSample]) -> dict[str, int]:
        materialized = tuple(samples)
        if not materialized:
            return {}
        for sample in materialized:
            self._validate(sample)
        positions: dict[str, int] = {}
        async with self._condition:
            for sample in materialized:
                channel_id = sample.channel_id
                buffer = self._buffers[channel_id]
                if len(buffer) == buffer.maxlen:
                    self.dropped_by_channel[channel_id] += 1
                self._channel_sequences[channel_id] += 1
                position = self._channel_sequences[channel_id]
                buffer.append(StoredSample(position, sample))
                positions[channel_id] = position
            self._condition.notify_all()
        return positions

    async def next_after(
        self,
        positions: Mapping[str, int],
        *,
        reliable_channels: frozenset[str] = frozenset(),
    ) -> tuple[tuple[StoredSample, ...], dict[str, int]]:
        channel_ids = tuple(positions)
        for channel_id in channel_ids:
            self.descriptor(channel_id)
        async with self._condition:
            await self._condition.wait_for(
                lambda: any(
                    self._channel_sequences[channel_id] > positions[channel_id]
                    for channel_id in channel_ids
                )
            )
            output: list[StoredSample] = []
            updated = dict(positions)
            for channel_id in channel_ids:
                unseen = [
                    item
                    for item in self._buffers[channel_id]
                    if item.broker_sequence > positions[channel_id]
                ]
                if not unseen:
                    continue
                if channel_id in reliable_channels:
                    oldest = self._buffers[channel_id][0].broker_sequence
                    if positions[channel_id] and oldest > positions[channel_id] + 1:
                        raise ChannelDataError(
                            f"reliable subscriber overran channel {channel_id}"
                        )
                    output.extend(unseen)
                else:
                    output.append(unseen[-1])
                updated[channel_id] = unseen[-1].broker_sequence
            output.sort(key=lambda item: item.sample.timing.alignment_time_ns)
            return tuple(output), updated

    async def current_positions(self, channel_ids: Iterable[str]) -> dict[str, int]:
        selected = tuple(channel_ids)
        for channel_id in selected:
            self.descriptor(channel_id)
        async with self._condition:
            return {
                channel_id: self._channel_sequences[channel_id]
                for channel_id in selected
            }

    async def sample_at(
        self,
        channel_id: str,
        target_ns: int,
        *,
        alignment: str,
        tolerance_ns: int,
        max_age_ns: int,
    ) -> ChannelSample:
        descriptor = self.descriptor(channel_id)
        async with self._condition:
            values = tuple(self._buffers[channel_id])
        if not values:
            raise ChannelDataError(f"channel {channel_id} has no samples")
        mode = alignment.lower()
        if mode == "latest_causal":
            candidates = [
                item for item in values if item.sample.timing.alignment_time_ns <= target_ns
            ]
            if not candidates:
                raise ChannelDataError(f"channel {channel_id} has no causal sample")
            selected = candidates[-1].sample
            distance = target_ns - selected.timing.alignment_time_ns
        elif mode == "nearest":
            selected = min(
                values,
                key=lambda item: (
                    abs(item.sample.timing.alignment_time_ns - target_ns),
                    item.sample.timing.alignment_time_ns > target_ns,
                ),
            ).sample
            distance = abs(target_ns - selected.timing.alignment_time_ns)
        elif mode == "interpolate":
            selected = self._interpolate(descriptor, values, target_ns)
            distance = 0
        else:
            raise ChannelDataError(f"unsupported alignment mode: {alignment}")
        if tolerance_ns > 0 and distance > tolerance_ns:
            raise ChannelDataError(
                f"channel {channel_id} exceeds alignment tolerance: {distance}ns"
            )
        age = max(0, target_ns - selected.timing.alignment_time_ns)
        if max_age_ns > 0 and age > max_age_ns:
            raise ChannelDataError(f"channel {channel_id} is stale: {age}ns")
        return selected

    @staticmethod
    def _interpolate(
        descriptor: ChannelDescriptor,
        values: tuple[StoredSample, ...],
        target_ns: int,
    ) -> ChannelSample:
        left = [item.sample for item in values if item.sample.timing.alignment_time_ns <= target_ns]
        right = [item.sample for item in values if item.sample.timing.alignment_time_ns >= target_ns]
        if not left or not right:
            raise ChannelDataError(
                f"channel {descriptor.channel_id} does not bracket target time"
            )
        before, after = left[-1], right[0]
        if before.timing.alignment_time_ns == after.timing.alignment_time_ns:
            return before
        if before.is_image or after.is_image or before.tensor is None or after.tensor is None:
            raise ChannelDataError(f"channel {descriptor.channel_id} cannot be interpolated")
        if before.tensor.dtype.kind != "f" or after.tensor.dtype.kind != "f":
            raise ChannelDataError(f"channel {descriptor.channel_id} is not floating point")
        if not before.timing.valid or not after.timing.valid:
            raise ChannelDataError(f"channel {descriptor.channel_id} bracket is invalid")
        start = before.timing.alignment_time_ns
        end = after.timing.alignment_time_ns
        alpha = (target_ns - start) / (end - start)
        if descriptor.semantic == "cartesian_pose_xyzw":
            tensor = np.empty(7, dtype=before.tensor.dtype)
            tensor[:3] = before.tensor[:3] + alpha * (after.tensor[:3] - before.tensor[:3])
            tensor[3:] = _slerp(before.tensor[3:], after.tensor[3:], alpha)
        else:
            tensor = before.tensor + alpha * (after.tensor - before.tensor)
            tensor = tensor.astype(before.tensor.dtype, copy=False)
        tensor = np.ascontiguousarray(tensor)
        tensor.setflags(write=False)
        timing = SampleTiming(
            source_time_ns=target_ns,
            host_receive_time_ns=max(
                before.timing.host_receive_time_ns, after.timing.host_receive_time_ns
            ),
            mapped_host_time_ns=target_ns,
            acquisition_start_ns=before.timing.acquisition_start_ns,
            acquisition_end_ns=after.timing.acquisition_end_ns,
            sequence=max(before.timing.sequence, after.timing.sequence),
            valid=True,
            invalid_reason=(
                f"interpolated:{before.timing.sequence}:{after.timing.sequence}"
            ),
            source_clock_domain=before.timing.source_clock_domain,
            host_clock_domain=before.timing.host_clock_domain,
            timing_valid=True,
        )
        return ChannelSample(descriptor.channel_id, timing, tensor=tensor)


def _slerp(left, right, alpha: float) -> np.ndarray:
    first = np.array(left, dtype=np.float64, copy=True)
    second = np.array(right, dtype=np.float64, copy=True)
    first /= np.linalg.norm(first)
    second /= np.linalg.norm(second)
    dot = float(np.dot(first, second))
    if dot < 0.0:
        second = -second
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        output = first + alpha * (second - first)
        return output / np.linalg.norm(output)
    angle = math.acos(dot)
    output = (
        math.sin((1.0 - alpha) * angle) * first
        + math.sin(alpha * angle) * second
    ) / math.sin(angle)
    return output / np.linalg.norm(output)
