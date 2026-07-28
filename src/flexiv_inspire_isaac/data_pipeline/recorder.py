"""Priority-aware asynchronous MCAP writer.

Episode/control/maintenance events use an unbounded critical queue. Native-rate
sensor records use a bounded queue with explicit drop-old statistics.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Protocol


@dataclass(frozen=True)
class RecordEnvelope:
    topic: str
    source_time_ns: int
    host_receive_time_ns: int
    sequence: int
    valid: bool
    payload: Any
    invalid_reason: str = ""
    source_clock_domain: str = "host_monotonic"
    host_clock_domain: str = "host_monotonic"
    mapped_host_time_ns: int | None = None

    @property
    def effective_mapped_host_time_ns(self) -> int | None:
        if self.mapped_host_time_ns is not None:
            return self.mapped_host_time_ns
        if self.source_clock_domain == self.host_clock_domain:
            return self.source_time_ns
        return None

    def _timing(self) -> tuple[int | None, bool, str]:
        mapped = self.effective_mapped_host_time_ns
        if mapped is None:
            return None, False, "source-clock-unmapped"
        age = self.host_receive_time_ns - mapped
        if age < 0:
            return None, False, "mapped-source-is-after-host-receive"
        return age, True, ""

    def json_bytes(self) -> bytes:
        document = asdict(self)
        document["mapped_host_time_ns"] = self.effective_mapped_host_time_ns
        age, timing_valid, timing_reason = self._timing()
        document["age_ns"] = age
        document["timing_valid"] = timing_valid
        document["timing_invalid_reason"] = timing_reason
        return json.dumps(
            document, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")


class RecordSink(Protocol):
    def write(self, envelope: RecordEnvelope) -> None: ...
    def spool_critical(self, envelope: RecordEnvelope) -> Path:
        """Durably preserve a critical event if the MCAP write path fails."""
        spool = self.path.with_suffix(self.path.suffix + ".critical-spool.jsonl")
        fd = os.open(spool, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, envelope.json_bytes() + b"\n")
            os.fsync(fd)
        finally:
            os.close(fd)
        return spool

    def close(self) -> None: ...


class McapJsonSink:
    """MCAP sink using one JSON schema and per-topic channels."""

    def __init__(self, path: str | Path, *, profile: str = "flexiv-inspire-v1") -> None:
        try:
            from mcap.writer import Writer
        except ImportError as exc:
            raise RuntimeError("install 'mcap' in envs/ros-py312") from exc
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("wb")
        self._writer = Writer(self._file)
        self._writer.start(profile=profile, library="isaac_teleop_flexiv_inspire")
        schema = {
            "type": "object",
            "required": [
                "topic",
                "source_time_ns",
                "host_receive_time_ns",
                "sequence",
                "valid",
                "payload",
            ],
            "properties": {
                "topic": {"type": "string"},
                "source_time_ns": {"type": "integer"},
                "host_receive_time_ns": {"type": "integer"},
                "sequence": {"type": "integer"},
                "valid": {"type": "boolean"},
                "invalid_reason": {"type": "string"},
                "source_clock_domain": {"type": "string"},
                "host_clock_domain": {"type": "string"},
                "mapped_host_time_ns": {"type": ["integer", "null"]},
                "age_ns": {"type": ["integer", "null"]},
                "timing_valid": {"type": "boolean"},
                "timing_invalid_reason": {"type": "string"},
                "payload": {},
            },
        }
        self._schema_id = self._writer.register_schema(
            name="flexiv_inspire.RecordEnvelope.v1",
            encoding="jsonschema",
            data=json.dumps(schema, separators=(",", ":")).encode(),
        )
        self._channels: dict[str, int] = {}
        self._closed = False

    def write(self, envelope: RecordEnvelope) -> None:
        channel = self._channels.get(envelope.topic)
        if channel is None:
            channel = self._writer.register_channel(
                topic=envelope.topic,
                message_encoding="json",
                schema_id=self._schema_id,
            )
            self._channels[envelope.topic] = channel
        self._writer.add_message(
            channel_id=channel,
            log_time=envelope.host_receive_time_ns,
            publish_time=(
                envelope.host_receive_time_ns
                if envelope.effective_mapped_host_time_ns is None
                else envelope.effective_mapped_host_time_ns
            ),
            sequence=envelope.sequence,
            data=envelope.json_bytes(),
        )

    def spool_critical(self, envelope: RecordEnvelope) -> Path:
        """Durably preserve a critical event if the MCAP write path fails."""
        spool = self.path.with_suffix(self.path.suffix + ".critical-spool.jsonl")
        fd = os.open(spool, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, envelope.json_bytes() + b"\n")
            os.fsync(fd)
        finally:
            os.close(fd)
        return spool

    def close(self) -> None:
        if not self._closed:
            self._writer.finish()
            self._file.flush()
            os.fsync(self._file.fileno())
            self._file.close()
            self._closed = True


class AsyncMcapRecorder:
    def __init__(self, sink: RecordSink, *, sensor_capacity: int = 8192) -> None:
        if sensor_capacity <= 0:
            raise ValueError("sensor_capacity must be positive")
        self.sink = sink
        self.sensor_capacity = sensor_capacity
        self._critical: deque[RecordEnvelope] = deque()
        self._sensor: deque[RecordEnvelope] = deque()
        self._condition = threading.Condition()
        self._closing = False
        self._thread: threading.Thread | None = None
        self.written_by_topic: dict[str, int] = defaultdict(int)
        self.dropped_by_topic: dict[str, int] = defaultdict(int)
        self.write_errors: list[str] = []
        self._fatal_error: str | None = None
        self._failed_critical: RecordEnvelope | None = None
        self._critical_spool: str = ""

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("recorder already started")
        self._thread = threading.Thread(
            target=self._writer_loop, name="async-mcap-writer", daemon=True
        )
        self._thread.start()

    def submit(self, envelope: RecordEnvelope, *, critical: bool = False) -> None:
        with self._condition:
            if self._fatal_error is not None:
                raise RuntimeError(f"recorder failed: {self._fatal_error}")
            if self._closing:
                raise RuntimeError("recorder is closing")
            if critical:
                self._critical.append(envelope)
            else:
                if len(self._sensor) >= self.sensor_capacity:
                    dropped = self._sensor.popleft()
                    self.dropped_by_topic[dropped.topic] += 1
                self._sensor.append(envelope)
            self._condition.notify()

    def close(self, timeout_s: float = 10.0) -> None:
        with self._condition:
            self._closing = True
            self._condition.notify_all()
        thread = self._thread
        if thread is not None:
            thread.join(timeout_s)
            if thread.is_alive():
                # Keep the live thread reference and sink open: closing either
                # here could race an in-flight write and corrupt the MCAP tail.
                raise TimeoutError("MCAP writer did not drain before timeout")
            self._thread = None
        self.sink.close()
        if self.write_errors:
            raise RuntimeError("; ".join(self.write_errors))

    def stats(self) -> dict:
        return {
            "written_by_topic": dict(self.written_by_topic),
            "dropped_by_topic": dict(self.dropped_by_topic),
            "write_errors": list(self.write_errors),
            "fatal_error": self._fatal_error,
            "queued_critical": len(self._critical),
            "critical_spool": self._critical_spool,
        }

    def _writer_loop(self) -> None:
        while True:
            is_critical = False
            with self._condition:
                self._condition.wait_for(
                    lambda: self._critical or self._sensor or self._closing
                )
                if self._critical:
                    envelope = self._critical.popleft()
                    is_critical = True
                elif self._sensor:
                    envelope = self._sensor.popleft()
                elif self._closing:
                    return
                else:
                    continue
            try:
                envelope.json_bytes()
                self.sink.write(envelope)
                self.written_by_topic[envelope.topic] += 1
            except Exception as exc:
                error = f"{envelope.topic}: {exc}"
                self.write_errors.append(error)
                if not is_critical:
                    continue
                with self._condition:
                    self._failed_critical = envelope
                    self._critical.appendleft(envelope)
                    self._fatal_error = error
                    spool = getattr(self.sink, "spool_critical", None)
                    if callable(spool):
                        try:
                            self._critical_spool = str(spool(envelope))
                        except Exception as spool_exc:
                            self.write_errors.append(
                                f"critical spool failed: {spool_exc}"
                            )
                    self._closing = True
                    self._condition.notify_all()
                return
