"""Non-blocking native DeviceIO transport used before the ROS/DDS boundary.

Producers enqueue JSON-compatible RecordEnvelope dictionaries. A background
thread sends Unix datagrams to the episode collector, so recorder outages never
block robot, Modbus, or camera I/O loops. Sensor data is bounded/drop-oldest;
critical control and maintenance events are retained and close() fails if they
cannot be delivered.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import os
from pathlib import Path
import socket
import threading
import time
from typing import Any, Mapping


SCHEMA_VERSION = 1
DEFAULT_MAX_DATAGRAM = 2 * 1024 * 1024


def default_deviceio_socket() -> Path:
    configured = os.environ.get("ISAAC_TELEOP_DEVICEIO_SOCKET", "").strip()
    if configured:
        return Path(configured)
    runtime = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"))
    return runtime / "isaac_teleop" / "deviceio.sock"


def record_envelope(
    *,
    producer: str,
    topic: str,
    source_time_ns: int,
    host_receive_time_ns: int,
    sequence: int,
    payload: Any,
    valid: bool = True,
    invalid_reason: str = "",
    source_clock_domain: str = "host_monotonic",
    host_clock_domain: str = "host_monotonic",
    mapped_host_time_ns: int | None = None,
    timing_valid: bool | None = None,
) -> dict[str, Any]:
    if not producer or not topic:
        raise ValueError("producer and topic are required")
    if not topic.startswith("/"):
        raise ValueError("DeviceIO topic must be absolute")
    mapped = (
        int(host_receive_time_ns)
        if mapped_host_time_ns is None
        and source_clock_domain == host_clock_domain
        else mapped_host_time_ns
    )
    timing_ok = mapped is not None if timing_valid is None else bool(timing_valid)
    return {
        "schema_version": SCHEMA_VERSION,
        "producer": str(producer),
        "topic": str(topic),
        "source_time_ns": int(source_time_ns),
        "host_receive_time_ns": int(host_receive_time_ns),
        "sequence": int(sequence),
        "valid": bool(valid),
        "invalid_reason": str(invalid_reason),
        "source_clock_domain": str(source_clock_domain),
        "host_clock_domain": str(host_clock_domain),
        "mapped_host_time_ns": None if mapped is None else int(mapped),
        "timing_valid": timing_ok,
        "payload": payload,
    }


@dataclass(frozen=True)
class DeviceIOStats:
    producer: str
    enqueued_sensor: int
    enqueued_critical: int
    sent: int
    dropped_queue: int
    dropped_transport: int
    reconnect_failures: int
    serialization_failures: int
    pending_sensor: int
    pending_critical: int


class AsyncDeviceIOEmitter:
    """Bounded priority-aware Unix datagram sender."""

    def __init__(
        self,
        producer: str,
        socket_path: str | Path | None = None,
        *,
        sensor_capacity: int = 4096,
        critical_capacity: int = 1024,
        reconnect_period_s: float = 0.05,
        max_datagram_bytes: int = DEFAULT_MAX_DATAGRAM,
    ) -> None:
        if not producer:
            raise ValueError("producer is required")
        if sensor_capacity < 1 or critical_capacity < 1:
            raise ValueError("queue capacities must be positive")
        self.producer = producer
        self.socket_path = (
            Path(socket_path) if socket_path else default_deviceio_socket()
        )
        self.sensor_capacity = int(sensor_capacity)
        self.critical_capacity = int(critical_capacity)
        self.reconnect_period_s = float(reconnect_period_s)
        self.max_datagram_bytes = int(max_datagram_bytes)
        self._condition = threading.Condition()
        self._sensor: deque[Mapping[str, Any]] = deque()
        self._critical: deque[Mapping[str, Any]] = deque()
        self._stop = False
        self._fatal: Exception | None = None
        self._counters = {
            "enqueued_sensor": 0,
            "enqueued_critical": 0,
            "sent": 0,
            "dropped_queue": 0,
            "dropped_transport": 0,
            "reconnect_failures": 0,
            "serialization_failures": 0,
        }
        self._stats_sequence = 0
        self._last_stats_send_ns = 0
        self._thread = threading.Thread(
            target=self._run,
            name=f"deviceio-{producer}",
            daemon=True,
        )
        self._thread.start()

    def emit(self, envelope: Mapping[str, Any], *, critical: bool = False) -> None:
        if str(envelope.get("producer", "")) != self.producer:
            raise ValueError("envelope producer does not match emitter")
        with self._condition:
            if self._stop:
                raise RuntimeError("DeviceIO emitter is closed")
            if self._fatal is not None:
                raise RuntimeError(f"DeviceIO emitter failed: {self._fatal}")
            if critical:
                if len(self._critical) >= self.critical_capacity:
                    raise BufferError("critical DeviceIO queue is full")
                self._critical.append(dict(envelope))
                self._counters["enqueued_critical"] += 1
            else:
                if len(self._sensor) >= self.sensor_capacity:
                    self._sensor.popleft()
                    self._counters["dropped_queue"] += 1
                self._sensor.append(dict(envelope))
                self._counters["enqueued_sensor"] += 1
            self._condition.notify()

    def stats(self) -> DeviceIOStats:
        with self._condition:
            return DeviceIOStats(
                producer=self.producer,
                **self._counters,
                pending_sensor=len(self._sensor),
                pending_critical=len(self._critical),
            )

    def close(self, timeout_s: float = 2.0) -> None:
        with self._condition:
            self._stop = True
            self._condition.notify_all()
        self._thread.join(timeout_s)
        if self._thread.is_alive():
            raise TimeoutError("DeviceIO emitter did not stop")
        with self._condition:
            if self._critical:
                raise RuntimeError(
                    f"{len(self._critical)} critical DeviceIO records were not delivered"
                )
            if self._fatal is not None:
                raise RuntimeError(f"DeviceIO emitter failed: {self._fatal}")

    def _next(self) -> tuple[Mapping[str, Any] | None, bool]:
        with self._condition:
            while not self._stop and not self._critical and not self._sensor:
                self._condition.wait(timeout=0.25)
            if self._critical:
                return self._critical.popleft(), True
            if self._stop:
                self._counters["dropped_transport"] += len(self._sensor)
                self._sensor.clear()
                return None, False
            if self._sensor:
                return self._sensor.popleft(), False
            return None, False

    def _requeue_critical(self, envelope: Mapping[str, Any]) -> None:
        with self._condition:
            self._critical.appendleft(envelope)

    def _run(self) -> None:
        sender = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        sender.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, self.max_datagram_bytes * 2)
        sender.setblocking(False)
        try:
            while True:
                envelope, critical = self._next()
                if envelope is None:
                    return
                try:
                    encoded = json.dumps(
                        envelope,
                        separators=(",", ":"),
                        ensure_ascii=False,
                        allow_nan=False,
                    ).encode("utf-8")
                    if len(encoded) > self.max_datagram_bytes:
                        raise ValueError(
                            f"DeviceIO datagram is {len(encoded)} bytes; limit is "
                            f"{self.max_datagram_bytes}"
                        )
                except Exception as exc:
                    with self._condition:
                        self._counters["serialization_failures"] += 1
                        if critical:
                            self._fatal = exc
                            self._critical.appendleft(envelope)
                            return
                    continue
                try:
                    sender.sendto(encoded, str(self.socket_path))
                except OSError:
                    with self._condition:
                        self._counters["reconnect_failures"] += 1
                    if critical:
                        self._requeue_critical(envelope)
                        time.sleep(self.reconnect_period_s)
                    else:
                        with self._condition:
                            self._counters["dropped_transport"] += 1
                    continue
                with self._condition:
                    self._counters["sent"] += 1
                self._maybe_send_stats(sender)
        finally:
            sender.close()

    def _maybe_send_stats(self, sender: socket.socket) -> None:
        now = time.monotonic_ns()
        if now - self._last_stats_send_ns < 1_000_000_000:
            return
        self._last_stats_send_ns = now
        self._stats_sequence += 1
        stats = self.stats()
        envelope = record_envelope(
            producer=self.producer,
            topic=f"/_deviceio/source_stats/{self.producer}",
            source_time_ns=now,
            host_receive_time_ns=now,
            sequence=self._stats_sequence,
            payload=stats.__dict__,
        )
        try:
            sender.sendto(
                json.dumps(
                    envelope, separators=(",", ":"), allow_nan=False
                ).encode(),
                str(self.socket_path),
            )
        except OSError:
            pass
