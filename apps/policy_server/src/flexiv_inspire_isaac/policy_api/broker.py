"""Drop-old/keep-latest buffers for remote policy traffic."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import threading
from typing import Generic, TypeVar


T = TypeVar("T")


class LatestActionBuffer(Generic[T]):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._item: T | None = None
        self.replaced_count = 0

    def put(self, item: T) -> None:
        with self._lock:
            if self._item is not None:
                self.replaced_count += 1
            self._item = item

    def take(self) -> T | None:
        with self._lock:
            item, self._item = self._item, None
            return item


@dataclass(frozen=True)
class ActivePolicyStream:
    stream_id: str
    lease_id: str
    session_id: str
    sequence: int
    action_deadline_ns: int


class PolicyStreamLiveness:
    """Thread-safe accepted action-stream state for ROS heartbeat publication."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: ActivePolicyStream | None = None

    def arm(
        self,
        *,
        stream_id: str,
        lease_id: str,
        session_id: str,
        sequence: int,
        action_deadline_ns: int,
    ) -> None:
        with self._lock:
            self._active = ActivePolicyStream(
                stream_id, lease_id, session_id, sequence, action_deadline_ns
            )

    def current(self) -> ActivePolicyStream | None:
        with self._lock:
            return self._active

    def disarm(self, stream_id: str) -> bool:
        with self._lock:
            if self._active is None or self._active.stream_id != stream_id:
                return False
            self._active = None
            return True

    def clear(self) -> None:
        with self._lock:
            self._active = None


class ObservationBroker(Generic[T]):
    """Async sequence broker; slow consumers skip intermediate observations."""

    def __init__(self) -> None:
        self._condition = asyncio.Condition()
        self._sequence = 0
        self._latest: T | None = None

    async def publish(self, observation: T) -> int:
        async with self._condition:
            self._sequence += 1
            self._latest = observation
            self._condition.notify_all()
            return self._sequence

    async def latest(self) -> tuple[int, T | None]:
        async with self._condition:
            return self._sequence, self._latest

    async def next_after(self, sequence: int) -> tuple[int, T]:
        async with self._condition:
            await self._condition.wait_for(
                lambda: self._sequence > sequence and self._latest is not None
            )
            assert self._latest is not None
            return self._sequence, self._latest
