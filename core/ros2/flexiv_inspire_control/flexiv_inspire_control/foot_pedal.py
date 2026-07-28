"""Minimal Linux input-event foot-pedal monitor with fail-closed disconnect."""

from __future__ import annotations

from collections.abc import Callable
import os
from pathlib import Path
import select
import struct
import threading

_INPUT_EVENT = struct.Struct("llHHI")
_EV_KEY = 0x01


class FootPedalMonitor:
    def __init__(self, path: Path, on_state: Callable[[bool], None]) -> None:
        self._path = path
        self._on_state = on_state
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name="foot-pedal")
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        self._on_state(False)
        while not self._stop.is_set():
            try:
                fd = os.open(self._path, os.O_RDONLY | os.O_NONBLOCK)
            except OSError:
                self._stop.wait(0.25)
                continue
            try:
                while not self._stop.is_set():
                    readable, _, _ = select.select([fd], [], [], 0.1)
                    if not readable:
                        continue
                    data = os.read(fd, _INPUT_EVENT.size * 16)
                    if not data:
                        break
                    for offset in range(0, len(data) - _INPUT_EVENT.size + 1, _INPUT_EVENT.size):
                        _, _, event_type, _, value = _INPUT_EVENT.unpack_from(data, offset)
                        if event_type == _EV_KEY and value in (0, 1):
                            self._on_state(bool(value))
            finally:
                os.close(fd)
                self._on_state(False)
