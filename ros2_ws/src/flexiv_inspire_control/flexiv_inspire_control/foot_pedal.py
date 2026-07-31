"""Three-key Linux input-event pedal monitor with fail-closed reconnects."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import os
from pathlib import Path
import select
import struct
import threading
import time

_INPUT_EVENT = struct.Struct("llHHI")
_EV_KEY = 0x01
KEY_LEFT = 105
KEY_RIGHT = 106
KEY_DOWN = 108


@dataclass(frozen=True)
class PedalEvent:
    key_code: int
    pressed: bool
    monotonic_ns: int


class FootPedalMonitor:
    """Decode Down as a momentary enable and expose other keys as edge events."""

    def __init__(
        self,
        path: Path,
        on_enable: Callable[[bool], None],
        on_event: Callable[[PedalEvent], None] | None = None,
        *,
        enable_key_code: int = KEY_DOWN,
    ) -> None:
        self._path = path
        self._on_enable = on_enable
        self._on_event = on_event
        self._enable_key_code = int(enable_key_code)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._enable_pressed = False

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name="foot-pedal")
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _set_enable(self, pressed: bool) -> None:
        pressed = bool(pressed)
        if pressed == self._enable_pressed:
            return
        self._enable_pressed = pressed
        self._on_enable(pressed)

    def _emit(self, key_code: int) -> None:
        if self._on_event is not None:
            self._on_event(PedalEvent(key_code, True, time.monotonic_ns()))

    def _run(self) -> None:
        self._on_enable(False)
        while not self._stop.is_set():
            try:
                fd = os.open(self._path, os.O_RDONLY | os.O_NONBLOCK)
            except OSError:
                self._set_enable(False)
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
                        _, _, event_type, key_code, value = _INPUT_EVENT.unpack_from(data, offset)
                        if event_type != _EV_KEY or value not in (0, 1, 2):
                            continue
                        if key_code == self._enable_key_code:
                            if value in (0, 1):
                                self._set_enable(value == 1)
                        elif value == 1:
                            self._emit(key_code)
            finally:
                os.close(fd)
                self._set_enable(False)
