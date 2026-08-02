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
import fcntl

_INPUT_EVENT = struct.Struct("llHHI")
_EV_KEY = 0x01
_EVIOCGRAB = 0x40044590
KEY_LEFT = 105
KEY_RIGHT = 106
KEY_DOWN = 108
KEY_SPACE = 57


def resolve_input_event_path(path: str | Path) -> Path:
    """Resolve either an evdev path or a stable ``name:DEVICE`` selector."""

    requested = str(path).strip()
    prefix = "name:"
    if not requested.lower().startswith(prefix):
        return Path(requested)
    target = requested[len(prefix) :].strip()
    if not target:
        raise FileNotFoundError("input device name is empty")
    for name_file in sorted(Path("/sys/class/input").glob("event*/device/name")):
        try:
            if name_file.read_text(encoding="utf-8").strip() != target:
                continue
        except OSError:
            continue
        return Path("/dev/input") / name_file.parents[1].name
    raise FileNotFoundError(f"no evdev device named {target!r}")


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
        on_status: Callable[[str], None] | None = None,
        *,
        enable_key_code: int = KEY_DOWN,
        grab: bool = False,
    ) -> None:
        self._path = path
        self._on_enable = on_enable
        self._on_event = on_event
        self._on_status = on_status
        self._enable_key_code = int(enable_key_code)
        self._grab = bool(grab)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._enable_pressed = False
        self._last_status = ""

    def _status(self, value: str) -> None:
        if value == self._last_status:
            return
        self._last_status = value
        if self._on_status is not None:
            self._on_status(value)

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
        self._set_enable(False)
        while not self._stop.is_set():
            try:
                path = resolve_input_event_path(self._path)
                fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
                if self._grab:
                    fcntl.ioctl(fd, _EVIOCGRAB, 1)
                self._status(f"connected:{path}")
            except OSError as exc:
                self._status(f"error:{exc}")
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
                if self._grab:
                    try:
                        fcntl.ioctl(fd, _EVIOCGRAB, 0)
                    except OSError:
                        pass
                os.close(fd)
                self._set_enable(False)
