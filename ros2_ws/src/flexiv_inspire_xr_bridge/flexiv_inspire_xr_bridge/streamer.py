"""Latest-only JPEG to H.264/RTP encoder workers.

The encoder is deliberately outside the camera acquisition callback. A slow XR
client can only drop display frames; it cannot stall or alter the recorded camera
observation stream.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
import subprocess
import threading
from typing import BinaryIO


@dataclass(frozen=True)
class EncoderSettings:
    ffmpeg: str
    host: str
    port: int
    fps: float
    bitrate_mbps: float
    gop: int
    packet_size: int = 1200
    payload_type: int = 96
    encoder: str = "auto"

    def __post_init__(self) -> None:
        if not 1024 <= self.port <= 65535:
            raise ValueError("RTP port must be in [1024,65535]")
        if not 0 < self.fps <= 90:
            raise ValueError("fps must be in (0,90]")
        if not 0 < self.bitrate_mbps <= 100:
            raise ValueError("bitrate_mbps must be in (0,100]")
        if self.gop <= 0:
            raise ValueError("gop must be positive")
        if self.encoder not in {"auto", "h264_nvenc", "libx264"}:
            raise ValueError("encoder must be auto, h264_nvenc, or libx264")


def build_ffmpeg_command(settings: EncoderSettings, encoder: str) -> list[str]:
    if encoder not in {"h264_nvenc", "libx264"}:
        raise ValueError(f"unsupported encoder: {encoder}")
    bitrate = f"{settings.bitrate_mbps:g}M"
    # Keep at most two frames in the encoder VBV instead of the old one-second
    # buffer.  The latest-only mailbox already handles overload by dropping.
    vbv_kbits = max(64, round(settings.bitrate_mbps * 1000 * 2 / settings.fps))
    vbv_buffer = f"{vbv_kbits}k"
    common = [
        settings.ffmpeg, "-hide_banner", "-loglevel", "warning", "-nostdin",
        "-fflags", "nobuffer+flush_packets", "-f", "image2pipe", "-vcodec", "mjpeg",
        "-r", f"{settings.fps:g}", "-i", "pipe:0", "-an",
    ]
    if encoder == "h264_nvenc":
        codec = [
            "-c:v", "h264_nvenc", "-preset", "p1", "-tune", "ull",
            "-rc", "cbr", "-b:v", bitrate, "-maxrate", bitrate,
            "-bufsize", vbv_buffer, "-g", str(settings.gop), "-bf", "0",
            "-rc-lookahead", "0", "-delay", "0", "-surfaces", "2",
            "-zerolatency", "1",
            "-pix_fmt", "yuv420p",
        ]
    else:
        codec = [
            "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
            "-b:v", bitrate, "-maxrate", bitrate, "-bufsize", vbv_buffer,
            "-g", str(settings.gop), "-keyint_min", str(settings.gop),
            "-sc_threshold", "0", "-bf", "0", "-pix_fmt", "yuv420p",
        ]
    destination = f"rtp://{settings.host}:{settings.port}?pkt_size={settings.packet_size}"
    return common + codec + [
        "-fps_mode", "passthrough", "-flush_packets", "1",
        "-muxdelay", "0", "-muxpreload", "0", "-f", "rtp",
        "-payload_type", str(settings.payload_type), destination,
    ]


class LatestFrameEncoder:
    """One encoder subprocess and a one-slot mailbox for a single camera."""

    def __init__(
        self,
        settings: EncoderSettings,
        *,
        process_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
    ) -> None:
        self.settings = settings
        self._process_factory = process_factory
        self._condition = threading.Condition()
        self._pending: bytes | None = None
        self._stop = False
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen | None = None
        self._encoder_candidates = (
            ["h264_nvenc", "libx264"] if settings.encoder == "auto" else [settings.encoder]
        )
        self._encoder_index = 0
        self.submitted = 0
        self.sent = 0
        self.dropped = 0
        self.restarts = 0

    @property
    def active_encoder(self) -> str:
        return self._encoder_candidates[self._encoder_index]

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._stop = False
        self._thread = threading.Thread(target=self._run, name=f"xr-rtp-{self.settings.port}", daemon=True)
        self._thread.start()

    def submit(self, jpeg: bytes) -> None:
        if not jpeg:
            return
        with self._condition:
            if self._stop:
                self.dropped += 1
                return
            self.submitted += 1
            if self._pending is not None:
                self.dropped += 1
            self._pending = bytes(jpeg)
            self._condition.notify()

    def stop(self, timeout: float = 3.0) -> None:
        with self._condition:
            self._stop = True
            self._condition.notify_all()
        # Closing ffmpeg first also unblocks a worker stuck in a pipe write.
        self._close_process()
        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                raise TimeoutError(
                    f"XR encoder worker on port {self.settings.port} did not stop"
                )
            self._thread = None

    def snapshot(self) -> dict[str, object]:
        return {
            "port": self.settings.port,
            "encoder": self.active_encoder,
            "running": self.running,
            "submitted": self.submitted,
            "sent": self.sent,
            "dropped": self.dropped,
            "restarts": self.restarts,
        }

    def _spawn(self) -> subprocess.Popen:
        command = build_ffmpeg_command(self.settings, self.active_encoder)
        return self._process_factory(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            bufsize=0,
            start_new_session=True,
        )

    def _switch_or_restart(self) -> None:
        self._close_process()
        if self._encoder_index + 1 < len(self._encoder_candidates):
            self._encoder_index += 1
        self.restarts += 1

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._stop:
                    self._condition.wait(timeout=0.5)
                if self._stop:
                    break
                payload = self._pending
                self._pending = None
            assert payload is not None
            try:
                process = self._process
                if process is None or process.poll() is not None:
                    if process is not None:
                        self._switch_or_restart()
                    self._process = self._spawn()
                    process = self._process
                stream: BinaryIO | None = process.stdin
                if stream is None:
                    raise BrokenPipeError("ffmpeg stdin unavailable")
                stream.write(payload)
                stream.flush()
                self.sent += 1
            except (BrokenPipeError, OSError):
                self._switch_or_restart()
        self._close_process()

    def _close_process(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)
