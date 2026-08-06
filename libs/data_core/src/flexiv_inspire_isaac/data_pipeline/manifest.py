"""Episode manifest with calibration, zeroing and stream accounting."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import yaml


def local_minute_timestamp(now: datetime | None = None) -> str:
    """Return a filesystem-safe local timestamp containing date/hour/minute."""

    # A supplied value is already the caller's chosen local clock, which keeps
    # tests and imported historical timestamps independent of the process TZ.
    local = datetime.now().astimezone() if now is None else now
    return local.strftime("%Y%m%d_%H%M")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_yaml_sha256(path: str | Path) -> str:
    """Hash parsed YAML exactly like the RDK daemon tool/payload identity."""
    source = Path(path).expanduser().resolve(strict=True)
    parsed = yaml.safe_load(source.read_bytes())
    if not isinstance(parsed, dict):
        raise ValueError("tool/payload YAML root must be a mapping")
    canonical = json.dumps(
        parsed,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class StreamStats:
    expected_hz: float
    observed_hz: float = 0.0
    samples: int = 0
    drops: int = 0
    invalid: int = 0
    first_source_time_ns: int | None = None
    last_source_time_ns: int | None = None


@dataclass
class EpisodeManifest:
    schema_version: str
    episode_uuid: str
    session_id: str
    deviceio_mcap: str
    ros_mcap: str
    software_versions: Mapping[str, str]
    calibration_hashes: Mapping[str, str]
    tool_configuration_hash: str
    ft_zero_event: Mapping[str, Any]
    tool_configuration_file_hash: str = ""
    deviceio_capture_layer: str = "post-dds-typed-mirror"
    dataset_name: str = ""
    episode_index: int = 0
    attempt: int = 1
    collection_timestamp_local: str = ""
    task_name: str = ""
    task_description: str = ""
    pause_count: int = 0
    paused_duration_ns: int = 0
    suppressed_samples: int = 0
    native_source_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    streams: dict[str, StreamStats] = field(default_factory=dict)
    completed: bool = False
    completion_reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def write_atomic(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(
            self.to_dict(), indent=2, ensure_ascii=False, allow_nan=False
        ).encode()
        fd, temporary = tempfile.mkstemp(
            prefix=f".{destination.name}.", dir=destination.parent
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
