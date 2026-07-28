"""Create the local mode-0600 hardware-write permit without following links."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from .guard import HardwareWriteGuard


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument(
        "--confirm",
        required=True,
        choices=[HardwareWriteGuard.PERMIT_FILE_VALUE],
    )
    args = parser.parse_args(argv)
    if not sys.stdin.isatty():
        raise SystemExit("permit creation requires a local interactive TTY")
    path = args.path.expanduser().resolve(strict=False)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        os.write(
            descriptor,
            (HardwareWriteGuard.PERMIT_FILE_VALUE + "\n").encode("ascii"),
        )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(path, 0o600)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
