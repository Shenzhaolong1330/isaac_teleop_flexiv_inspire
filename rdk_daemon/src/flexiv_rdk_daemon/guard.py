"""Local, explicit guard for every state-changing RDK call."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import stat


class HardwareWriteRejected(PermissionError):
    pass


@dataclass(frozen=True)
class HardwareWriteGuard:
    cli_enabled: bool = False
    environment: dict[str, str] | None = None
    permit_file: Path | None = None
    test_backend: bool = False

    ENVIRONMENT_KEY = "ISAAC_TELEOP_ALLOW_HARDWARE_WRITES"
    ENVIRONMENT_VALUE = "FLEXIV-RDK-WRITES-ENABLED"
    PERMIT_FILE_VALUE = "FLEXIV-RDK-WRITES-ENABLED"

    def require(
        self,
        operation: str,
        *,
        local_console: bool,
        confirmation: str | None = None,
    ) -> None:
        if self.test_backend:
            return
        environment = os.environ if self.environment is None else self.environment
        if not self.cli_enabled:
            raise HardwareWriteRejected(
                f"{operation}: daemon was not started with the local write flag"
            )
        if environment.get(self.ENVIRONMENT_KEY) != self.ENVIRONMENT_VALUE:
            raise HardwareWriteRejected(
                f"{operation}: local hardware-write environment guard is absent"
            )
        if not local_console:
            raise HardwareWriteRejected(
                f"{operation}: remote clients cannot authorize hardware writes"
            )
        if self.permit_file is None:
            raise HardwareWriteRejected(
                f"{operation}: local permit file is not configured"
            )
        try:
            file_stat = self.permit_file.lstat()
        except FileNotFoundError as exc:
            raise HardwareWriteRejected(
                f"{operation}: local permit file does not exist"
            ) from exc
        if not stat.S_ISREG(file_stat.st_mode):
            raise HardwareWriteRejected(
                f"{operation}: local permit must be a regular file"
            )
        if (
            file_stat.st_uid != os.getuid()
            or stat.S_IMODE(file_stat.st_mode) != 0o600
        ):
            raise HardwareWriteRejected(
                f"{operation}: permit must be user-owned and exactly mode 0600"
            )
        file_value = self.permit_file.read_text(encoding="utf-8").strip()
        if file_value != self.PERMIT_FILE_VALUE:
            raise HardwareWriteRejected(
                f"{operation}: permit content must be {self.PERMIT_FILE_VALUE!r}"
            )
        if confirmation is not None and file_value != confirmation:
            raise HardwareWriteRejected(
                f"{operation}: permit content does not match confirmation"
            )
