"""Compatibility entry point for rclpy's non-printf RcutilsLogger API."""

from __future__ import annotations

from rclpy.impl.rcutils_logger import RcutilsLogger


def _install_printf_compatibility() -> None:
    for method_name in ("debug", "info", "warning", "warn", "error", "fatal"):
        original = getattr(RcutilsLogger, method_name, None)
        if original is None or getattr(original, "_isaac_flexiv_compat", False):
            continue

        def compatible(self, message, *args, _original=original, **kwargs):
            if args:
                try:
                    message = str(message) % args
                except (TypeError, ValueError):
                    message = " ".join([str(message), *(str(arg) for arg in args)])
            return _original(self, str(message), **kwargs)

        compatible._isaac_flexiv_compat = True
        setattr(RcutilsLogger, method_name, compatible)


_install_printf_compatibility()

from .ros2_node import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())

