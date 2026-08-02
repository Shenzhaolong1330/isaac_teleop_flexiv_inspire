"""Classification of OpenXR startup errors handled by the XR source."""

from __future__ import annotations


_RETRYABLE_OPENXR_SESSION_ERRORS = (
    "Failed to get OpenXR system",
    "Failed to create OpenXR instance: -51",
)


def is_retryable_openxr_session_error(exc: RuntimeError) -> bool:
    """Return whether a live CloudXR launcher may recover by retrying."""

    message = str(exc)
    return any(marker in message for marker in _RETRYABLE_OPENXR_SESSION_ERRORS)
