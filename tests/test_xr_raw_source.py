from __future__ import annotations

import pytest

from flexiv_inspire_isaac.openxr_errors import is_retryable_openxr_session_error


@pytest.mark.parametrize(
    "message",
    (
        "Failed to get OpenXR system: -35",
        "Failed to create OpenXR instance: -51",
    ),
)
def test_transient_openxr_startup_errors_are_retryable(message: str) -> None:
    assert is_retryable_openxr_session_error(RuntimeError(message))


@pytest.mark.parametrize(
    "message",
    (
        "Failed to create OpenXR instance: -9",
        "CloudXR runtime process exited unexpectedly",
        "invalid hand transport",
    ),
)
def test_non_transient_openxr_errors_remain_fatal(message: str) -> None:
    assert not is_retryable_openxr_session_error(RuntimeError(message))
