from __future__ import annotations

import pytest

pytest.importorskip("flexiv_inspire_interfaces")

from flexiv_inspire_control.node import _require_matching_ft_zero_epoch


def test_ft_zero_epoch_accepts_the_current_daemon_connection() -> None:
    _require_matching_ft_zero_epoch(
        result_generation=8,
        result_instance_id="daemon-a",
        current_generation=8,
        current_instance_id="daemon-a",
    )


@pytest.mark.parametrize(
    ("current_generation", "current_instance_id", "message"),
    [
        (None, "daemon-a", "generation is unavailable"),
        (8, "", "instance is unavailable"),
        (9, "daemon-a", "generation changed"),
        (8, "daemon-b", "instance changed"),
    ],
)
def test_ft_zero_result_is_rejected_if_reconnect_wins_before_ready(
    current_generation,
    current_instance_id,
    message,
) -> None:
    # Model the observation thread updating the current epoch after zero_ft
    # returns but before the bridge attempts mark_ft_zeroed()/declare_ready().
    with pytest.raises(RuntimeError, match=message):
        _require_matching_ft_zero_epoch(
            result_generation=8,
            result_instance_id="daemon-a",
            current_generation=current_generation,
            current_instance_id=current_instance_id,
        )
