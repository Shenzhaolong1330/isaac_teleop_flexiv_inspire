from __future__ import annotations

import numpy as np
import pytest

from isaac_teleop_core.command import (
    BimanualCommand,
    CommandPoint,
    CommandSource,
    ValidMask,
)
from isaac_teleop_core.control import (
    ControlArbiter,
    ControlState,
    GateInputs,
    HoldReason,
    TransitionError,
)


BASE = 10_000_000_000


def action(
    source: CommandSource,
    sequence: int,
    *,
    now: int = BASE,
    ttl_s: float = 0.05,
    deadman: bool = True,
) -> BimanualCommand:
    return BimanualCommand.from_policy_vectors(
        [CommandPoint.identity().to_policy_vector()],
        session_id="session",
        source=source,
        sequence=sequence,
        ttl_s=ttl_s,
        frame_id="world",
        deadman=deadman,
        valid_mask=ValidMask.LEFT_ARM | ValidMask.RIGHT_ARM,
        issued_monotonic_ns=now,
    )


def ready_arbiter(
    source: CommandSource = CommandSource.TELEOP,
    *,
    policy_period_s: float | None = None,
) -> ControlArbiter:
    arbiter = ControlArbiter(policy_period_s=policy_period_s)
    arbiter.begin_hardware_session("session")
    arbiter.mark_ft_zeroed(session_id="session", connection_generation=1)
    arbiter.declare_ready(connection_generation=1)
    arbiter.update_gates(
        GateInputs(
            local_permission=True,
            physical_pedal=True,
            arms_online=True,
            hands_online=True,
            limits_ok=True,
            collision_clear=True,
        ),
        now_monotonic_ns=BASE,
    )
    arbiter.heartbeat(source, now_monotonic_ns=BASE)
    arbiter.arm(source)
    return arbiter


def test_source_is_exclusive_and_first_valid_command_activates() -> None:
    arbiter = ready_arbiter()
    with pytest.raises(TransitionError):
        arbiter.arm(CommandSource.POLICY)
    approved = arbiter.submit(action(CommandSource.TELEOP, 1), now_monotonic_ns=BASE)
    assert approved.sequence == 1
    assert arbiter.snapshot.state is ControlState.ACTIVE


def test_older_approved_command_ack_remains_valid_after_newer_submit() -> None:
    arbiter = ready_arbiter()
    first = arbiter.submit(
        action(CommandSource.TELEOP, 1), now_monotonic_ns=BASE
    )
    second = arbiter.submit(
        action(CommandSource.TELEOP, 2), now_monotonic_ns=BASE + 1
    )

    arbiter.mark_sent(first)
    assert arbiter.snapshot.last_sent is first
    arbiter.mark_sent(second)
    assert arbiter.snapshot.last_sent is second
    assert arbiter.snapshot.state is ControlState.ACTIVE


def test_positive_ack_arriving_after_pedal_release_preserves_hold() -> None:
    arbiter = ready_arbiter()
    approved = arbiter.submit(
        action(CommandSource.TELEOP, 1), now_monotonic_ns=BASE
    )

    arbiter.update_gates(
        GateInputs(
            local_permission=True,
            physical_pedal=False,
            arms_online=True,
            hands_online=True,
            limits_ok=True,
            collision_clear=True,
        ),
        now_monotonic_ns=BASE + 1,
    )
    assert arbiter.snapshot.state is ControlState.HOLD_LATCHED
    assert arbiter.snapshot.hold_reason is HoldReason.PEDAL_RELEASED

    arbiter.mark_sent(approved)

    assert arbiter.snapshot.state is ControlState.HOLD_LATCHED
    assert arbiter.snapshot.hold_reason is HoldReason.PEDAL_RELEASED
    assert arbiter.snapshot.last_sent is approved


def test_unapproved_future_command_cannot_be_marked_sent() -> None:
    arbiter = ready_arbiter()
    arbiter.submit(action(CommandSource.TELEOP, 1), now_monotonic_ns=BASE)

    with pytest.raises(TransitionError, match="not approved"):
        arbiter.mark_sent(action(CommandSource.TELEOP, 2))


def test_ttl_watchdog_latches_and_never_auto_resumes() -> None:
    arbiter = ready_arbiter()
    arbiter.submit(action(CommandSource.TELEOP, 1), now_monotonic_ns=BASE)
    snapshot = arbiter.tick(now_monotonic_ns=BASE + 50_000_001)
    assert snapshot.state is ControlState.HOLD_LATCHED
    assert snapshot.hold_reason is HoldReason.COMMAND_STALE
    arbiter.heartbeat(CommandSource.TELEOP, now_monotonic_ns=BASE + 50_000_002)
    assert arbiter.tick(now_monotonic_ns=BASE + 50_000_003).state is ControlState.HOLD_LATCHED


def test_invalid_or_gate_failed_first_packet_latches_from_armed() -> None:
    arbiter = ready_arbiter()
    arbiter.update_gates(GateInputs(local_permission=True), now_monotonic_ns=BASE)
    with pytest.raises(TransitionError):
        arbiter.submit(action(CommandSource.TELEOP, 1), now_monotonic_ns=BASE)
    assert arbiter.snapshot.state is ControlState.HOLD_LATCHED

    second = ready_arbiter()
    second.reject_invalid_command(CommandSource.TELEOP, now_monotonic_ns=BASE)
    assert second.snapshot.state is ControlState.HOLD_LATCHED
    assert second.snapshot.hold_reason is HoldReason.INVALID_COMMAND


def test_expired_local_authorization_holds_without_marking_command_invalid() -> None:
    arbiter = ready_arbiter()

    arbiter.require_reauthorization(
        CommandSource.TELEOP, now_monotonic_ns=BASE
    )

    assert arbiter.snapshot.state is ControlState.HOLD_LATCHED
    assert arbiter.snapshot.hold_reason is HoldReason.AUTHORIZATION_EXPIRED


def test_daemon_release_latch_is_recoverable_without_fault() -> None:
    arbiter = ready_arbiter()
    arbiter.submit(action(CommandSource.TELEOP, 1), now_monotonic_ns=BASE)

    arbiter.observe_hardware_hold(
        CommandSource.TELEOP,
        HoldReason.PEDAL_RELEASED,
        now_monotonic_ns=BASE + 1,
    )

    assert arbiter.snapshot.state is ControlState.HOLD_LATCHED
    assert arbiter.snapshot.hold_reason is HoldReason.PEDAL_RELEASED
    arbiter.clear_hold(local_acknowledged=True)
    assert arbiter.snapshot.state is ControlState.READY


def test_deadman_release_can_be_locally_cleared_but_requires_rearm() -> None:
    arbiter = ready_arbiter()
    with pytest.raises(TransitionError):
        arbiter.submit(
            action(CommandSource.TELEOP, 1, deadman=False),
            now_monotonic_ns=BASE,
        )
    assert arbiter.snapshot.state is ControlState.HOLD_LATCHED
    arbiter.clear_hold(local_acknowledged=True)
    assert arbiter.snapshot.state is ControlState.READY
    assert arbiter.snapshot.active_source is None


def test_rdk_reconnect_invalidates_zero_and_ready() -> None:
    arbiter = ready_arbiter()
    arbiter.on_rdk_reconnect()
    assert arbiter.snapshot.state is ControlState.MAINTENANCE
    assert arbiter.snapshot.ft_zero_generation is None
    with pytest.raises(TransitionError):
        arbiter.arm(CommandSource.TELEOP)


def test_policy_heartbeat_timeout_is_max_200ms_or_three_periods() -> None:
    arbiter = ready_arbiter(CommandSource.POLICY, policy_period_s=0.1)
    arbiter.submit(
        action(CommandSource.POLICY, 1, ttl_s=0.5),
        now_monotonic_ns=BASE,
    )
    assert arbiter.tick(now_monotonic_ns=BASE + 299_000_000).state is ControlState.ACTIVE
    snapshot = arbiter.tick(now_monotonic_ns=BASE + 301_000_000)
    assert snapshot.state is ControlState.HOLD_LATCHED
    assert snapshot.hold_reason is HoldReason.HEARTBEAT_STALE


def test_prepare_home_drops_active_source_and_returns_ready() -> None:
    arbiter = ready_arbiter()
    arbiter.submit(action(CommandSource.TELEOP, 1), now_monotonic_ns=BASE)

    arbiter.prepare_home()

    assert arbiter.snapshot.state is ControlState.READY
    assert arbiter.snapshot.active_source is None
    assert arbiter.snapshot.last_safe is None


def test_prepare_home_rejects_maintenance_without_current_ft_zero() -> None:
    arbiter = ControlArbiter()
    arbiter.begin_hardware_session("session")

    with pytest.raises(TransitionError, match="current F/T zero"):
        arbiter.prepare_home()


def test_local_reset_can_recover_fault_for_home_with_current_zero() -> None:
    arbiter = ready_arbiter()
    arbiter.fault()

    arbiter.recover_fault_for_home()

    assert arbiter.snapshot.state is ControlState.READY
    assert arbiter.snapshot.active_source is None
    assert arbiter.snapshot.ft_zero_generation == 1


def test_fault_recovery_home_rejects_missing_ft_zero() -> None:
    arbiter = ControlArbiter()
    arbiter.begin_hardware_session("session")
    arbiter.fault()

    with pytest.raises(TransitionError, match="current F/T zero"):
        arbiter.recover_fault_for_home()
