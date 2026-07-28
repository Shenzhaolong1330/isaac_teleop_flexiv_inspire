from __future__ import annotations

import numpy as np

from flexiv_inspire_isaac.mapping import (
    AXES,
    BridgeState,
    DualArmAbsoluteMapper,
    Pose,
    PosePairSample,
    action_vectors,
)

from conftest import make_config


def _sample(
    sequence: int,
    now_ns: int,
    *,
    left_x: float = 0.0,
    right_x: float = 0.0,
) -> PosePairSample:
    identity = [0.0, 0.0, 0.0, 1.0]
    return PosePairSample(
        left=Pose.from_values([left_x, 0.0, 0.0], identity),
        right=Pose.from_values([right_x, 0.0, 0.0], identity),
        sequence=sequence,
        receive_monotonic_ns=now_ns,
        source_stamp_ns=now_ns,
        frame_id="world",
        valid=True,
    )


def _arm(mapper: DualArmAbsoluteMapper, base_ns: int = 1_000_000_000) -> int:
    mapper.ingest_sample(_sample(0, base_ns))
    mapper.set_deadman(True, base_ns + 1_000_000)
    mapper.ingest_sample(_sample(1, base_ns + 2_000_000))
    decision = mapper.tick(base_ns + 3_000_000)
    assert decision is not None and decision.kind == "arm"
    assert decision.applied_action is None
    mapper.acknowledge_arm(
        epoch=decision.epoch, success=True, now_ns=base_ns + 4_000_000
    )
    return base_ns + 4_000_000


def test_engage_and_reengage_first_sample_is_zero() -> None:
    cfg = make_config()
    mapper = DualArmAbsoluteMapper(cfg.mapping, cfg.safety)
    mapper.ingest_sample(_sample(0, 1_000_000_000, left_x=0.8, right_x=-0.6))
    mapper.set_deadman(True, 1_001_000_000)
    mapper.ingest_sample(_sample(1, 1_002_000_000, left_x=0.8, right_x=-0.6))
    first = mapper.tick(1_003_000_000)
    assert first is not None and first.kind == "arm"
    assert first.applied_action is None
    mapper.acknowledge_arm(epoch=first.epoch, success=True, now_ns=1_004_000_000)

    mapper.set_deadman(False, 1_005_000_000)
    hold = mapper.tick(1_006_000_000)
    assert hold is not None and hold.kind == "hold"
    mapper.ingest_sample(_sample(2, 1_007_000_000, left_x=0.2, right_x=-0.2))
    mapper.set_deadman(True, 1_008_000_000)
    mapper.ingest_sample(_sample(3, 1_009_000_000, left_x=0.2, right_x=-0.2))
    second = mapper.tick(1_010_000_000)
    assert second is not None and second.kind == "arm"
    assert second.applied_action is None


def test_absolute_target_converges_then_does_not_drift() -> None:
    cfg = make_config()
    mapper = DualArmAbsoluteMapper(cfg.mapping, cfg.safety)
    now_ns = _arm(mapper)
    total_left = np.zeros(3)
    total_right = np.zeros(3)
    sequence = 2
    reached = False
    for _ in range(180):
        now_ns += 10_000_000
        mapper.ingest_sample(
            _sample(sequence, now_ns, left_x=0.05, right_x=-0.03)
        )
        sequence += 1
        decision = mapper.tick(now_ns)
        if decision is None:
            continue
        if decision.kind == "idle":
            reached = True
            break
        assert decision.kind == "action"
        left_dp, _, right_dp, _ = action_vectors(decision.applied_action)
        total_left += left_dp
        total_right += right_dp
        mapper.acknowledge_action(
            epoch=decision.epoch,
            proposal_id=decision.proposal_id,
            success=True,
            now_ns=now_ns + 100_000,
        )
    assert reached
    np.testing.assert_allclose(total_left, [0.05, 0.0, 0.0], atol=1e-8)
    np.testing.assert_allclose(total_right, [-0.03, 0.0, 0.0], atol=1e-8)

    for _ in range(20):
        now_ns += 10_000_000
        mapper.ingest_sample(
            _sample(sequence, now_ns, left_x=0.05, right_x=-0.03)
        )
        sequence += 1
        decision = mapper.tick(now_ns)
        assert decision is None or decision.kind == "idle"


def test_schema_limits_and_manus_hand_ownership() -> None:
    cfg = make_config()
    mapper = DualArmAbsoluteMapper(cfg.mapping, cfg.safety)
    now_ns = _arm(mapper)
    mapper.ingest_sample(
        _sample(2, now_ns + 10_000_000, left_x=0.04, right_x=-0.04)
    )
    decision = mapper.tick(now_ns + 10_000_000)
    assert decision is not None and decision.kind == "action"
    expected = {
        f"{side}_delta_ee_pose.{axis}"
        for side in ("left", "right")
        for axis in AXES
    }
    assert set(decision.applied_action) == expected | {"teleop_enable_pressed"}
    assert not any("hand_cmd" in key for key in decision.applied_action)
    left_dp, left_dr, right_dp, right_dr = action_vectors(decision.applied_action)
    assert np.linalg.norm(left_dp) <= cfg.safety.max_translation_step_m + 1e-12
    assert np.linalg.norm(right_dp) <= cfg.safety.max_translation_step_m + 1e-12
    assert np.linalg.norm(left_dr) <= cfg.safety.max_rotation_step_rad + 1e-12
    assert np.linalg.norm(right_dr) <= cfg.safety.max_rotation_step_rad + 1e-12


def test_jump_or_failed_ack_latches_coupled_hold() -> None:
    cfg = make_config()
    mapper = DualArmAbsoluteMapper(cfg.mapping, cfg.safety)
    now_ns = _arm(mapper)
    mapper.ingest_sample(_sample(2, now_ns + 10_000_000, left_x=0.13))
    assert mapper.state == BridgeState.HOLD_LATCHED
    hold = mapper.tick(now_ns + 11_000_000)
    assert hold is not None and hold.kind == "hold"

    mapper = DualArmAbsoluteMapper(cfg.mapping, cfg.safety)
    now_ns = _arm(mapper, 2_000_000_000)
    mapper.ingest_sample(_sample(2, now_ns + 10_000_000, left_x=0.04))
    action = mapper.tick(now_ns + 10_000_000)
    assert action is not None and action.kind == "action"
    mapper.acknowledge_action(
        epoch=action.epoch,
        proposal_id=action.proposal_id,
        success=False,
        now_ns=now_ns + 11_000_000,
        reason="fake send failure",
    )
    assert mapper.state == BridgeState.HOLD_LATCHED


def _run_rate(rate_hz: int) -> np.ndarray:
    cfg = make_config()
    mapper = DualArmAbsoluteMapper(cfg.mapping, cfg.safety)
    now_ns = _arm(mapper)
    total = np.zeros(3)
    sequence = 2
    for index in range(int(2.0 * rate_hz)):
        now_ns += int(1e9 / rate_hz)
        target = 0.03 * min(1.0, ((index + 1) / rate_hz) / 0.5)
        mapper.ingest_sample(_sample(sequence, now_ns, left_x=target))
        sequence += 1
        decision = mapper.tick(now_ns)
        if decision is not None and decision.kind == "action":
            left_dp, _, _, _ = action_vectors(decision.applied_action)
            total += left_dp
            mapper.acknowledge_action(
                epoch=decision.epoch,
                proposal_id=decision.proposal_id,
                success=True,
                now_ns=now_ns + 10_000,
            )
        elif decision is not None:
            assert decision.kind == "idle"
    return total


def test_final_target_is_callback_rate_independent() -> None:
    low = _run_rate(30)
    high = _run_rate(120)
    np.testing.assert_allclose(low, [0.03, 0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(high, [0.03, 0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(low, high, atol=1e-7)

