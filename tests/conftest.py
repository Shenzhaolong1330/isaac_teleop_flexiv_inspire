from __future__ import annotations

from flexiv_inspire_isaac.config import BridgeConfig


def make_config(*, warmup_samples: int = 1) -> BridgeConfig:
    return BridgeConfig.from_dict(
        {
            "command_enabled": False,
            "mapping": {
                "left": {
                    "pose_index": 0,
                    "translation_gain": 1.0,
                    "rotation_gain": 1.0,
                    "axis_rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                },
                "right": {
                    "pose_index": 1,
                    "translation_gain": 1.0,
                    "rotation_gain": 1.0,
                    "axis_rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                },
            },
            "safety": {
                "warmup_samples": warmup_samples,
                "max_input_age_s": 0.12,
                "max_tf_age_s": 0.12,
                "max_deadman_age_s": 0.15,
                "max_control_dt_s": 0.05,
                "ack_timeout_s": 0.08,
                "tracking_jump_translation_m": 0.12,
                "tracking_jump_rotation_rad": 0.60,
                "max_anchor_translation_m": 0.35,
                "max_anchor_rotation_rad": 2.20,
                "max_tracking_lag_translation_m": 0.12,
                "max_tracking_lag_rotation_rad": 0.70,
                "max_linear_speed_m_s": 0.18,
                "max_angular_speed_rad_s": 0.60,
                "max_linear_accel_m_s2": 0.80,
                "max_angular_accel_rad_s2": 2.50,
                "max_translation_step_m": 0.015,
                "max_rotation_step_rad": 0.030,
                "gateway_watchdog_s": 0.10,
            },
            "deadman": {
                "source": "quest_squeeze_both",
                "gateway_pedal_required": False,
            },
            "ros": {},
            "ipc": {},
            "existing_stack": {},
        }
    )

