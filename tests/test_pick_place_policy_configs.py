from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "policies"


def _yaml(name: str) -> dict:
    return yaml.safe_load((CONFIG / name).read_text())


def test_pick_place_train_and_inference_architectures_match() -> None:
    trained = _yaml("pick_place_act_train_policy.yaml")
    inferred = _yaml("pick_place_act_inference_policy.yaml")
    architecture = (
        "n_obs_steps",
        "chunk_size",
        "normalization_mapping",
        "vision_backbone",
        "pretrained_backbone_weights",
        "replace_final_stride_with_dilation",
        "pre_norm",
        "dim_model",
        "n_heads",
        "dim_feedforward",
        "feedforward_activation",
        "n_encoder_layers",
        "n_decoder_layers",
        "use_vae",
        "latent_dim",
        "n_vae_encoder_layers",
    )
    assert {key: trained[key] for key in architecture} == {
        key: inferred[key] for key in architecture
    }
    assert trained["pretrained_path"] is None
    assert inferred["pretrained_path"] is None
    assert inferred["n_action_steps"] == 1
    assert inferred["temporal_ensemble_coeff"] > 0


def test_pick_place_job_and_shadow_use_one_dataset_contract() -> None:
    train = _yaml("pick_place_act_train.yaml")["train"]
    shadow = _yaml("pick_place_act_shadow.yaml")["record"]
    rpc = _yaml("pick_place_rpc_robot.yaml")
    dataset_root = Path(train["dataset"]["root"])
    checkpoint = Path(shadow["policy"]["pretrained_path"])

    assert train["dataset"]["repo_id"] == "local/pick-place-demo-dual-arm-v1"
    assert train["batch_size"] == 8
    assert train["steps"] == 10_000
    assert Path(train["output_dir"]) in checkpoint.parents
    assert checkpoint.parent.name == "last"
    assert checkpoint.name == "pretrained_model"
    assert shadow["fps"] == 15
    assert shadow["run_mode"] == "run_policy"
    assert shadow["robot_type"] == "isaac_flexiv_rpc"
    assert rpc["robot"]["shadow_only"] is True
    assert Path(rpc["robot"]["feature_contract_info"]) == dataset_root / "meta" / "info.json"
