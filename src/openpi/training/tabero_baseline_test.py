"""CPU-only regression checks for separately trained RGB+state baseline."""

import dataclasses
import os
from pathlib import Path
import subprocess
import sys

import flax.nnx as nnx
from flax.traverse_util import flatten_dict
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi import transforms
from openpi.models import model
from openpi.policies import tabero_offline as offline
from openpi.shared.tactile_type import TactileType
from openpi.training import config
from openpi.training import weight_loaders
from scripts.eval_tabero_offline import validate_model_contract

BASELINE = "pi0_lora_tabero_rgb_state"
TOUCH = "pi0_lora_tabero_rgb_state_touch"


def sample():
    state = np.array([0.3, 0.0, 0.4, 3.13, 0.01, 0.0, 0.02], np.float32)
    return {
        "image": np.zeros((8, 8, 3), np.uint8),
        "wrist_image": np.zeros((8, 8, 3), np.uint8),
        "state": state,
        "actions": np.tile(state, (50, 1)),
        "episode_index": 4,
        "frame_index": 0,
        "actions_is_pad": np.arange(50) >= 1,
        "prompt": "move recorded object",
    }


def test_baseline_and_matched_profile_contract():
    base, touch = config.get_config(BASELINE), config.get_config(TOUCH)
    assert not validate_model_contract(base)
    assert validate_model_contract(touch)
    assert base.model.tactile_type is TactileType.NO
    assert base.model.tactile_streams == ()
    assert base.model.tactile_dim_in == base.model.tactile_prefix_dim_in == base.model.tactile_prefix_lora_rank == 0
    assert base.model.action_dim == touch.model.action_dim == 32
    assert base.model.supervised_action_dim == touch.model.supervised_action_dim == 7
    assert base.model.action_horizon == touch.model.action_horizon == 50
    assert base.weight_loader.params_path == touch.weight_loader.params_path
    assert base.weight_loader.params_path.endswith("49999/params")
    assert base.weight_loader.strict_allow_missing_regex is None
    assert base.weight_loader.strict
    for field in (
        "lr_schedule",
        "optimizer",
        "batch_size",
        "num_train_steps",
        "seed",
        "eval_num_batches",
        "freeze_filter",
    ):
        assert getattr(base, field) == getattr(touch, field)
    assert base.batch_size == 4
    assert base.num_train_steps == 3000
    for field in ("episodes", "validation_episodes", "root", "action_sequence_keys", "video_backend"):
        assert getattr(base.data.base_config, field) == getattr(touch.data.base_config, field)
    assert len(base.data.base_config.episodes) == 26
    assert base.data.base_config.validation_episodes == (4, 14, 24)
    assert not any("tactile" in key or "wrench" in key for key in base.data.base_config.columns)
    assert base.trainable_filter(("PaliGemma", "llm", "lora_a"), nnx.Param(jnp.ones(1)))
    assert not base.trainable_filter(("action_out_proj", "kernel"), nnx.Param(jnp.ones(1)))


def test_abstract_model_has_no_tactile_weights_and_shared_shapes_match():
    def shapes(name):
        return flatten_dict(
            jax.eval_shape(
                lambda: nnx.state(config.get_config(name).model.create(jax.random.key(0)), nnx.Param).to_pure_dict()
            ),
            sep="/",
        )

    base, touch = shapes(BASELINE), shapes(TOUCH)
    assert not any("tactile" in key for key in base)
    assert set(touch) - set(base)
    assert all(key.startswith("tactile_prefix_encoder/") for key in set(touch) - set(base))
    assert all(base[key].shape == touch[key].shape for key in base)


def test_no_tactile_data_transform_matches_shared_inputs_and_inverse(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ModelTransformFactory", lambda: lambda _: transforms.Group())
    stats = {
        key: transforms.NormStats(mean=np.zeros(dim), std=np.ones(dim))
        for key, dim in (("state", 7), ("actions", 7), ("tactile_prefix", 396))
    }
    monkeypatch.setattr(config.DataConfigFactory, "_load_norm_stats", lambda *_: stats.copy())
    configs = [config.get_config(name) for name in (BASELINE, TOUCH)]
    data = [cfg.data.create(tmp_path, cfg.model) for cfg in configs]
    assert set(data[0].norm_stats) == {"state", "actions"}
    assert set(data[1].norm_stats) == {"state", "actions", "tactile_prefix"}
    raw = sample()
    target = raw["actions"].copy()
    base = transforms.compose(data[0].data_transforms.inputs)(raw)
    raw_touch = sample()
    raw_touch["tactile_marker_motion"] = np.ones((9, 198, 2), np.float32)
    touch = transforms.compose(data[1].data_transforms.inputs)(raw_touch)
    assert "tactile_prefix" not in base
    assert "tactile_suffix" not in base
    for key in ("state", "actions"):
        np.testing.assert_array_equal(base[key], touch[key])
    for key in base["image"]:
        np.testing.assert_array_equal(base["image"][key], touch["image"][key])
        assert base["image_mask"][key] == touch["image_mask"][key]
    # Match the policy/data loader's array conversion and batch insertion.
    batched = jax.tree.map(lambda x: np.asarray(x)[None], {k: base[k] for k in ("image", "image_mask", "state")})
    obs = model.Observation.from_dict(batched)
    assert obs.tactile_prefix is None
    assert obs.tactile_suffix is None
    restored = transforms.compose(data[0].data_transforms.outputs)(base)
    np.testing.assert_allclose(restored["actions"], target, atol=1e-6)


@pytest.mark.parametrize("fault", [None, "shared_missing", "shared_shape", "unrelated_extra", "tactile_adapter_extra"])
def test_no_tactile_restore_only_discards_known_removed_leaves(monkeypatch, fault):
    loaded = {
        "shared": np.ones(2),
        "lora_a": np.full(2, 7.0),
        "tactile_prefix_encoder": {"out_proj": {"kernel": np.ones((2, 2)), "bias": np.ones(2)}},
    }
    expected = {"shared": np.zeros(2, np.float32), "lora_a": np.zeros(2, np.float32)}
    if fault == "shared_missing":
        del loaded["lora_a"]
    elif fault == "shared_shape":
        loaded["shared"] = np.ones(3)
    elif fault == "unrelated_extra":
        loaded["unrelated"] = np.ones(2)
    elif fault == "tactile_adapter_extra":
        loaded["tactile_prefix_encoder"]["out_proj"]["lora_a"] = np.ones(2)
    monkeypatch.setattr(weight_loaders.download, "maybe_download", lambda p: p)
    monkeypatch.setattr(model, "restore_params", lambda *_, **__: loaded)
    loader = config.get_config(BASELINE).weight_loader
    if fault:
        with pytest.raises(ValueError, match="Strict checkpoint mismatch"):
            loader.load(expected)
    else:
        result = loader.load(expected)
        assert set(result) == set(expected)
        np.testing.assert_array_equal(result["lora_a"], 7)
        assert result["lora_a"].dtype == np.float32


def test_extra_exception_requires_strict_mode():
    with pytest.raises(ValueError, match="strict=True"):
        weight_loaders.CheckpointWeightLoader("unused", strict_allow_extra_regex=".*")


@pytest.mark.parametrize("include_invalid_touch", [False, True])
def test_no_touch_evaluation_never_reads_sensor_fields(tmp_path, include_invalid_touch):
    raw = sample()
    if include_invalid_touch:
        raw.update(tactile_marker_motion=np.nan, tactile_depth=np.nan, wrist_wrench=np.nan)

    class Policy:
        def infer(self, observation, *, noise):
            assert set(observation) == {"image", "wrist_image", "state", "prompt"}
            np.testing.assert_array_equal(noise, offline.sampling_noise(42, 4, 0, 50, 32))
            return {"actions": np.tile(observation["state"], (50, 1))}

    result = offline.evaluate_dataset(Policy(), [raw], {4: 1}, tmp_path, use_tactile=False, make_plots=False)
    assert result["input_modality"] == "rgb_state"
    assert result["overall"]["first_action"]["position_mm"]["mean"] == 0
    assert result["overall"]["valid_chunk_targets"] == 1


def test_inconsistent_model_modality_rejected():
    base = config.get_config(BASELINE)
    with pytest.raises(ValueError, match="no-tactile"):
        validate_model_contract(
            dataclasses.replace(base, policy_metadata={**base.policy_metadata, "tactile_input": "touch"})
        )


@pytest.mark.parametrize(
    ("gpus", "success"), [("0", True), ("0,1", True), ("0,1,2,3", True), ("0,1,2", False), ("1,01", False), ("", False)]
)
@pytest.mark.parametrize("variant", ["rgb_state", "rgb_state_touch"])
def test_launcher_dry_run_never_starts_training(gpus, success, variant):
    script = Path(__file__).resolve().parents[3] / "scripts/run_tabero_baseline.sh"
    result = subprocess.run(
        ["bash", str(script), "--dry-run", "unit_test_baseline_dry_run", variant],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": gpus},
        capture_output=True,
        text=True,
        check=False,
    )
    assert (result.returncode == 0) == success, result.stderr
    if success:
        assert "--batch-size=4" in result.stdout
        assert "--num-train-steps=3000" in result.stdout
        assert "49999/params" in result.stdout
        assert "JAX devices:" not in result.stdout


@pytest.mark.parametrize("name", [BASELINE, TOUCH])
def test_training_arguments_parse_without_starting_main(monkeypatch, name):
    cfg = config.get_config(name)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            name,
            "--exp-name=unit_test_parse_only",
            f"--weight-loader.params-path={cfg.weight_loader.params_path}",
            "--seed=42",
            "--fsdp-devices=1",
            "--batch-size=4",
            "--num-workers=4",
            "--num-train-steps=3000",
            "--lr-schedule.warmup-steps=100",
            "--lr-schedule.decay-steps=3000",
            "--lr-schedule.peak-lr=1e-5",
            "--lr-schedule.decay-lr=1e-6",
            "--eval-interval=250",
            "--eval-num-batches=173",
            "--save-interval=100",
            "--keep-period=500",
            "--no-wandb-enabled",
        ],
    )
    parsed = config.cli()
    assert parsed.batch_size == 4
    assert parsed.num_train_steps == 3000
    assert parsed.lr_schedule == cfg.lr_schedule
    assert not parsed.resume
