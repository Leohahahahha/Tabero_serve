"""CPU-only contracts for the paired whiteboard action-source experiments."""

# The environment must be set before importing the JAX-backed training config.
# ruff: noqa: E402, I001

import dataclasses
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest

os.environ["JAX_PLATFORMS"] = "cpu"

from openpi import transforms
from openpi.policies import libero_policy
from openpi.training import config


NEXT_CONFIG = "pi0_lora_tabero_whiteboard_next_state_force_20k"
SENT_CONFIG = "pi0_lora_tabero_whiteboard_sent_command_force_20k"
V3_CONFIG = "pi0_lora_tabero_v3_touch_20k"
VALIDATION_EPISODES = (1, 7, 17)


def _marker_reference_grid():
    y = np.rint(np.linspace(0, 239, 9)).astype(np.float32)
    x = np.rint(np.linspace(0, 319, 11)).astype(np.float32)
    gx, gy = np.meshgrid(x, y)
    side = np.stack((gx, gy), axis=-1).reshape(99, 2)
    return np.concatenate((side, side)).astype(np.float32)


@pytest.mark.parametrize(
    ("name", "dataset", "source_mode", "label_source", "step_offset"),
    [
        (
            NEXT_CONFIG,
            "test2_tabero_next_state_compact",
            "next-state",
            "next_observation_state_within_episode",
            1,
        ),
        (
            SENT_CONFIG,
            "test2_tabero_sent_command_compact",
            "sent-command",
            "synchronized_absolute_sent_command_current_frame",
            0,
        ),
    ],
)
def test_whiteboard_config_dataset_and_split(name, dataset, source_mode, label_source, step_offset):
    cfg = config.get_config(name)
    data = cfg.data.base_config
    assert cfg.data.repo_id == f"local/{dataset}"
    assert data.root == f"/data/yanghaojun/datasets/{dataset}"
    assert data.validation_episodes == VALIDATION_EPISODES
    assert data.episodes == tuple(i for i in range(39) if i not in VALIDATION_EPISODES)
    assert random.Random(42).sample(range(39), 3) == [7, 1, 17]
    assert data.prompt_from_task
    assert cfg.data.extra_delta_transform
    assert cfg.data.use_so3_relative_actions
    assert tuple(data.action_sequence_keys) == ("actions", "wrist_wrench")
    assert "wrist_wrench" in data.columns
    assert cfg.policy_metadata["action_source_mode"] == source_mode
    assert cfg.policy_metadata["action_label_source"] == label_source
    assert cfg.policy_metadata["action_state_step_offset"] == step_offset
    assert cfg.policy_metadata["validation_split_seed"] == 42
    assert tuple(cfg.policy_metadata["validation_episode_ids"]) == VALIDATION_EPISODES
    assert cfg.policy_metadata["prediction_layout"] == "7d_action_then_6d_wrist_wrench"
    assert cfg.policy_metadata["predicts_wrench"]
    assert cfg.policy_metadata["wrist_wrench_dim"] == 6
    assert cfg.policy_metadata["wrist_wrench_loss_weight"] == 0.1
    assert cfg.policy_metadata["wrist_wrench_units"] == ["N", "N", "N", "N_m", "N_m", "N_m"]
    assert cfg.policy_metadata["wrist_wrench_frame"] == "K"


@pytest.mark.parametrize("name", [NEXT_CONFIG, SENT_CONFIG])
def test_whiteboard_training_settings_match_v3(name):
    cfg = config.get_config(name)
    base = config.get_config(V3_CONFIG)
    expected_model = dataclasses.replace(
        base.model,
        supervised_action_dim=None,
        tactile_loss_weight=config.TACTILE_LOSS_WEIGHT,
        padding_loss_weight=0.0,
    )
    assert cfg.model == expected_model
    assert cfg.model.effective_action_dim == 13
    assert cfg.model.tactile_dim == 6
    assert cfg.model.supervised_action_dim is None
    assert cfg.model.tactile_loss_weight == 0.1
    assert cfg.model.padding_loss_weight == 0.0
    assert cfg.weight_loader == base.weight_loader
    assert cfg.freeze_filter == base.freeze_filter
    assert cfg.optimizer == base.optimizer
    assert cfg.ema_decay == base.ema_decay
    assert cfg.batch_size == base.batch_size == 4
    assert cfg.num_workers == base.num_workers == 4
    assert cfg.num_train_steps == base.num_train_steps == 20_000
    assert cfg.log_interval == base.log_interval == 10
    assert cfg.eval_interval == base.eval_interval == 1_000
    assert cfg.eval_num_batches == 250
    assert cfg.save_interval == cfg.keep_period == 4_000
    assert cfg.checkpoint_step_is_update_count
    assert cfg.wandb_enabled
    assert not cfg.wandb_log_images
    assert dataclasses.asdict(cfg.lr_schedule) == dataclasses.asdict(base.lr_schedule)
    assert cfg.weight_loader.params_path.endswith("pi0_lora_tacfield_tabero/49999/params")


def test_whiteboard_configs_have_independent_asset_ids():
    next_cfg = config.get_config(NEXT_CONFIG)
    sent_cfg = config.get_config(SENT_CONFIG)
    assert next_cfg.data.repo_id != sent_cfg.data.repo_id
    assert next_cfg.assets_dirs != sent_cfg.assets_dirs


def test_action_wrench_transform_and_so3_round_trip():
    cfg = config.get_config(NEXT_CONFIG)
    state = np.array([0.4, 0.1, 0.2, 2.2, -2.1, 0.05, 0.02], dtype=np.float32)
    actions = np.repeat(state[None], 50, axis=0)
    actions[:, :3] += np.array([0.001, -0.002, 0.003], dtype=np.float32)
    wrist_wrench = np.arange(50 * 6, dtype=np.float32).reshape(50, 6) / 10
    reference = _marker_reference_grid()
    motion = np.repeat(reference[None], 9, axis=0)
    raw = {
        "image": np.zeros((32, 32, 3), dtype=np.uint8),
        "wrist_image": np.zeros((32, 32, 3), dtype=np.uint8),
        "state": state,
        "actions": actions.copy(),
        "wrist_wrench": wrist_wrench.copy(),
        "tactile_marker_motion": motion,
    }
    encoded = libero_policy.TaberoActionWrenchInputs(cfg.model.model_type)(raw)
    assert encoded["actions"].shape == (50, 13)
    np.testing.assert_array_equal(encoded["actions"][:, :7], actions)
    np.testing.assert_array_equal(encoded["actions"][:, 7:13], wrist_wrench)
    relative = transforms.RelativePoseActions()(encoded)
    np.testing.assert_array_equal(relative["actions"][:, 7:13], wrist_wrench)
    absolute = transforms.AbsolutePoseActions()(relative)
    decoded = libero_policy.TaberoActionWrenchOutputs()(absolute)
    np.testing.assert_allclose(decoded["actions"], actions, atol=1e-6)
    np.testing.assert_array_equal(decoded["wrist_wrench"], wrist_wrench)


def test_action_wrench_transform_rejects_missing_or_invalid_wrench():
    cfg = config.get_config(NEXT_CONFIG)
    reference = _marker_reference_grid()
    raw = {
        "image": np.zeros((32, 32, 3), dtype=np.uint8),
        "wrist_image": np.zeros((32, 32, 3), dtype=np.uint8),
        "state": np.array([0.4, 0.1, 0.2, 2.2, -2.1, 0.05, 0.02], dtype=np.float32),
        "actions": np.zeros((50, 7), dtype=np.float32),
        "tactile_marker_motion": np.repeat(reference[None], 9, axis=0),
    }
    adapter = libero_policy.TaberoActionWrenchInputs(cfg.model.model_type)
    with pytest.raises(KeyError, match="wrist_wrench"):
        adapter(raw)
    raw["wrist_wrench"] = np.full((50, 6), np.nan, dtype=np.float32)
    with pytest.raises(ValueError, match="finite float32"):
        adapter(raw)


def test_pair_launcher_dry_run_is_ordered_and_non_training(tmp_path):
    initial = tmp_path / "params"
    initial.mkdir()
    script = Path(__file__).resolve().parents[3] / "scripts/run_tabero_whiteboard_pair_20k.sh"
    result = subprocess.run(
        ["bash", str(script), "--dry-run", "unit_whiteboard_pair"],
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "0,1",
            "TABERO_PYTHON": sys.executable,
            "TABERO_INITIAL_PARAMS": str(initial),
            "TABERO_OUTPUT_ROOT": str(tmp_path / "outputs"),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.index(NEXT_CONFIG) < result.stdout.index(SENT_CONFIG)
    assert result.stdout.count("--num-train-steps=20000") == 2
    assert result.stdout.count("--save-interval=4000") == 2
    assert result.stdout.count("--wandb-enabled") == 2
    assert "target: 7D action + 6D wrist wrench" in result.stdout
    assert "Order: next_state must exit 0 before sent_command starts." in result.stdout
    assert not (tmp_path / "outputs").exists()


def _write_fake_python(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ " $* " == *" scripts/prepare_tabero_smoke.py "* ]]; then
  config=; assets=; audit=
  for arg in "$@"; do
    case "$arg" in
      --config=*) config=${arg#*=} ;;
      --assets-base-dir=*) assets=${arg#*=} ;;
      --output-dir=*) audit=${arg#*=} ;;
    esac
  done
  if [[ $config == *next_state* ]]; then dataset=test2_tabero_next_state_compact; else dataset=test2_tabero_sent_command_compact; fi
  mkdir -p "$assets/$config/local/$dataset" "$audit"
  printf '{}\n' > "$assets/$config/local/$dataset/norm_stats.json"
  printf '{}\n' > "$assets/$config/local/$dataset/split_provenance.json"
  exit 0
fi
if [[ ${1:-} == -c ]]; then exit 0; fi
if [[ " $* " == *" scripts/train.py "* ]]; then
  config=; run=; checkpoints=; after_train=no
  for arg in "$@"; do
    if [[ $after_train == yes && -z $config ]]; then config=$arg; after_train=no; continue; fi
    [[ $arg == scripts/train.py ]] && after_train=yes
    case "$arg" in
      --exp-name=*) run=${arg#*=} ;;
      --checkpoint-base-dir=*) checkpoints=${arg#*=} ;;
    esac
  done
  printf '%s\n' "$config" >> "$FAKE_ORDER_FILE"
  if [[ ${FAKE_FAIL_NEXT:-0} == 1 && $config == *next_state* ]]; then exit 9; fi
  mkdir -p "$checkpoints/$config/$run/20000/params"
  printf 'fake-wandb\n' > "$checkpoints/$config/$run/wandb_id.txt"
  exit 0
fi
exit 3
"""
    )
    path.chmod(0o755)


def _run_fake_pair(tmp_path: Path, *, fail_next: bool) -> subprocess.CompletedProcess[str]:
    initial = tmp_path / "params"
    initial.mkdir()
    fake_python = tmp_path / "fake-python"
    _write_fake_python(fake_python)
    output = tmp_path / "outputs"
    order = tmp_path / "order.txt"
    script = Path(__file__).resolve().parents[3] / "scripts/run_tabero_whiteboard_pair_20k.sh"
    return subprocess.run(
        ["bash", str(script), "unit_pair"],
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "0",
            "TABERO_PYTHON": str(fake_python),
            "TABERO_INITIAL_PARAMS": str(initial),
            "TABERO_OUTPUT_ROOT": str(output),
            "FAKE_ORDER_FILE": str(order),
            "FAKE_FAIL_NEXT": "1" if fail_next else "0",
        },
        capture_output=True,
        text=True,
        check=False,
    )


def test_pair_launcher_requires_next_success_before_sent(tmp_path):
    result = _run_fake_pair(tmp_path, fail_next=True)
    assert result.returncode == 9
    assert (tmp_path / "order.txt").read_text().splitlines() == [NEXT_CONFIG]
    status = (tmp_path / "outputs/whiteboard_pairs/unit_pair/status.txt").read_text()
    assert "state=failed" in status
    assert "stage=train_next_state" in status


def test_pair_launcher_completes_both_in_order(tmp_path):
    result = _run_fake_pair(tmp_path, fail_next=False)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "order.txt").read_text().splitlines() == [NEXT_CONFIG, SENT_CONFIG]
    status = (tmp_path / "outputs/whiteboard_pairs/unit_pair/status.txt").read_text()
    assert "state=complete" in status
    assert "stage=all_complete" in status
    for name, label in ((NEXT_CONFIG, "next_state"), (SENT_CONFIG, "sent_command")):
        final = tmp_path / f"outputs/checkpoints/{name}/unit_pair_{label}/20000/params"
        assert final.is_dir()
