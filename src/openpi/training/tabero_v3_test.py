"""CPU-only contract tests for the v3 tactile 20k training profile."""

import dataclasses
import hashlib
import os
from pathlib import Path
import subprocess
import sys

os.environ["JAX_PLATFORMS"] = "cpu"

from openpi.training import config
from scripts import train

CONFIG_NAME = "pi0_lora_tabero_v3_touch_20k"


def test_v3_touch_profile_is_fresh_published_tabero_training():
    cfg = config.get_config(CONFIG_NAME)
    data = cfg.data.base_config
    assert data.root == "/data/yanghaojun/datasets/tabero_lerobot_compact_v3"
    assert cfg.data.repo_id == "local/tabero_lerobot_compact_v3"
    assert data.episodes == tuple(i for i in range(29) if i not in (4, 14, 24))
    assert data.validation_episodes == (4, 14, 24)
    assert data.prompt_from_task
    assert cfg.data.use_so3_relative_actions
    assert "tactile_marker_motion" in data.columns
    assert cfg.data.assets.assets_dir is None
    assert cfg.weight_loader.params_path.endswith("pi0_lora_tacfield_tabero/49999/params")
    assert "outputs/checkpoints" not in cfg.weight_loader.params_path
    assert cfg.weight_loader.strict
    assert cfg.model.tactile_prefix_lora_rank == 16
    assert cfg.model.tactile_prefix_lora_alpha == 16.0
    assert cfg.model.supervised_action_dim == 7


def test_v3_schedule_wandb_and_checkpoint_contract():
    cfg = config.get_config(CONFIG_NAME)
    assert cfg.project_name == "tabero-vtla"
    assert cfg.wandb_enabled
    assert not cfg.wandb_log_images
    assert cfg.num_train_steps == 20_000
    assert cfg.batch_size == 4
    assert cfg.eval_interval == 1_000
    assert cfg.eval_num_batches == 172
    assert cfg.save_interval == cfg.keep_period == 4_000
    assert cfg.checkpoint_step_is_update_count
    assert cfg.lr_schedule.warmup_steps == 500
    assert cfg.lr_schedule.decay_steps == 20_000
    due = [train.checkpoint_step_and_due(cfg, step, 0) for step in range(cfg.num_train_steps)]
    assert [step for step, save in due if save] == [4_000, 8_000, 12_000, 16_000, 20_000]


def test_cli_arguments_parse_without_training(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            CONFIG_NAME,
            "--exp-name=unit-v3",
            "--wandb-enabled",
            "--no-wandb-log-images",
            "--num-train-steps=20000",
            "--save-interval=4000",
            "--keep-period=4000",
        ],
    )
    parsed = config.cli()
    assert parsed.wandb_enabled
    assert not parsed.wandb_log_images
    assert parsed.num_train_steps == 20_000
    assert parsed.save_interval == parsed.keep_period == 4_000


def test_launcher_dry_run_is_non_training(tmp_path):
    initial = tmp_path / "params"
    initial.mkdir()
    stats = tmp_path / "norm_stats.json"
    provenance = tmp_path / "split_provenance.json"
    stats.write_text("{}")
    provenance.write_text("{}")
    script = Path(__file__).resolve().parents[3] / "scripts/run_tabero_v3_touch_20k.sh"
    result = subprocess.run(
        ["bash", str(script), "--dry-run", "unit-v3-dry-run"],
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "0,1,2,3",
            "TABERO_PYTHON": sys.executable,
            "TABERO_INITIAL_PARAMS": str(initial),
            "TABERO_STATS_PATH": str(stats),
            "TABERO_PROVENANCE_PATH": str(provenance),
            "TABERO_EXPECTED_STATS_SHA256": hashlib.sha256(stats.read_bytes()).hexdigest(),
            "TABERO_OUTPUT_ROOT": str(tmp_path / "outputs"),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--num-train-steps=20000" in result.stdout
    assert "--save-interval=4000" in result.stdout
    assert "--wandb-enabled" in result.stdout
    assert "--no-wandb-log-images" in result.stdout
    assert "JAX devices:" not in result.stdout


def test_legacy_checkpoint_numbering_remains_unchanged():
    cfg = dataclasses.replace(config.get_config(CONFIG_NAME), checkpoint_step_is_update_count=False)
    assert train.checkpoint_step_and_due(cfg, 4_000, 0) == (4_000, True)
