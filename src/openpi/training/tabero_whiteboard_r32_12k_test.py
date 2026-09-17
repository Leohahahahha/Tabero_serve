"""CPU-only contracts for the one-GPU tactile-rank-32 whiteboard experiment."""

# The environment must be set before importing the JAX-backed training config.
# ruff: noqa: E402, I001

import asyncio
import dataclasses
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest

os.environ["JAX_PLATFORMS"] = "cpu"

from openpi.training import config
from openpi.training import checkpoints


NEXT_R32_CONFIG = "pi0_lora_tabero_whiteboard_next_state_force_tactile_r32_12k"
SENT_R32_CONFIG = "pi0_lora_tabero_whiteboard_sent_command_force_tactile_r32_12k"
NEXT_R16_CONFIG = "pi0_lora_tabero_whiteboard_next_state_force_20k"
SENT_R16_CONFIG = "pi0_lora_tabero_whiteboard_sent_command_force_20k"
LAUNCHER = Path(__file__).resolve().parents[3] / "scripts/run_tabero_whiteboard_pair_r32_12k.sh"


@pytest.mark.parametrize(
    ("name", "r16_name", "dataset"),
    [
        (NEXT_R32_CONFIG, NEXT_R16_CONFIG, "test2_tabero_next_state_compact"),
        (SENT_R32_CONFIG, SENT_R16_CONFIG, "test2_tabero_sent_command_compact"),
    ],
)
def test_r32_12k_config_contract(name, r16_name, dataset):
    cfg = config.get_config(name)
    r16 = config.get_config(r16_name)

    assert cfg.model == dataclasses.replace(
        r16.model,
        tactile_prefix_lora_rank=32,
        tactile_prefix_lora_alpha=32.0,
    )
    assert cfg.model.effective_action_dim == 13
    assert cfg.model.tactile_prefix_lora_rank == 32
    assert cfg.model.tactile_prefix_lora_alpha == 32.0
    assert cfg.data == r16.data
    assert cfg.data.repo_id == f"local/{dataset}"
    assert cfg.weight_loader == r16.weight_loader
    assert cfg.freeze_filter == r16.freeze_filter
    assert cfg.optimizer == r16.optimizer
    assert cfg.batch_size == 2
    assert cfg.num_workers == 4
    assert cfg.num_train_steps == 12_000
    assert cfg.eval_interval == 1_000
    assert cfg.eval_num_batches == 500
    assert cfg.save_interval == cfg.keep_period == 4_000
    assert cfg.wait_for_checkpoint_on_save
    assert cfg.lr_schedule.warmup_steps == 500
    assert cfg.lr_schedule.peak_lr == 1e-5
    assert cfg.lr_schedule.decay_steps == 12_000
    assert cfg.lr_schedule.decay_lr == 1e-6
    assert cfg.policy_metadata["tactile_lora_rank"] == 32
    assert cfg.policy_metadata["tactile_lora_alpha"] == 32.0
    assert cfg.policy_metadata["checkpoint_mode"] == "wait_after_each_save"
    assert cfg.assets_dirs != r16.assets_dirs


def test_r32_configs_have_separate_assets():
    next_cfg = config.get_config(NEXT_R32_CONFIG)
    sent_cfg = config.get_config(SENT_R32_CONFIG)
    assert next_cfg.data.repo_id != sent_cfg.data.repo_id
    assert next_cfg.assets_dirs != sent_cfg.assets_dirs


def test_old_configs_do_not_wait_after_each_checkpoint():
    assert not config.get_config(NEXT_R16_CONFIG).wait_for_checkpoint_on_save
    assert not config.get_config(SENT_R16_CONFIG).wait_for_checkpoint_on_save


def test_assets_callback_defers_until_orbax_directory_signal(monkeypatch, tmp_path):
    def save_assets(directory):
        (directory / "asset.txt").write_text("ok")

    process_index_threads = []

    def process_index():
        process_index_threads.append(threading.get_ident())
        return 0

    deferred_coroutines = []
    sentinel = object()

    def deferred_future(coroutine):
        deferred_coroutines.append(coroutine)
        return sentinel

    monkeypatch.setattr(checkpoints.jax, "process_index", process_index)
    monkeypatch.setattr(checkpoints.future, "CommitFutureAwaitingContractedSignals", deferred_future)

    futures = asyncio.run(
        checkpoints.CallbackHandler().async_save(
            tmp_path,
            checkpoints.CallbackSave(save_assets),
        )
    )
    assert process_index_threads == [threading.get_ident()]
    assert futures == [sentinel]
    assert len(deferred_coroutines) == 1
    assert not (tmp_path / "asset.txt").exists()
    deferred_coroutines[0].close()


def test_assets_callback_is_skipped_on_non_primary_process(monkeypatch, tmp_path):
    monkeypatch.setattr(checkpoints.jax, "process_index", lambda: 1)
    futures = asyncio.run(
        checkpoints.CallbackHandler().async_save(
            tmp_path,
            checkpoints.CallbackSave(lambda directory: (directory / "asset.txt").write_text("unexpected")),
        )
    )
    assert futures == []
    assert not (tmp_path / "asset.txt").exists()


@pytest.mark.parametrize("wait_until_finished", [False, True])
def test_save_state_wait_contract(wait_until_finished):
    @dataclasses.dataclass
    class FakeState:
        params: dict
        ema_params: None = None

    class FakeManager:
        def __init__(self):
            self.saved = []
            self.wait_calls = 0

        def save(self, step, items):
            self.saved.append((step, items))

        def wait_until_finished(self):
            self.wait_calls += 1

    manager = FakeManager()
    checkpoints.save_state(
        manager,
        FakeState(params={"weight": 1}),
        object(),
        4_000,
        wait_until_finished=wait_until_finished,
    )
    assert len(manager.saved) == 1
    assert manager.saved[0][0] == 4_000
    assert manager.wait_calls == int(wait_until_finished)


def test_r32_launcher_dry_run_matches_config_contract(tmp_path):
    initial = tmp_path / "params"
    initial.mkdir()
    result = subprocess.run(
        ["bash", str(LAUNCHER), "--dry-run", "unit_whiteboard_r32_pair"],
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "0",
            "TABERO_PYTHON": sys.executable,
            "TABERO_INITIAL_PARAMS": str(initial),
            "TABERO_OUTPUT_ROOT": str(tmp_path / "outputs"),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.index(NEXT_R32_CONFIG) < result.stdout.index(SENT_R32_CONFIG)
    assert result.stdout.count("--batch-size=2") == 2
    assert result.stdout.count("--num-train-steps=12000") == 2
    assert result.stdout.count("--lr-schedule.decay-steps=12000") == 2
    assert result.stdout.count("--eval-num-batches=500") == 2
    assert result.stdout.count("--save-interval=4000") == 2
    assert result.stdout.count("--keep-period=4000") == 2
    assert result.stdout.count("--wandb-enabled") == 2
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
  config=; run=; checkpoints=; steps=; after_train=no
  for arg in "$@"; do
    if [[ $after_train == yes && -z $config ]]; then config=$arg; after_train=no; continue; fi
    [[ $arg == scripts/train.py ]] && after_train=yes
    case "$arg" in
      --exp-name=*) run=${arg#*=} ;;
      --checkpoint-base-dir=*) checkpoints=${arg#*=} ;;
      --num-train-steps=*) steps=${arg#*=} ;;
    esac
  done
  printf '%s\n' "$config" >> "$FAKE_ORDER_FILE"
  mkdir -p "$checkpoints/$config/$run/$steps/params"
  printf 'fake-wandb\n' > "$checkpoints/$config/$run/wandb_id.txt"
  exit 0
fi
exit 3
"""
    )
    path.chmod(0o755)


def test_r32_launcher_requires_12000_before_second_run(tmp_path):
    initial = tmp_path / "params"
    initial.mkdir()
    fake_python = tmp_path / "fake-python"
    _write_fake_python(fake_python)
    output = tmp_path / "outputs"
    order = tmp_path / "order.txt"
    result = subprocess.run(
        ["bash", str(LAUNCHER), "unit_r32_pair"],
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "0",
            "TABERO_PYTHON": str(fake_python),
            "TABERO_INITIAL_PARAMS": str(initial),
            "TABERO_OUTPUT_ROOT": str(output),
            "FAKE_ORDER_FILE": str(order),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert order.read_text().splitlines() == [NEXT_R32_CONFIG, SENT_R32_CONFIG]
    status = (output / "whiteboard_pairs/unit_r32_pair/status.txt").read_text()
    assert "state=complete" in status
    assert "stage=all_complete" in status
    for name, label in ((NEXT_R32_CONFIG, "next_state"), (SENT_R32_CONFIG, "sent_command")):
        final = output / f"checkpoints/{name}/unit_r32_pair_{label}/12000/params"
        assert final.is_dir()
        assert not (final.parents[1] / "20000").exists()
