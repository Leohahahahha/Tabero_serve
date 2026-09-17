"""CPU-only contracts for two-GPU full-parameter whiteboard training."""

# The environment must be set before importing the JAX-backed training config.
# ruff: noqa: E402, I001

import dataclasses
import os
from pathlib import Path
import subprocess
import sys

from flax import nnx
import jax
import jax.numpy as jnp
import optax
import pytest

os.environ["JAX_PLATFORMS"] = "cpu"

from openpi.training import config
from openpi.training import optimizer
from scripts import train


NEXT_CONFIG = "pi0_tabero_whiteboard_next_state_force_full_ft_sgd_12k"
SENT_CONFIG = "pi0_tabero_whiteboard_sent_command_force_full_ft_sgd_12k"
NEXT_MIXED_CONFIG = "pi0_tabero_whiteboard_next_state_force_full_ft_mixed_bf16_adamw_12k"
SENT_MIXED_CONFIG = "pi0_tabero_whiteboard_sent_command_force_full_ft_mixed_bf16_adamw_12k"
NEXT_MIXED_LOWMEM_CONFIG = "pi0_tabero_whiteboard_next_state_force_full_ft_mixed_bf16_lowmem_adamw_12k"
SENT_MIXED_LOWMEM_CONFIG = "pi0_tabero_whiteboard_sent_command_force_full_ft_mixed_bf16_lowmem_adamw_12k"
NEXT_FP32_ADAM_CONFIG = "pi0_tabero_whiteboard_next_state_force_full_ft_12k"
SENT_FP32_ADAM_CONFIG = "pi0_tabero_whiteboard_sent_command_force_full_ft_12k"
NEXT_LORA_CONFIG = "pi0_lora_tabero_whiteboard_next_state_force_20k"
SENT_LORA_CONFIG = "pi0_lora_tabero_whiteboard_sent_command_force_20k"
LAUNCHER = Path(__file__).resolve().parents[3] / "scripts/run_tabero_whiteboard_pair_full_ft_2gpu_12k.sh"


@pytest.mark.parametrize(
    ("name", "lora_name", "dataset"),
    [
        (NEXT_CONFIG, NEXT_LORA_CONFIG, "test2_tabero_next_state_compact"),
        (SENT_CONFIG, SENT_LORA_CONFIG, "test2_tabero_sent_command_compact"),
    ],
)
def test_full_finetune_config_contract(name, lora_name, dataset):
    cfg = config.get_config(name)
    lora_cfg = config.get_config(lora_name)

    assert cfg.model == dataclasses.replace(lora_cfg.model, tactile_prefix_lora_rank=0)
    assert cfg.model.effective_action_dim == 13
    assert cfg.model.tactile_prefix_lora_rank == 0
    assert cfg.data == lora_cfg.data
    assert cfg.data.repo_id == f"local/{dataset}"
    assert cfg.freeze_filter is nnx.Nothing
    param = nnx.Param(jnp.ones(1))
    assert cfg.trainable_filter(("PaliGemma", "img", "kernel"), param)
    assert cfg.trainable_filter(("PaliGemma", "llm", "lora_a"), param)
    assert cfg.trainable_filter(("tactile_prefix_encoder", "out_proj", "kernel"), param)
    assert cfg.weight_loader.strict
    assert cfg.weight_loader.strict_allow_missing_regex is None
    assert cfg.weight_loader.params_path.endswith("pi0_lora_tacfield_tabero/49999/params")
    assert cfg.ema_decay is None
    assert cfg.optimizer == optimizer.StatelessSGD(clip_gradient_norm=1.0)
    assert cfg.fsdp_devices == 2
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
    assert cfg.policy_metadata["optimization_method"] == "full_parameter_finetuning"
    assert cfg.policy_metadata["trainable_scope"] == "all_parameter_leaves"
    assert cfg.policy_metadata["tactile_adaptation"] == "full_tcn_and_all_model_parameters"
    assert cfg.policy_metadata["fsdp_devices"] == 2
    assert cfg.policy_metadata["global_batch_size"] == 2
    assert cfg.policy_metadata["optimizer"] == "stateless_sgd"
    assert cfg.policy_metadata["optimizer_state_strategy"] == "no_momentum_or_second_moment_tensors"
    assert cfg.parameter_dtype_policy.name == "full_float32"
    assert cfg.assets_dirs != lora_cfg.assets_dirs


@pytest.mark.parametrize("name", [NEXT_MIXED_CONFIG, SENT_MIXED_CONFIG])
def test_mixed_bfloat16_full_finetune_config_contract(name):
    cfg = config.get_config(name)

    assert cfg.freeze_filter is nnx.Nothing
    assert cfg.optimizer == optimizer.AdamW(clip_gradient_norm=1.0, moment_dtype="float32")
    assert cfg.parameter_dtype_policy.name == "mixed_bfloat16_fp32_train_state"
    assert cfg.parameter_dtype_policy.default_trainable_dtype == "bfloat16"
    assert cfg.parameter_dtype_policy.gradient_dtype == "float32"
    assert cfg.parameter_dtype_policy.optimizer_state_dtype == "float32"
    assert cfg.policy_metadata["parameter_dtype_policy"] == "mixed_bfloat16_fp32_train_state"
    assert cfg.policy_metadata["gradient_storage_dtype"] == "float32"
    assert cfg.policy_metadata["gradient_cast_stage"] == "after_autodiff_before_clipping_and_optimizer"
    assert cfg.policy_metadata["optimizer_state_dtype"] == "float32"
    assert cfg.lr_schedule.peak_lr == cfg.policy_metadata["peak_lr"] == 2e-5
    assert cfg.lr_schedule.decay_lr == cfg.policy_metadata["decay_lr"] == 2e-6
    assert cfg.policy_metadata["trainable_scope"] == "all_parameter_leaves"
    assert cfg.model.tactile_prefix_lora_rank == 0
    assert cfg.fsdp_devices == cfg.batch_size == 2


@pytest.mark.parametrize("name", [NEXT_FP32_ADAM_CONFIG, SENT_FP32_ADAM_CONFIG])
def test_full_float32_adamw_fallback_config_contract(name):
    cfg = config.get_config(name)

    assert cfg.freeze_filter is nnx.Nothing
    assert cfg.optimizer == optimizer.AdamW(clip_gradient_norm=1.0)
    assert cfg.parameter_dtype_policy.name == "full_float32"
    assert cfg.parameter_dtype_policy.default_trainable_dtype == "float32"
    assert cfg.policy_metadata["parameter_dtype_policy"] == "full_float32"
    assert cfg.lr_schedule.peak_lr == cfg.policy_metadata["peak_lr"] == 2e-6
    assert cfg.lr_schedule.decay_lr == cfg.policy_metadata["decay_lr"] == 2e-7
    assert cfg.fsdp_devices == cfg.batch_size == 2


def test_mixed_bfloat16_policy_keeps_sensitive_modules_and_adam_moments_float32():
    cfg = config.get_config(NEXT_MIXED_CONFIG)
    params = nnx.State(
        {
            "PaliGemma": {
                "llm": {
                    "layers": {
                        "mlp": {
                            "kernel": nnx.Param(jnp.ones((2, 3), dtype=jnp.float32)),
                            "lora_a": nnx.Param(jnp.ones((3, 4), dtype=jnp.float32)),
                        },
                        "pre_attention_norm": {"scale": nnx.Param(jnp.ones((5,), dtype=jnp.float32))},
                    }
                },
                "img": {"embedding": {"kernel": nnx.Param(jnp.ones((7, 1), dtype=jnp.float32))}},
            },
            "state_proj": {"kernel": nnx.Param(jnp.ones((8, 1), dtype=jnp.float32))},
            "action_time_mlp_in": {"kernel": nnx.Param(jnp.ones((9, 1), dtype=jnp.float32))},
            "action_out_proj": {"kernel": nnx.Param(jnp.ones((10, 1), dtype=jnp.float32))},
            "tactile_prefix_encoder": {"kernel": nnx.Param(jnp.ones((11, 1), dtype=jnp.float32))},
        }
    )

    cast = train.apply_parameter_dtype_policy(params, cfg)
    assert cast["PaliGemma"]["llm"]["layers"]["mlp"]["kernel"].dtype == jnp.bfloat16
    assert cast["PaliGemma"]["llm"]["layers"]["mlp"]["lora_a"].dtype == jnp.bfloat16
    assert cast["PaliGemma"]["llm"]["layers"]["pre_attention_norm"]["scale"].dtype == jnp.float32
    assert cast["PaliGemma"]["img"]["embedding"]["kernel"].dtype == jnp.float32
    assert cast["state_proj"]["kernel"].dtype == jnp.float32
    assert cast["action_time_mlp_in"]["kernel"].dtype == jnp.float32
    assert cast["action_out_proj"]["kernel"].dtype == jnp.float32
    assert cast["tactile_prefix_encoder"]["kernel"].dtype == jnp.float32

    tx = optimizer.create_optimizer(cfg.optimizer, cfg.lr_schedule, weight_decay_mask=None)
    opt_state = tx.init(cast)
    for shape in ((2, 3), (3, 4), (5,)):
        moments = [x for x in jax.tree_util.tree_leaves(opt_state) if getattr(x, "shape", None) == shape]
        assert len(moments) == 2
        assert all(x.dtype == jnp.float32 for x in moments)

    grads = jax.tree.map(lambda variable: jnp.ones(variable.shape, dtype=variable.dtype), cast)
    cast_grads = train.apply_gradient_dtype_policy(grads, cfg)
    assert all(grad.dtype == jnp.float32 for grad in jax.tree.leaves(cast_grads))
    updates, new_opt_state = tx.update(cast_grads, opt_state, cast)
    updated = optax.apply_updates(cast, updates)
    assert updated["PaliGemma"]["llm"]["layers"]["mlp"]["kernel"].dtype == jnp.bfloat16
    assert updated["state_proj"]["kernel"].dtype == jnp.float32
    floating_state = [x for x in jax.tree.leaves(new_opt_state) if jnp.issubdtype(x.dtype, jnp.floating)]
    assert floating_state
    assert all(x.dtype == jnp.float32 for x in floating_state)


@pytest.mark.parametrize("name", [NEXT_MIXED_LOWMEM_CONFIG, SENT_MIXED_LOWMEM_CONFIG])
def test_mixed_bfloat16_lowmem_fallback_matches_parameter_state_dtypes(name):
    cfg = config.get_config(name)

    assert cfg.optimizer == optimizer.AdamW(clip_gradient_norm=1.0)
    assert cfg.parameter_dtype_policy.name == "mixed_bfloat16"
    assert cfg.parameter_dtype_policy.gradient_dtype == "match_parameter"
    assert cfg.parameter_dtype_policy.optimizer_state_dtype == "match_parameter"
    assert cfg.policy_metadata["gradient_storage_dtype"] == "match_parameter"
    assert cfg.policy_metadata["gradient_cast_stage"] == "none"
    assert cfg.policy_metadata["optimizer_state_dtype"] == "match_parameter"


def test_gradient_norm_and_clipping_accumulate_in_float32_without_promoting_storage():
    grads = {"large_matrix": jnp.array([3.0, 4.0], dtype=jnp.bfloat16)}

    norm = optimizer.global_norm(grads)
    assert norm.dtype == jnp.float32
    assert float(norm) == pytest.approx(5.0)

    tx = optimizer.clip_by_global_norm(1.0)
    clipped, _ = tx.update(grads, tx.init(grads))
    assert clipped["large_matrix"].dtype == jnp.bfloat16
    assert float(optimizer.global_norm(clipped)) == pytest.approx(1.0, abs=0.01)


def test_full_finetune_configs_keep_dataset_assets_separate():
    next_cfg = config.get_config(NEXT_CONFIG)
    sent_cfg = config.get_config(SENT_CONFIG)
    assert next_cfg.data.repo_id != sent_cfg.data.repo_id
    assert next_cfg.assets_dirs != sent_cfg.assets_dirs


def test_full_finetune_launcher_requires_exactly_two_gpus(tmp_path):
    initial = tmp_path / "params"
    initial.mkdir()
    common_env = {
        **os.environ,
        "TABERO_PYTHON": sys.executable,
        "TABERO_INITIAL_PARAMS": str(initial),
        "TABERO_OUTPUT_ROOT": str(tmp_path / "outputs"),
    }
    for visible in ("0", "0,0", "0,1,2"):
        result = subprocess.run(
            ["bash", str(LAUNCHER), "--dry-run", "unit_full_ft_pair"],
            env={**common_env, "CUDA_VISIBLE_DEVICES": visible},
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 2


def test_full_finetune_launcher_dry_run_matches_config(tmp_path):
    initial = tmp_path / "params"
    initial.mkdir()
    result = subprocess.run(
        ["bash", str(LAUNCHER), "--dry-run", "unit_full_ft_pair"],
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "1,2",
            "TABERO_PYTHON": sys.executable,
            "TABERO_INITIAL_PARAMS": str(initial),
            "TABERO_OUTPUT_ROOT": str(tmp_path / "outputs"),
            "TABERO_TARGET_STEPS": "12000",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.index(NEXT_CONFIG) < result.stdout.index(SENT_CONFIG)
    assert result.stdout.count("--fsdp-devices=2") == 2
    assert result.stdout.count("--batch-size=2") == 2
    assert result.stdout.count("--num-train-steps=12000") == 2
    assert result.stdout.count("--lr-schedule.peak-lr=1e-5") == 2
    assert result.stdout.count("--lr-schedule.decay-lr=1e-6") == 2
    assert result.stdout.count("--eval-num-batches=500") == 2
    assert result.stdout.count("--save-interval=4000") == 2
    assert result.stdout.count("--keep-period=4000") == 2
    assert "trainable: all parameters" in result.stdout
    assert "optimizer: stateless SGD" in result.stdout
    assert "Order: next_state must exit 0 before sent_command starts." in result.stdout
    assert not (tmp_path / "outputs").exists()


def test_full_finetune_launcher_mixed_bfloat16_adamw_profile(tmp_path):
    initial = tmp_path / "params"
    initial.mkdir()
    result = subprocess.run(
        ["bash", str(LAUNCHER), "--dry-run", "unit_mixed_full_ft_pair"],
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "1,2",
            "TABERO_FULL_FT_PROFILE": "mixed_bf16_adamw",
            "TABERO_PYTHON": sys.executable,
            "TABERO_INITIAL_PARAMS": str(initial),
            "TABERO_OUTPUT_ROOT": str(tmp_path / "outputs"),
            "TABERO_TARGET_STEPS": "12000",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.index(NEXT_MIXED_CONFIG) < result.stdout.index(SENT_MIXED_CONFIG)
    assert result.stdout.count("--fsdp-devices=2") == 2
    assert result.stdout.count("--lr-schedule.peak-lr=2e-5") == 2
    assert result.stdout.count("--lr-schedule.decay-lr=2e-6") == 2
    assert "profile: mixed_bf16_adamw" in result.stdout
    assert "optimizer: AdamW; mixed BF16/FP32 parameters; FP32 gradients, moments" in result.stdout
    assert not (tmp_path / "outputs").exists()


def test_full_finetune_launcher_full_float32_adamw_profile(tmp_path):
    initial = tmp_path / "params"
    initial.mkdir()
    result = subprocess.run(
        ["bash", str(LAUNCHER), "--dry-run", "unit_fp32_adamw_full_ft_pair"],
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "1,2",
            "TABERO_FULL_FT_PROFILE": "fp32_adamw",
            "TABERO_PYTHON": sys.executable,
            "TABERO_INITIAL_PARAMS": str(initial),
            "TABERO_OUTPUT_ROOT": str(tmp_path / "outputs"),
            "TABERO_TARGET_STEPS": "12000",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.index(NEXT_FP32_ADAM_CONFIG) < result.stdout.index(SENT_FP32_ADAM_CONFIG)
    assert result.stdout.count("--fsdp-devices=2") == 2
    assert "profile: fp32_adamw" in result.stdout
    assert "full FP32 parameter, gradient and optimizer-state storage" in result.stdout
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


def test_full_finetune_launcher_runs_pair_in_order(tmp_path):
    initial = tmp_path / "params"
    initial.mkdir()
    fake_python = tmp_path / "fake-python"
    _write_fake_python(fake_python)
    output = tmp_path / "outputs"
    order = tmp_path / "order.txt"
    result = subprocess.run(
        ["bash", str(LAUNCHER), "unit_full_ft_pair"],
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "1,2",
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
    assert order.read_text().splitlines() == [NEXT_CONFIG, SENT_CONFIG]
    status = (output / "whiteboard_pairs/unit_full_ft_pair/status.txt").read_text()
    assert "state=complete" in status
    assert "stage=all_complete" in status
