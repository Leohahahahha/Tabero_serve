#!/usr/bin/env bash
# User-operated two-GPU full-parameter whiteboard training. The second run starts only after the first succeeds.
set -euo pipefail

usage() {
  echo 'Usage: CUDA_VISIBLE_DEVICES=1,2 bash scripts/run_tabero_whiteboard_pair_full_ft_2gpu_12k.sh [--dry-run|--resume] PAIR_NAME'
  echo 'Runs next-state first, then sent-command, predicting 7D action + 6D wrist wrench in separate W&B runs.'
  echo 'Set TABERO_FULL_FT_PROFILE=mixed_bf16_adamw for mixed parameters with FP32 gradients/Adam state; mixed_bf16_lowmem_adamw keeps gradient/state storage low precision. The legacy default is fp32_sgd.'
}

if [[ ${1:-} == --help ]]; then usage; exit 0; fi
mode=new
if [[ ${1:-} == --dry-run ]]; then mode=dry-run; shift; fi
if [[ ${1:-} == --resume ]]; then mode=resume; shift; fi
if (( $# != 1 )); then usage >&2; exit 2; fi
pair_name=$1
if [[ ! $pair_name =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then echo 'Invalid pair name.' >&2; exit 2; fi
if [[ ! ${CUDA_VISIBLE_DEVICES:-} =~ ^[0-9]+,[0-9]+$ ]]; then
  echo 'Set CUDA_VISIBLE_DEVICES to exactly two permitted GPU indices, with no spaces.' >&2
  exit 2
fi

IFS=',' read -r -a gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
gpu_0=$((10#${gpu_ids[0]}))
gpu_1=$((10#${gpu_ids[1]}))
if (( gpu_0 == gpu_1 )); then echo 'GPU indices must be different.' >&2; exit 2; fi
gpu_count=2

cd /home/yanghaojun/Tabero-VTLA
python_bin=${TABERO_PYTHON:-/data/yanghaojun/envs/tabero-smoke/bin/python}
initial_params=${TABERO_INITIAL_PARAMS:-/data/yanghaojun/checkpoints/tabero-pretrained/checkpoints/pi0_lora_tacfield_tabero/pi0_lora_tacfield_tabero/49999/params}
output_root=${TABERO_OUTPUT_ROOT:-/data/yanghaojun/outputs}
wandb_project=${TABERO_WANDB_PROJECT:-tabero-vtla}
target_steps=${TABERO_TARGET_STEPS:-20000}
if [[ ! $target_steps =~ ^[1-9][0-9]*$ ]]; then echo 'TABERO_TARGET_STEPS must be a positive integer.' >&2; exit 2; fi
train_batch_size=2
eval_num_batches=500
save_interval=4000
profile=${TABERO_FULL_FT_PROFILE:-fp32_sgd}
case $profile in
  mixed_bf16_adamw)
    configs=(pi0_tabero_whiteboard_next_state_force_full_ft_mixed_bf16_adamw_12k pi0_tabero_whiteboard_sent_command_force_full_ft_mixed_bf16_adamw_12k)
    optimizer_label='AdamW; mixed BF16/FP32 parameters; FP32 gradients, moments, loss and norm reductions'
    peak_lr=2e-5
    decay_lr=2e-6
    ;;
  mixed_bf16_lowmem_adamw)
    configs=(pi0_tabero_whiteboard_next_state_force_full_ft_mixed_bf16_lowmem_adamw_12k pi0_tabero_whiteboard_sent_command_force_full_ft_mixed_bf16_lowmem_adamw_12k)
    optimizer_label='AdamW; mixed BF16/FP32 parameters; gradients and moments match parameter dtype'
    peak_lr=2e-5
    decay_lr=2e-6
    ;;
  fp32_adamw)
    configs=(pi0_tabero_whiteboard_next_state_force_full_ft_12k pi0_tabero_whiteboard_sent_command_force_full_ft_12k)
    optimizer_label='AdamW; full FP32 parameter, gradient and optimizer-state storage'
    peak_lr=2e-6
    decay_lr=2e-7
    ;;
  fp32_sgd)
    configs=(pi0_tabero_whiteboard_next_state_force_full_ft_sgd_12k pi0_tabero_whiteboard_sent_command_force_full_ft_sgd_12k)
    optimizer_label='stateless SGD; full FP32 parameter storage'
    peak_lr=1e-5
    decay_lr=1e-6
    ;;
  *)
    echo 'TABERO_FULL_FT_PROFILE must be mixed_bf16_adamw, mixed_bf16_lowmem_adamw, fp32_adamw or fp32_sgd.' >&2
    exit 2
    ;;
esac
dataset_names=(test2_tabero_next_state_compact test2_tabero_sent_command_compact)
labels=(next_state sent_command)

if [[ ! -x $python_bin ]]; then echo "Python environment missing: $python_bin" >&2; exit 2; fi
if [[ ! -d $initial_params ]]; then echo "Published Tabero params missing: $initial_params" >&2; exit 2; fi

build_train_command() {
  local config=$1 run_name=$2 resume_arg=$3
  TRAIN_COMMAND=("$python_bin" -u scripts/train.py "$config" --exp-name="$run_name"
    --weight-loader.params-path="$initial_params" --project-name="$wandb_project"
    --checkpoint-base-dir="$output_root/checkpoints" --assets-base-dir="$output_root/assets"
    --seed=42 --fsdp-devices=2 --batch-size="$train_batch_size" --num-workers=4
    --num-train-steps="$target_steps" --lr-schedule.warmup-steps=500
    --lr-schedule.decay-steps="$target_steps" --lr-schedule.peak-lr="$peak_lr"
    --lr-schedule.decay-lr="$decay_lr" --eval-interval=1000 --eval-num-batches="$eval_num_batches"
    --save-interval="$save_interval" --keep-period="$save_interval" --wandb-enabled --no-wandb-log-images)
  if [[ $resume_arg == yes ]]; then TRAIN_COMMAND+=(--resume); fi
}

if [[ $mode == dry-run ]]; then
  echo "Pair: $pair_name; profile: $profile; GPUs: $CUDA_VISIBLE_DEVICES; FSDP devices: 2; global batch: $train_batch_size; trainable: all parameters; optimizer: $optimizer_label; target: 7D action + 6D wrist wrench; updates per model: $target_steps"
  for i in 0 1; do
    config=${configs[$i]}
    run_name=${pair_name}_${labels[$i]}
    audit_dir=$output_root/audits/$config
    printf 'Prepare %s: JAX_PLATFORMS=cpu %q scripts/prepare_tabero_smoke.py --config=%q --output-dir=%q --assets-base-dir=%q\n' "${labels[$i]}" "$python_bin" "$config" "$audit_dir" "$output_root/assets"
    build_train_command "$config" "$run_name" no
    printf 'Train %s: ' "${labels[$i]}"; printf '%q ' "${TRAIN_COMMAND[@]}"; printf '\n'
  done
  echo 'Order: next_state must exit 0 before sent_command starts.'
  exit 0
fi

pair_dir=$output_root/whiteboard_pairs/$pair_name
pair_log=$output_root/logs/${pair_name}_pair.log
status_file=$pair_dir/status.txt
if [[ $mode == new ]]; then
  if [[ -e $pair_dir || -e $pair_log ]]; then echo 'Pair name already exists; choose a new name or use --resume.' >&2; exit 2; fi
  for i in 0 1; do
    run_name=${pair_name}_${labels[$i]}
    run_dir=$output_root/checkpoints/${configs[$i]}/$run_name
    run_log=$output_root/logs/$run_name.log
    if [[ -e $run_dir || -e $run_log ]]; then echo "Run already exists: $run_name" >&2; exit 2; fi
  done
else
  if [[ ! -d $pair_dir ]]; then echo 'Resume requires the existing pair directory.' >&2; exit 2; fi
fi

mkdir -p "$pair_dir" "$output_root/logs" "$output_root/audits" "$output_root/wandb"
exec > >(tee -a "$pair_log") 2>&1
stage=starting
write_status() {
  local state=$1 exit_code=$2
  local tmp=$status_file.tmp
  printf 'state=%s\nstage=%s\nexit_code=%s\npair=%s\nupdated=%s\n' "$state" "$stage" "$exit_code" "$pair_name" "$(date --iso-8601=seconds)" > "$tmp"
  mv "$tmp" "$status_file"
}
on_exit() {
  local exit_code=$?
  if (( exit_code == 0 )); then write_status complete 0; else write_status failed "$exit_code"; fi
}
trap on_exit EXIT
write_status running 0

export HF_HOME=/data/yanghaojun/cache/huggingface
export OPENPI_DATA_HOME=/data/yanghaojun/cache/openpi
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.97}
export WANDB_MODE=online
export WANDB_DIR=$output_root/wandb
export WANDB_RUN_GROUP=$pair_name

if [[ $mode == new ]]; then
  for i in 0 1; do
    config=${configs[$i]}
    stage=prepare_${labels[$i]}
    write_status running 0
    audit_dir=$output_root/audits/$config
    echo "Preparing independent train-only statistics: $config"
    JAX_PLATFORMS=cpu "$python_bin" scripts/prepare_tabero_smoke.py --config="$config" --output-dir="$audit_dir" --assets-base-dir="$output_root/assets"
  done
fi

for i in 0 1; do
  config=${configs[$i]}
  dataset_name=${dataset_names[$i]}
  label=${labels[$i]}
  run_name=${pair_name}_${label}
  run_dir=$output_root/checkpoints/$config/$run_name
  run_log=$output_root/logs/$run_name.log
  stats_path=$output_root/assets/$config/local/$dataset_name/norm_stats.json
  provenance_path=$output_root/assets/$config/local/$dataset_name/split_provenance.json
  if [[ ! -f $stats_path || ! -f $provenance_path ]]; then
    echo "Independent normalization assets missing for $config" >&2
    exit 2
  fi
  read -r stats_hash _ < <(sha256sum "$stats_path")
  if [[ -d $run_dir/$target_steps/params ]]; then
    echo "Already complete, skipping: $run_name"
    continue
  fi
  resume_arg=no
  if [[ -d $run_dir ]]; then
    checkpoint_found=$(find "$run_dir" -mindepth 1 -maxdepth 1 -type d -regextype posix-extended -regex '.*/[0-9]+' -print -quit)
    if [[ -z $checkpoint_found || ! -f $run_dir/wandb_id.txt ]]; then
      echo "Cannot safely resume $run_name: numeric checkpoint or wandb_id.txt is missing." >&2
      exit 2
    fi
    resume_arg=yes
  elif [[ $mode == resume && -e $run_log ]]; then
    echo "Cannot start missing run $run_name because its log already exists." >&2
    exit 2
  fi
  stage=train_${label}
  write_status running 0
  echo "Starting $label only after all preceding stages succeeded."
  echo "Config: $config; dataset: $dataset_name; optimization: all parameters with $optimizer_label; FSDP: 2; target: 7D action + 6D wrist wrench; validation episodes: 1,7,17; train/val frames: 11125/1001"
  echo "Initialization: $initial_params (published Tabero 49999; independent optimizer)"
  echo "Normalization: $stats_path; SHA256: $stats_hash"
  echo "Checkpoints: $run_dir/{4000,8000,12000}; W&B group: $pair_name"
  JAX_PLATFORMS=cuda "$python_bin" -c 'import jax, sys; d=jax.devices(); print("JAX devices:", d); assert len(d)==int(sys.argv[1]) and all(x.platform=="gpu" for x in d), "GPU count/backend mismatch"' "$gpu_count"
  build_train_command "$config" "$run_name" "$resume_arg"
  JAX_PLATFORMS=cuda WANDB_JOB_TYPE=${label}_force_full_ft_${profile} "${TRAIN_COMMAND[@]}" 2>&1 | tee -a "$run_log"
  if [[ ! -d $run_dir/$target_steps/params ]]; then echo "Final checkpoint missing after successful command: $run_dir/$target_steps" >&2; exit 2; fi
  stage=completed_${label}
  write_status running 0
done

stage=all_complete
echo "Both full-parameter whiteboard models completed successfully: $pair_name"
