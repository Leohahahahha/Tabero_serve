#!/usr/bin/env bash
# User-operated recovery run. No robot code or raw-data writes.
set -euo pipefail
usage() {
  echo 'Usage: CUDA_VISIBLE_DEVICES=1,3 bash scripts/run_tabero_recovery.sh [--dry-run] RUN_NAME [INITIAL_PARAMS]'
  echo 'Select permitted idle GPU indices yourself. Default: 3000 updates, batch size = 2 x GPU count.'
}
if [[ ${1:-} == --help ]]; then usage; exit 0; fi
dry_run=false
if [[ ${1:-} == --dry-run ]]; then dry_run=true; shift; fi
if (( $# < 1 || $# > 2 )); then usage >&2; exit 2; fi
run_name="$1"
if [[ ! "$run_name" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
  echo 'RUN_NAME must contain only letters, digits, underscores and hyphens.' >&2
  exit 2
fi
if [[ ! ${CUDA_VISIBLE_DEVICES:-} =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo 'Set CUDA_VISIBLE_DEVICES to permitted GPU indices, e.g. 1 or 1,3 (no spaces).' >&2
  exit 2
fi
IFS=',' read -r -a gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
declare -A seen_gpus=()
for gpu_id in "${gpu_ids[@]}"; do
  gpu_id=$((10#$gpu_id))
  if [[ ${seen_gpus[$gpu_id]:-} == yes ]]; then
    echo 'Duplicate GPU indices are not allowed.' >&2
    exit 2
  fi
  seen_gpus[$gpu_id]=yes
done
gpu_count=${#gpu_ids[@]}
batch_size=$((2 * gpu_count))
# Held-out episodes (4, 14, 24) contain 693 frames; omit the incomplete batch.
eval_batches=$((693 / batch_size))
if (( eval_batches < 1 )); then echo 'Too many GPUs for this dataset.' >&2; exit 2; fi
cd /home/yanghaojun/Tabero-VTLA
export HF_HOME=/data/yanghaojun/cache/huggingface
export OPENPI_DATA_HOME=/data/yanghaojun/cache/openpi
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70
initial_params="${2:-/data/yanghaojun/outputs/checkpoints/pi0_lora_tacfield_local_tactile_lora_smoke/real_fr3_tactile_lora_r16_20260902_195449/99/params}"
config=pi0_lora_tacfield_local_tactile_lora_smoke
run_dir="/data/yanghaojun/outputs/checkpoints/$config/$run_name"
log_path="/data/yanghaojun/outputs/logs/$run_name.log"
python_bin=/data/yanghaojun/envs/tabero-smoke/bin/python
if [[ ! -d "$initial_params" ]]; then echo "Missing initial params: $initial_params" >&2; exit 2; fi
if [[ -e "$run_dir" || -e "$log_path" ]]; then echo 'Run name already exists; choose a fresh name.' >&2; exit 2; fi
train_command=("$python_bin" -u scripts/train.py "$config"
  --exp-name="$run_name" --weight-loader.params-path="$initial_params"
  --fsdp-devices=1 --batch-size="$batch_size" --num-workers=4 --num-train-steps=3000
  --lr-schedule.warmup-steps=100 --lr-schedule.decay-steps=3000
  --lr-schedule.peak-lr=1e-5 --lr-schedule.decay-lr=1e-6
  --eval-interval=250 --eval-num-batches="$eval_batches" --save-interval=100 --keep-period=500
  --no-wandb-enabled)
if "$dry_run"; then
  printf 'GPUs: %s; global batch: %s; validation batches: %s\n' "$CUDA_VISIBLE_DEVICES" "$batch_size" "$eval_batches"
  printf 'Command: '; printf '%q ' "${train_command[@]}"; printf '\n'
  exit 0
fi
mkdir -p /data/yanghaojun/outputs/logs
{
  echo "GPUs: $CUDA_VISIBLE_DEVICES; global batch: $batch_size; steps: 3000"
  echo 'Split: 26 training episodes / 3 validation episodes (4, 14, 24).'
  echo "Initialize weights: $initial_params (new optimizer and LR schedule, not --resume)"
  echo "Log: $log_path"
  echo "Metrics: $run_dir/metrics.jsonl"
  echo "View in another terminal: tail -n 50 -F '$log_path'"
  "$python_bin" -c 'import jax, sys; d = jax.devices(); print("JAX devices:", d); assert len(d) == int(sys.argv[1]) and all(x.platform == "gpu" for x in d), "GPU count/backend mismatch"' "$gpu_count"
  "${train_command[@]}"
} 2>&1 | tee "$log_path"
