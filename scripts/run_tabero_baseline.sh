#!/usr/bin/env bash
# User-operated matched retraining; never calls a robot or starts offline inference.
set -euo pipefail
usage() {
  echo 'Usage: CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_tabero_baseline.sh [--dry-run] RUN_NAME [rgb_state|rgb_state_touch]'
  echo 'Default rgb_state. Published Tabero initialization, fixed global batch 4, 3000 updates.'
  echo 'Select 1, 2 or 4 permitted GPUs. Single-GPU batch-4 memory has not been measured.'
}
if [[ ${1:-} == --help ]]; then usage; exit 0; fi
dry_run=false
if [[ ${1:-} == --dry-run ]]; then dry_run=true; shift; fi
if (( $# < 1 || $# > 2 )); then usage >&2; exit 2; fi
run_name="$1"
variant="${2:-rgb_state}"
case "$variant" in rgb_state|rgb_state_touch) ;; *) usage >&2; exit 2;; esac
if [[ ! "$run_name" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then echo 'Invalid run name.' >&2; exit 2; fi
if [[ ! ${CUDA_VISIBLE_DEVICES:-} =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo 'Set CUDA_VISIBLE_DEVICES to permitted GPU indices, with no spaces.' >&2; exit 2
fi
IFS=',' read -r -a gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
declare -A seen_gpus=()
for gpu_id in "${gpu_ids[@]}"; do
  gpu_id=$((10#$gpu_id))
  if [[ ${seen_gpus[$gpu_id]:-} == yes ]]; then echo 'Duplicate GPU indices.' >&2; exit 2; fi
  seen_gpus[$gpu_id]=yes
done
gpu_count=${#gpu_ids[@]}
if (( gpu_count != 1 && gpu_count != 2 && gpu_count != 4 )); then
  echo 'Use 1, 2 or 4 GPUs so global batch stays fixed at 4; do not change batch for this comparison.' >&2; exit 2
fi
cd /home/yanghaojun/Tabero-VTLA
export HF_HOME=/data/yanghaojun/cache/huggingface
export OPENPI_DATA_HOME=/data/yanghaojun/cache/openpi
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70
config="pi0_lora_tabero_$variant"
initial_params=/data/yanghaojun/checkpoints/tabero-pretrained/checkpoints/pi0_lora_tacfield_tabero/pi0_lora_tacfield_tabero/49999/params
stats_path=/data/yanghaojun/outputs/checkpoints/pi0_lora_tacfield_local_tactile_lora_smoke/real_fr3_recovery_20260903_112020/2999/assets/local/tabero_lerobot_compact_v1/norm_stats.json
run_dir="/data/yanghaojun/outputs/checkpoints/$config/$run_name"
log_path="/data/yanghaojun/outputs/logs/$run_name.log"
python_bin=/data/yanghaojun/envs/tabero-smoke/bin/python
if [[ ! -d "$initial_params" || ! -f "$stats_path" ]]; then echo 'Published params or baseline statistics missing.' >&2; exit 2; fi
if [[ -e "$run_dir" || -e "$log_path" ]]; then echo 'Run name already exists; choose a new one.' >&2; exit 2; fi
read -r stats_hash _ < <(sha256sum "$stats_path")
if [[ "$stats_hash" != 7491dc0f0baa0765d15f62ad99f8c6499aca54e802ccb18991a888d1dae9acb7 ]]; then
  echo 'Normalization assets changed: review provenance before comparison.' >&2; exit 2
fi
train_command=("$python_bin" -u scripts/train.py "$config" --exp-name="$run_name"
  --weight-loader.params-path="$initial_params" --seed=42 --fsdp-devices=1
  --batch-size=4 --num-workers=4 --num-train-steps=3000
  --lr-schedule.warmup-steps=100 --lr-schedule.decay-steps=3000
  --lr-schedule.peak-lr=1e-5 --lr-schedule.decay-lr=1e-6
  --eval-interval=250 --eval-num-batches=173 --save-interval=100 --keep-period=500 --no-wandb-enabled)
if "$dry_run"; then
  printf 'Variant: %s; GPUs: %s; global batch: 4; steps: 3000\n' "$variant" "$CUDA_VISIBLE_DEVICES"
  printf 'Command: '; printf '%q ' "${train_command[@]}"; printf '\n'
  exit 0
fi
mkdir -p /data/yanghaojun/outputs/logs
{
  echo "Variant: $variant; GPUs: $CUDA_VISIBLE_DEVICES; fixed global batch: 4; steps: 3000; seed: 42"
  echo 'Split: 26 train / 3 held-out episodes (4,14,24). Labels unchanged; known startup/rotation risks remain.'
  echo "Initialize shared weights: $initial_params (new optimizer, not local smoke99, not resume)"
  echo "Normalization source: $stats_path; SHA256: $stats_hash"
  echo 'Old tactile recovery used local smoke99: comparison with that run is exploratory, not matched retraining.'
  echo "Log: $log_path"
  echo "Metrics: $run_dir/metrics.jsonl"
  echo "Final checkpoint on successful completion: $run_dir/2999"
  "$python_bin" -c 'import jax, sys; d = jax.devices(); print("JAX devices:", d); assert len(d) == int(sys.argv[1]) and all(x.platform == "gpu" for x in d), "GPU count/backend mismatch"' "$gpu_count"
  "${train_command[@]}"
} 2>&1 | tee "$log_path"
