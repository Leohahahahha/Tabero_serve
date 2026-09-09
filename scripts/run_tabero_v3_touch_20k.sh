#!/usr/bin/env bash
# User-operated 20k tactile-LoRA training. Does not contact a robot.
set -euo pipefail

usage() {
  echo 'Usage: CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_tabero_v3_touch_20k.sh [--dry-run|--resume] RUN_NAME'
  echo 'Uses 1, 2, or 4 selected GPUs, fixed global batch 4, published Tabero initialization, and W&B online logging.'
}

if [[ ${1:-} == --help ]]; then usage; exit 0; fi
mode=new
if [[ ${1:-} == --dry-run ]]; then mode=dry-run; shift; fi
if [[ ${1:-} == --resume ]]; then mode=resume; shift; fi
if (( $# != 1 )); then usage >&2; exit 2; fi
run_name=$1
if [[ ! $run_name =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then echo 'Invalid run name.' >&2; exit 2; fi
if [[ ! ${CUDA_VISIBLE_DEVICES:-} =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo 'Set CUDA_VISIBLE_DEVICES to permitted GPU indices, with no spaces.' >&2
  exit 2
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
  echo 'Use 1, 2, or 4 GPUs; the fixed global batch 4 must divide evenly.' >&2
  exit 2
fi

cd /home/yanghaojun/Tabero-VTLA
config=pi0_lora_tabero_v3_touch_20k
python_bin=${TABERO_PYTHON:-/data/yanghaojun/envs/tabero-smoke/bin/python}
initial_params=${TABERO_INITIAL_PARAMS:-/data/yanghaojun/checkpoints/tabero-pretrained/checkpoints/pi0_lora_tacfield_tabero/pi0_lora_tacfield_tabero/49999/params}
stats_path=${TABERO_STATS_PATH:-/data/yanghaojun/outputs/assets/$config/local/tabero_lerobot_compact_v3/norm_stats.json}
provenance_path=${TABERO_PROVENANCE_PATH:-/data/yanghaojun/outputs/assets/$config/local/tabero_lerobot_compact_v3/split_provenance.json}
expected_stats_hash=${TABERO_EXPECTED_STATS_SHA256:-3179b7ee7553cac64f3ac8a40897ed620b78b2ee354f9a297534f88cc66a0aed}
output_root=${TABERO_OUTPUT_ROOT:-/data/yanghaojun/outputs}
run_dir=$output_root/checkpoints/$config/$run_name
log_path=$output_root/logs/$run_name.log
wandb_project=${TABERO_WANDB_PROJECT:-tabero-vtla}

if [[ ! -x $python_bin ]]; then echo "Python environment missing: $python_bin" >&2; exit 2; fi
if [[ ! -d $initial_params ]]; then echo "Published Tabero params missing: $initial_params" >&2; exit 2; fi
if [[ ! -f $stats_path || ! -f $provenance_path ]]; then
  echo 'v3 train-only normalization assets are missing; run prepare_tabero_smoke.py first.' >&2
  exit 2
fi
read -r stats_hash _ < <(sha256sum "$stats_path")
if [[ $stats_hash != "$expected_stats_hash" ]]; then
  echo "Unexpected v3 normalization SHA256: $stats_hash" >&2
  exit 2
fi
if [[ $mode == resume ]]; then
  if [[ ! -d $run_dir || ! -f $run_dir/wandb_id.txt ]]; then
    echo 'Resume requires an existing checkpoint run directory and wandb_id.txt.' >&2
    exit 2
  fi
  checkpoint_found=$(find "$run_dir" -mindepth 1 -maxdepth 1 -type d -regextype posix-extended -regex '.*/[0-9]+' -print -quit)
  if [[ -z $checkpoint_found ]]; then echo 'Resume requires at least one numeric checkpoint directory.' >&2; exit 2; fi
else
  if [[ -e $run_dir || -e $log_path ]]; then echo 'Run name already exists; choose a new name.' >&2; exit 2; fi
fi

export HF_HOME=/data/yanghaojun/cache/huggingface
export OPENPI_DATA_HOME=/data/yanghaojun/cache/openpi
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.70}
export WANDB_MODE=online
export WANDB_DIR=$output_root/wandb

wandb_image_arg=--no-wandb-log-images
if [[ ${TABERO_WANDB_LOG_IMAGES:-0} == 1 ]]; then wandb_image_arg=--wandb-log-images; fi
train_command=("$python_bin" -u scripts/train.py "$config" --exp-name="$run_name"
  --weight-loader.params-path="$initial_params" --project-name="$wandb_project"
  --seed=42 --fsdp-devices=1 --batch-size=4 --num-workers=4
  --num-train-steps=20000 --lr-schedule.warmup-steps=500
  --lr-schedule.decay-steps=20000 --lr-schedule.peak-lr=1e-5
  --lr-schedule.decay-lr=1e-6 --eval-interval=1000 --eval-num-batches=172
  --save-interval=4000 --keep-period=4000 --wandb-enabled "$wandb_image_arg")
if [[ $mode == resume ]]; then train_command+=(--resume); fi

if [[ $mode == dry-run ]]; then
  printf 'Config: %s; GPUs: %s; global batch: 4; updates: 20000; W&B project: %s\n' "$config" "$CUDA_VISIBLE_DEVICES" "$wandb_project"
  printf 'Initialization: %s\nNormalization: %s (SHA256 %s)\n' "$initial_params" "$stats_path" "$stats_hash"
  printf 'Command: '; printf '%q ' "${train_command[@]}"; printf '\n'
  exit 0
fi

mkdir -p "$output_root/logs" "$WANDB_DIR"
{
  echo "Config: $config; GPUs: $CUDA_VISIBLE_DEVICES; global batch: 4; updates: 20000; seed: 42"
  echo 'Split: 26 training episodes / validation episodes 4,14,24; v3 next-state action labels.'
  echo 'Action transform: relative XYZ + SO(3) relative rotvec + absolute gripper; output is restored to absolute pose.'
  echo 'Known data/runtime mismatch: 18 compacted source intervals; expert translations are not clipped to the 0.02 m/s deployment guard.'
  echo "Initialization: $initial_params (published Tabero 49999 only; new optimizer)"
  echo "Normalization: $stats_path; SHA256: $stats_hash"
  echo "W&B: online project=$wandb_project; camera upload=$([[ $wandb_image_arg == --wandb-log-images ]] && echo enabled || echo disabled)"
  echo "Checkpoints: $run_dir/{4000,8000,12000,16000,20000}"
  echo "Local log: $log_path"
  "$python_bin" -c 'import jax, sys, wandb; d=jax.devices(); print("JAX devices:", d); print("wandb:", wandb.__version__); assert len(d)==int(sys.argv[1]) and all(x.platform=="gpu" for x in d), "GPU count/backend mismatch"' "$gpu_count"
  "${train_command[@]}"
} 2>&1 | tee -a "$log_path"
