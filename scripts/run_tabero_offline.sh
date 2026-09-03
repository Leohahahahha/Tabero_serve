#!/usr/bin/env bash
# User-operated inference only; no server or robot connection.
set -euo pipefail
if [[ ${1:-} == --help ]] || (( $# < 1 )); then
  echo 'Usage: CUDA_VISIBLE_DEVICES=1 bash scripts/run_tabero_offline.sh RUN_NAME [EXTRA_EVALUATOR_FLAGS...]'
  echo 'Default: checkpoint 2999 of real_fr3_recovery_20260903_112020; all validation frames.'
  exit 0
fi
run_name="$1"
shift
if [[ ! "$run_name" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then echo 'Invalid run name.' >&2; exit 2; fi
if [[ ! ${CUDA_VISIBLE_DEVICES:-} =~ ^[0-9]+$ ]]; then
  echo 'Explicitly select exactly one permitted idle GPU index in CUDA_VISIBLE_DEVICES.' >&2
  exit 2
fi
cd /home/yanghaojun/Tabero-VTLA
export HF_HOME=/data/yanghaojun/cache/huggingface
export OPENPI_DATA_HOME=/data/yanghaojun/cache/openpi
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70
export PYTHONUNBUFFERED=1
checkpoint=/data/yanghaojun/outputs/checkpoints/pi0_lora_tacfield_local_tactile_lora_smoke/real_fr3_recovery_20260903_112020/2999
exec /data/yanghaojun/envs/tabero-smoke/bin/python -u scripts/eval_tabero_offline.py --checkpoint="$checkpoint" --output-dir="/data/yanghaojun/outputs/offline_eval/$run_name" "$@"
