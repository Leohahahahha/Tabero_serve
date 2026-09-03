#!/usr/bin/env bash
# Run data preparation and CPU batch inspection; never launch training.
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "Usage: bash scripts/run_tabero_checks.sh [CONFIG]"
    echo "Default: pi0_lora_tacfield_local_tactile_lora_smoke"
    echo "Prepares normalization/audit, then checks CPU batches. No training."
    echo "Logs: /data/yanghaojun/outputs/<CONFIG>-audit/{prepare,batch}-<timestamp>.log"
    exit 0
fi
if (( $# > 1 )); then
    echo "Expected at most one configuration name. Use --help." >&2
    exit 2
fi

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
python_bin=/data/yanghaojun/envs/tabero-smoke/bin/python
config="${1:-pi0_lora_tacfield_local_tactile_lora_smoke}"
if [[ ! "$config" =~ ^[a-zA-Z0-9_]+$ ]]; then
    echo "Invalid configuration name: $config" >&2
    exit 2
fi
if [[ ! -x "$python_bin" ]]; then
    echo "Python environment not found: $python_bin" >&2
    exit 1
fi

export HF_HOME="${HF_HOME:-/data/yanghaojun/cache/huggingface}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/data/yanghaojun/cache/openpi}"
export JAX_PLATFORMS=cpu
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

output_dir="/data/yanghaojun/outputs/${config}-audit"
mkdir -p -- "$output_dir"
stamp="$(date +%Y%m%d-%H%M%S)-$$"
phase=prepare
trap 'rc=$?; echo "Failed during $phase (exit $rc). Logs: $output_dir" >&2; exit "$rc"' ERR

echo "Preparing $config on CPU. Logs: $output_dir"
"$python_bin" scripts/prepare_tabero_smoke.py --config="$config" --output-dir "$output_dir" 2>&1 | tee "$output_dir/prepare-$stamp.log"
phase=batch
echo "Preparation passed; checking train and validation batches on CPU."
"$python_bin" scripts/check_tabero_batch.py --config="$config" 2>&1 | tee "$output_dir/batch-$stamp.log"
echo "Both checks passed. No training was run."
