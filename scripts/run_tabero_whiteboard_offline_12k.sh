#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_tabero_whiteboard_offline_12k.sh --gpu ID [--output-root DIR] [--targets NAME...]
  scripts/run_tabero_whiteboard_offline_12k.sh --check-only [--output-root DIR] [--targets NAME...]
  scripts/run_tabero_whiteboard_offline_12k.sh --dry-run [--targets NAME...]

Targets:
  ab_next_12k           rank-16 A/B experiment, next-state labels
  r32_next_12k          rank-32 experiment, next-state labels
  r32_sent_12k          rank-32 experiment, sent-command labels
  full_ft_sgd_next_12k  full-parameter SGD replacement, next-state labels

Default targets are all four available 12000-step checkpoints:
  r32_sent_12k r32_next_12k full_ft_sgd_next_12k ab_next_12k

Every target uses validation episodes 1, 7 and 17, stride 1, seed 42 and 10 denoise steps.
EOF
}

mode=run
gpu=
output_root=
targets=()
while (($#)); do
  case "$1" in
    --gpu)
      gpu=${2:?--gpu requires an ID}
      shift 2
      ;;
    --output-root)
      output_root=${2:?--output-root requires a directory}
      shift 2
      ;;
    --targets)
      shift
      while (($#)) && [[ $1 != --* ]]; do targets+=("$1"); shift; done
      ;;
    --check-only)
      mode=check
      shift
      ;;
    --dry-run)
      mode=dry
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ((${#targets[@]} == 0)); then
  targets=(r32_sent_12k r32_next_12k full_ft_sgd_next_12k ab_next_12k)
fi
if [[ $mode == run && -z $gpu ]]; then
  echo "Full inference requires --gpu ID." >&2
  exit 2
fi

python_bin=${TABERO_PYTHON:-/data/yanghaojun/envs/tabero-smoke/bin/python}
python_command=("$python_bin")
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-/data/yanghaojun/cache/openpi}
export HF_HOME=${HF_HOME:-/data/yanghaojun/cache/huggingface}

stamp=$(date +%Y%m%d_%H%M%S)
output_root=${output_root:-/data/yanghaojun/outputs/offline_eval/whiteboard_12k_${stamp}}

resolve_target() {
  case "$1" in
    ab_next_12k)
      config=pi0_lora_tabero_whiteboard_next_state_force_20k
      checkpoint=/data/yanghaojun/outputs/checkpoints/pi0_lora_tabero_whiteboard_next_state_force_20k/whiteboard_force_ab_20260915_next_state/12000
      ;;
    r32_next_12k)
      config=pi0_lora_tabero_whiteboard_next_state_force_tactile_r32_12k
      checkpoint=/data/yanghaojun/outputs/checkpoints/pi0_lora_tabero_whiteboard_next_state_force_tactile_r32_12k/whiteboard_force_r32_12k_20260916_next_state/12000
      ;;
    r32_sent_12k)
      config=pi0_lora_tabero_whiteboard_sent_command_force_tactile_r32_12k
      checkpoint=/data/yanghaojun/outputs/checkpoints/pi0_lora_tabero_whiteboard_sent_command_force_tactile_r32_12k/whiteboard_force_r32_12k_20260916_sent_command/12000
      ;;
    full_ft_sgd_next_12k)
      config=pi0_tabero_whiteboard_next_state_force_full_ft_sgd_12k
      checkpoint=/data/yanghaojun/outputs/checkpoints/pi0_tabero_whiteboard_next_state_force_full_ft_sgd_12k/full_ft_sgd_whiteboard_20260916_next_state/12000
      ;;
    *)
      echo "Unknown target: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
}

for target in "${targets[@]}"; do
  resolve_target "$target"
  if [[ $mode != dry && ! -d $checkpoint/params ]]; then
    echo "Missing finalized checkpoint for $target: $checkpoint" >&2
    exit 1
  fi
  command=(
    "${python_command[@]}" scripts/eval_tabero_whiteboard_offline.py
    --config "$config"
    --checkpoint "$checkpoint"
    --output-dir "$output_root/$target"
    --episodes 1 7 17
    --stride 1
    --num-denoise-steps 10
    --seed 42
  )
  if [[ $mode == check ]]; then command+=(--check-only --no-plots); fi
  printf 'Target: %s\n' "$target"
  printf '  %q' "${command[@]}"
  printf '\n'
  if [[ $mode == run ]]; then
    CUDA_VISIBLE_DEVICES=$gpu JAX_PLATFORMS=cuda "${command[@]}"
  elif [[ $mode == check ]]; then
    "${command[@]}"
  fi
done

if [[ $mode == run ]]; then
  "${python_command[@]}" scripts/summarize_tabero_whiteboard_offline.py "$output_root"
fi
printf 'Output root: %s\n' "$output_root"
