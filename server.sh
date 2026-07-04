#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# One-click script for starting the inference WebSocket policy server.
#
# This script:
# - Derives the Hugging Face repository and local clone path from the training config name
# - Pulls checkpoints from Hugging Face
# - Starts the WebSocket service with scripts/serve_policy.py
#
# Usage (from the project root):
#   1) Default step=49999, port 8000:
#        bash server.sh pi0_lora_tacfield_tabero
#   2) Specify step:
#        bash server.sh pi0_lora_tacfield_tabero 39999
#   3) Specify port:
#        PORT=9000 bash server.sh pi0_lora_tacfield_tabero 49999
#
# Optional environment variables:
#   - HF_OWNER: HF repository owner (default: NathanWu7)
#   - HF_BASE_DIR: local clone base directory (default: "${HOME}/hf")
#   - EXP_NAME: override experiment name (defaults to config name)
###############################################################################

usage() {
  local default_owner default_base
  default_owner="${HF_OWNER:-NathanWu7}"
  default_base="${HF_BASE_DIR:-$HOME/hf}"

  cat <<EOF
Usage:
  bash server.sh <config_name> [ckpt_step]

Examples:
  bash server.sh pi0_lora_tacfield_tabero 49999
  PORT=9000 bash server.sh pi0_lora_tacfield_tabero

Common configs (see src/openpi/training/config.py for more):
  - pi0_lora_tacimg_tabero
  - pi0_lora_tacfield_tabero
  - pi0_lora_tacforce_tabero
  - pi0_lora_tacall_tabero
  - pi05_lora_tacfield_tabero

Notes:
  - HF repository defaults to: https://huggingface.co/${default_owner}/<config_name>
  - Local clone directory defaults to: ${default_base}/<config_name>
  - Checkpoint directory defaults to: <repo_dir>/checkpoints/<config_name>/<exp_name>/<ckpt_step>
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "${#}" -lt 1 ]]; then
  usage
  exit 0
fi

########################
# Arguments / defaults
########################

CONFIG_NAME="${1}"
CKPT_STEP="${2:-49999}"
EXP_NAME="${EXP_NAME:-${CONFIG_NAME}}"
PORT="${PORT:-8000}"

HF_OWNER="${HF_OWNER:-NathanWu7}"
HF_BASE_DIR="${HF_BASE_DIR:-${HOME}/hf}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

HF_REPO_URL="https://huggingface.co/${HF_OWNER}/${CONFIG_NAME}"
HF_REPO_DIR="${HF_BASE_DIR}/${CONFIG_NAME}"

echo "[INFO] Current project root: ${ROOT_DIR}"
echo "[INFO] config: ${CONFIG_NAME}"
echo "[INFO] exp   : ${EXP_NAME}"
echo "[INFO] step  : ${CKPT_STEP}"
echo "[INFO] port  : ${PORT}"
echo "[INFO] HF repo: ${HF_REPO_URL}"
echo "[INFO] local : ${HF_REPO_DIR}"

########################
# Basic checks
########################

if ! command -v git >/dev/null 2>&1; then
  echo "[ERROR] git was not found. Please install git first." >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "[ERROR] uv (uvx) was not found. Please install uv and make sure it is in PATH." >&2
  echo "        Reference: \`https://github.com/astral-sh/uv\`" >&2
  exit 1
fi

if ! command -v git-lfs >/dev/null 2>&1; then
  echo "[WARN] git-lfs was not detected; pulling large files from HF may fail." >&2
  echo "       We recommend installing git-lfs and rerunning this script:" >&2
  echo "         sudo apt install git-lfs && git lfs install" >&2
fi

########################
# Prepare local HF repository
########################

mkdir -p "$(dirname "${HF_REPO_DIR}")"

if [[ ! -d "${HF_REPO_DIR}/.git" ]]; then
  echo "[INFO] ${HF_REPO_DIR} does not exist locally; cloning from HF"
  git lfs install --skip-repo >/dev/null 2>&1 || true
  git clone "${HF_REPO_URL}" "${HF_REPO_DIR}"
else
  echo "[INFO] Local repository exists; running git pull to sync latest changes"
  (
    cd "${HF_REPO_DIR}"
    git pull --ff-only || true
  )
fi

########################
# Build and check checkpoint path
########################

CKPT_DIR="${HF_REPO_DIR}/checkpoints/${CONFIG_NAME}/${EXP_NAME}/${CKPT_STEP}"

if [[ ! -d "${CKPT_DIR}" ]]; then
  echo "[ERROR] Checkpoint directory not found: ${CKPT_DIR}" >&2
  echo "        Possible reasons:" >&2
  echo "        - The step does not exist (for example, 49999 / 39999)" >&2
  echo "        - EXP_NAME does not match (override it with the EXP_NAME environment variable)" >&2
  echo "        You can list available steps with:" >&2
  echo "          ls \"${HF_REPO_DIR}/checkpoints/${CONFIG_NAME}/${EXP_NAME}\"" >&2
  exit 1
fi

echo "[INFO] Using checkpoint:"
echo "       ${CKPT_DIR}"
echo "[INFO] Starting WebSocket policy server..."

########################
# Start service
########################

# Workaround for `uv` installing deps (e.g. `lerobot`) from GitHub with git-lfs:
# some upstream LFS objects may be missing and break checkout.
# These artifacts are not needed for serving, so we skip LFS smudge by default.
: "${GIT_LFS_SKIP_SMUDGE:=1}"
export GIT_LFS_SKIP_SMUDGE

uv run scripts/serve_policy.py \
  --port "${PORT}" \
  policy:checkpoint \
  --policy.config="${CONFIG_NAME}" \
  --policy.dir="${CKPT_DIR}"
