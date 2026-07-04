#!/usr/bin/env bash
#set -e

# Switch to the project root
#cd /path/to/Tabero-VTLA

###############################################################################
# Example experiment: pi0_lora_tacfield_tabero (two image streams + tactile force field + 13D joint action/force prediction)
###############################################################################

# Compute normalization statistics (only needs to be run once)
uv run scripts/compute_norm_stats.py --config-name pi0_lora_tacfield_tabero

# Train
uv run scripts/train.py pi0_lora_tacfield_tabero --exp-name=pi0_lora_tacfield_tabero_25 --overwrite

###############################################################################
# Other available configs (see src/openpi/training/config.py):
#
#   pi0_lora_tacimg_tabero      - three image streams + 13D actions (no tactile token)
#   pi0_lora_tacfield_tabero    - two image streams + tactile force field + 13D joint action/force
#   pi0_lora_tacforce_tabero    - two image streams + 8x6 gripper force history + 13D joint action/force
#   pi0_lora_tacall_tabero      - three image streams + dual tactile streams + 13D joint action/force
#   pi0_lora_notac_tabero       - vision-only baseline (two image streams + 7D joint actions)
#   pi05_lora_tacfield_tabero   - Pi05 + tactile force field
#   pi05_lora_tacimg_tabero     - Pi05 + three image streams
#   pi05_lora_tacforce_tabero   - Pi05 + gripper force history
###############################################################################
