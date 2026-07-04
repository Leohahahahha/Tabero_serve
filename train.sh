#!/usr/bin/env bash
#set -e

# 切到工程根目录
#cd /path/to/Tabero-VTLA

###############################################################################
# 示例实验：pi0_lora_tacfield_tabero（两路图像 + 触觉力场 + 13D 动作/力联合预测）
###############################################################################

# 计算归一化统计（只需要跑一次）
uv run scripts/compute_norm_stats.py --config-name pi0_lora_tacfield_tabero

# 训练
uv run scripts/train.py pi0_lora_tacfield_tabero --exp-name=pi0_lora_tacfield_tabero_25 --overwrite

###############################################################################
# 其他可用配置（详见 src/openpi/training/config.py）：
#
#   pi0_lora_tacimg_tabero      - 三路图像 + 13D 动作（无 tactile token）
#   pi0_lora_tacfield_tabero    - 两路图像 + 触觉力场 + 13D 动作/力
#   pi0_lora_tacforce_tabero    - 两路图像 + 8×6 指力历史 + 13D 动作/力
#   pi0_lora_tacall_tabero      - 三路图像 + 双触觉通道 + 13D 动作/力
#   pi0_lora_notac_tabero       - 纯视觉基线（两路图像 + 7D 关节动作）
#   pi05_lora_tacfield_tabero   - Pi05 + 触觉力场
#   pi05_lora_tacimg_tabero     - Pi05 + 三路图像
#   pi05_lora_tacforce_tabero   - Pi05 + 指力历史
###############################################################################
