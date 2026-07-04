# Tabero-VTLA

Tabero-VTLA 是一个基于 [Physical Intelligence 团队](https://www.physicalintelligence.company/) 开源的 openpi 框架进行深度定制的视觉-触觉-语言-动作（VTLA）模型训练与推理仓库。本仓库专注于**触觉/力觉信息融合的机器人操作策略学习**，基于 Tabero 触觉数据集进行微调。

目前，本仓库支持以下模型：
- [π₀ 模型](https://www.physicalintelligence.company/blog/pi0)，基于 flow matching 的视觉-语言-动作模型（VLA）
- [π₀.₅ 模型](https://www.physicalintelligence.company/blog/pi05)，π₀ 的升级版本，具有更好的开放世界泛化能力

本仓库的核心特色是**触觉/力觉模态融合**：在原始 VLA 模型基础上，新增了触觉力场（marker motion）、指力历史（gripper force）等触觉模态的支持，并支持在动作空间中显式建模力/力矩槽位的监督。

## 环境要求

运行本仓库的模型需要 NVIDIA GPU，具体要求如下（单 GPU 估算，也可通过 `fsdp_devices` 配置使用多 GPU 模型并行来降低单卡显存）：

| 模式         | 显存要求  | 推荐 GPU            |
| ------------ | --------- | ------------------- |
| 推理         | > 8 GB    | RTX 4090            |
| 微调（LoRA） | > 22.5 GB | RTX 4090            |
| 微调（全参） | > 70 GB   | A100 (80GB) / H100  |

本仓库在 Ubuntu 22.04 上测试通过。

## 安装

克隆本仓库时，请确保更新子模块：

```bash
git clone --recurse-submodules git@github.com:NathanWu7/Tabero-VTLA.git

# 如果已经克隆了仓库：
git submodule update --init --recursive
```

我们使用 [uv](https://docs.astral.sh/uv/) 管理 Python 依赖。请参考 [uv 安装说明](https://docs.astral.sh/uv/getting-started/installation/) 进行安装。安装完成后，运行以下命令设置环境：

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

注意：`GIT_LFS_SKIP_SMUDGE=1` 是为了将 LeRobot 作为依赖拉取时需要。

## 模型检查点

### 基础模型

我们提供多个基础 VLA 模型检查点，这些检查点已经在 10k+ 小时的机器人数据上预训练，可用于微调：

| 模型          | 用途       | 描述                                                  | 检查点路径                                      |
| ------------- | ---------- | ----------------------------------------------------- | ----------------------------------------------- |
| π₀            | 微调       | 基础 π₀ 模型                                          | `gs://openpi-assets/checkpoints/pi0_base`       |
| π₀.₅          | 微调       | 基础 π₀.₅ 模型                                        | `gs://openpi-assets/checkpoints/pi05_base`      |

检查点默认从 `gs://openpi-assets` 自动下载，缓存在 `~/.cache/openpi` 中。可通过设置 `OPENPI_DATA_HOME` 环境变量覆盖下载路径。

## 触觉/力觉融合概述

本仓库在原始 VLA 模型基础上引入了两种触觉/力觉信息融合路径：

- **作为额外 token 融入 Transformer**：支持 **encoder-prefix**（进 LLM 前缀）与 **decoder-suffix**（进 action expert 后缀）两条触觉流。
- **作为动作向量的一部分参与监督**：将动作向量的后若干维视为"力/触觉力矩槽位"，在 loss 中与关节动作拆分并加权。

详细说明请参考 [触觉集成文档](docs/tactile_integration.md)。

## 快速开始：在 Tabero 数据上微调 π₀

以 `pi0_lora_tacfield_tabero`（两路图像 + 触觉力场 + 13D 动作/力联合预测）为例，训练分为三步：

### 1. 准备数据

本仓库的训练配置直接使用 Hugging Face 上的 LeRobot 格式 Tabero 数据集（`NathanWu7/tabero`），无需手动转换数据。

### 2. 计算归一化统计量并启动训练

首先计算归一化统计量：

```bash
uv run scripts/compute_norm_stats.py --config-name pi0_lora_tacfield_tabero
```

然后启动训练：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_lora_tacfield_tabero --exp-name=my_experiment --overwrite
```

训练进度会输出到控制台，检查点保存在 `checkpoints` 目录下。你也可以通过 Weights & Biases 仪表盘监控训练进度。为最大化 GPU 显存利用率，建议设置 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`。

> **注意**：所有可用的训练配置可在 [src/openpi/training/config.py](src/openpi/training/config.py) 中查看，包括不同触觉模态组合的配置（tacimg、tacfield、tacforce、tacall 等）。

### 3. 启动策略服务器进行推理

训练完成后，可以通过启动策略服务器来进行推理（这里以迭代 50000 步的检查点为例）：

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_lora_tacfield_tabero --policy.dir=checkpoints/pi0_lora_tacfield_tabero/my_experiment/50000
```

这将启动一个 WebSocket 服务器，监听 8000 端口，等待传入观测数据进行推理。

也可以使用便捷脚本一键启动：

```bash
bash server.sh pi0_lora_tacfield_tabero 50000
```

## LIBERO 示例

我们提供了在 LIBERO 仿真基准上进行微调的示例：

```bash
# 计算归一化统计量
uv run scripts/compute_norm_stats.py --config-name pi05_libero

# 训练
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero --exp-name=my_libero_exp --overwrite
```

## PyTorch 支持

本仓库同时提供 π₀ 和 π₀.₅ 模型的 PyTorch 实现。PyTorch 实现已在 LIBERO 基准上验证通过（推理和微调）。

### 环境配置
1. 确保已安装最新依赖：`uv sync`
2. 确认 transformers 版本为 4.53.2：`uv pip show transformers`
3. 应用 transformers 库补丁：
   ```bash
   cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
   ```

### JAX 模型转 PyTorch

```bash
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir /path/to/jax/checkpoint \
    --config_name pi0_lora_tacfield_tabero \
    --output_path /path/to/converted/pytorch/checkpoint
```

### PyTorch 训练

```bash
# 单 GPU 训练：
uv run scripts/train_pytorch.py pi0_lora_tacfield_tabero --exp_name pytorch_test

# 多 GPU 训练（单节点）：
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_lora_tacfield_tabero --exp_name pytorch_ddp_test
```

## 常见问题

| 问题                                     | 解决方法                                                                                                                                                                                     |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `uv sync` 失败，依赖冲突                 | 尝试删除虚拟环境目录（`rm -rf .venv`）后重新运行 `uv sync`。如问题仍然存在，检查是否安装了最新版本的 `uv`（`uv self update`）。                                                       |
| 训练时 GPU 显存不足                      | 确保设置 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`（或更高）以允许 JAX 使用更多 GPU 显存。也可以使用 `--fsdp-devices <n>` 启用全分片数据并行。如仍不足，可考虑禁用 EMA。                 |
| 策略服务器连接错误                        | 检查服务器是否在预期端口运行。确认客户端和服务器之间的网络连接和防火墙设置。                                                                                                                   |
| 训练时报错找不到 norm stats               | 在开始训练前运行 `scripts/compute_norm_stats.py` 计算归一化统计量。                                                                                                                           |
| 数据集下载失败                            | 检查网络连接。对于 HuggingFace 数据集，确保已登录（`huggingface-cli login`）。                                                                                                               |
| CUDA/GPU 错误                            | 验证 NVIDIA 驱动是否正确安装。无需在系统级别安装 CUDA 库——它们会通过 uv 安装。如遇到 CUDA 问题，甚至可以尝试卸载系统 CUDA 库，因为系统库有时会引起冲突。                      |
| 动作维度不匹配                            | 验证数据处理变换是否与你机器人的预期输入/输出维度匹配。检查策略类中的动作空间定义。                                                                                                             |
| 训练 loss 发散                            | 检查数据集的 `norm_stats.json` 中的 `q01`、`q99` 和 `std` 值。某些很少使用的维度可能具有非常小的 `q01`、`q99` 或 `std` 值，导致归一化后的状态和动作值巨大。可以手动调整 norm stats 作为临时解决方案。 |

## 许可证

本项目基于 Apache 2.0 许可证开源。详见 [LICENSE](LICENSE) 文件。

## 致谢

本仓库基于 [Physical Intelligence](https://www.physicalintelligence.company/) 的 [openpi](https://github.com/Physical-Intelligence/openpi) 项目进行定制开发，感谢原作者的杰出工作。
