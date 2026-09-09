# Tabero 仅动作预测 LoRA：面向真实 FR3 机器人的训练操作文档

## 当前状态与适用范围

本配置面向真实 Franka FR3 机器人，**不是 LIBERO 或其他仿真环境**。
虽然适配器位于沿用历史命名的 `libero_policy.py` 文件中，但新增的
`TaberoActionOnlyInputs` / `TaberoActionOnlyOutputs` 使用的是真实数据集的输入输出约定。

**截至本次准备工作的记录，尚未执行训练、权重下载、完整策略推理或机器人动作。**
用户已允许安装依赖和运行单元测试，但训练仍由用户手动启动。
安装与测试进度记录在 `progress.md` 中；这些准备检查通过，不代表训练已经成功。
下文命令均供用户手动执行。

2026-09-02 已验证：独立环境的依赖安装完成，版本为 Python 3.11.16、
JAX 0.5.3、Flax 0.10.2、PyTorch 2.7.1、LeRobot 0.3.3、PyAV 14.2.0。
选定的 CPU 测试中，**49 项通过，2 项需要下载 tokenizer 的测试未运行**，
其中包含 20 项触觉 LoRA 测试。两个本地配置的训练命令 `--help` 参数解析均通过。
测试覆盖小型 CPU 模块、合成的检查点参数树、梯度和编码器参数数量的抽象形状检查；
没有执行完整策略，也没有运行优化器更新或训练循环。
尚未执行真实数据集审计、归一化统计计算、预训练检查点恢复或 GPU 模型运行。
环境已安装，但训练流程尚未通过实际运行验证。

## 已实现的配置

`src/openpi/training/config.py` 中提供两个配置：

- `pi0_lora_tacfield_local_smoke`：保留不变的基线，冻结触觉编码器。
- `pi0_lora_tacfield_local_tactile_lora_smoke`：主干 LoRA **加上触觉 TCN LoRA**，
  用于新的传感器适配实验。下文手动命令默认选择此配置。

两个配置的共同约定及差异如下：

- 使用 JAX/Orbax，从 `NathanWu7/pi0_lora_tacfield_tabero` 的 `49999/params` 权重开始微调。
- 严格恢复全部已有预训练参数，包括主干已有的 LoRA 和触觉基础参数。
  仅允许新配置明确列出的触觉适配器参数路径在发布的检查点中不存在。
  预训练参数缺失、出现额外参数、形状不匹配，或新增适配器只存在一部分，都会报错。
- 冻结**所有非 LoRA 参数**。图像编码器、触觉基础权重与偏置、动作头和其他基础投影层均保持冻结。
  语言模型和动作专家的 LoRA 分支可以训练；新配置还会训练触觉 LoRA。
- 保持模型内部动作维度为 32、动作序列长度为 50，以兼容原有检查点。
  训练目标仅使用维度 `0:7`；力对应的位置和填充位置不接受直接监督。
  输出适配器只返回前 7 维。
- 输入为两路 RGB、机器人状态，以及由 marker motion 构成的触觉前缀。
  不使用触觉图像、深度或腕部力/力矩输入，也不进行力/力矩预测或计算对应损失。
- 验证轨迹为 **4、14、24**，共 693 帧；其余轨迹用于训练，共 5962 帧。
  不重写原始元数据或数据文件，也不另设测试集。
- 冒烟训练默认设置为 batch size 2、100 次优化器更新、学习率 1e-5、10 步预热、BF16 模型，
  不使用 EMA，不上传到 W&B。这些是**尚未经过实际训练验证的起始值**。
- 在首次参数更新前以及第 50、100 步进行确定性验证，使用 12 个 batch，
  即均匀分布于留出数据中的 24 帧。这是抽样验证，不是完整验证集评估。
  每次评估使用固定的流匹配噪声和时间采样随机种子。
- 本地 `metrics.jsonl` 记录训练与验证损失。日志还会记录可训练参数的路径、数量和严格恢复结果，
  便于核对初始化及冻结范围。

最后一个检查点目录沿用仓库从零开始编号的约定：
完成 100 次更新后生成目录 `99`，而 `TrainState.step` 和最终验证步数为 100。
这不代表少执行了一次优化器更新。

### 触觉传感器适配

新配置为 TCN 的全部 8 个线性映射添加 LoRA：6 个时序核、第一块的残差投影和输出投影。
设置为 rank=16、alpha=16、缩放系数 scale=alpha/rank=1；
**新增 779,008 个可训练参数**，同时保留 **65,239,040 个冻结的触觉基础参数**。
这些数量可通过 CPU 抽象形状测试核对，无需分配完整模型权重。

每个适配器的 A 随机初始化，B 初始化为零。恢复同一份基础检查点后，
适配器的初始贡献为零，不会重新初始化原有触觉编码器。
仅动作损失产生的梯度即可立即训练 B；当 B 变为非零后，A 也能获得梯度。
不引入力标签、辅助力损失，也不全量解冻触觉基础参数。
rank 和学习率只是实验默认值，并非已经验证的最优设置。

同为 `[9,198,2]` 不代表物理含义相同。解释实验结果前，必须核对采样点顺序与位置、
左右侧对应关系、坐标轴方向、shear 尺度与单位、传感器响应，以及参考帧和历史序列的构造方式。
归一化和 LoRA 不能保证自动修正这些差异。本次修改不自动执行坐标翻转、重采样、尺度标定或标签修正。
后续确认的标定处理必须一致地用于离线数据准备和在线输入；
预处理发生变化后，应重新仅使用训练集计算统计量。

与冻结触觉的基线比较时，应使用相同的 26/3 轨迹划分、预处理、随机种子和评估设置。
由于本地训练集统计量与发布者的统计量不同，包含归一化处理的初始策略可能不同于发布者的原始部署。
LoRA 初始输出为零，仅保证在**相同基础权重和输入**下，不额外引入适配器贡献。
训练损失降低或完成 100 步冒烟训练，都不能证明触觉有效或机器人任务成功。

## 真机输入输出约定

| 字段 | 约定 |
|---|---|
| `image` | 前视 RGB，裁剪与训练一致：原始 ZED 图像为 960x540，xyxy 裁剪范围为 `(350,0,740,520)`，输出 HWC `[520,390,3]` |
| `wrist_image` | D405 腕部 RGB 图像，HWC `[480,640,3]` |
| `state` | `[x,y,z,rx,ry,rz,finger_m]`，形状 `[7]`，记录当前观测到的绝对位姿；xyz 单位为米，旋转向量单位为弧度 |
| `tactile_marker_motion` | 浮点数组 `[9,198,2]`，按从旧到新排列；先左侧 99 点，再右侧 99 点；内容为参考网格加 shear，不是单独的原始 shear |
| `prompt` | 与训练集 `meta/tasks.jsonl` 的任务文本逐字一致；当前任务为黑色圆形部件对孔插入装配 |
| 返回的 `actions` | Float32 `[50,7]`，包含绝对目标位姿和米制单指位置；位姿必须使用**与训练目标相同的坐标系** |

使用 RGB，而不是 OpenCV 默认的 BGR。公共变换会将图像保持比例缩放并填充至 224x224。
推理时的图像通道和裁剪必须与训练一致。
机器人桥接程序不能跳过策略预处理、归一化或输出端的逆变换。

当前触觉编码器禁用了参考帧相减，因此完整使用全部 9 帧，不做参考差分。
本地导出的是滚动历史，不一定符合上游训练数据集的固定参考帧约定。
在线输入应使用相同的滚动构造方式；重置时清空历史，并重复首个有效帧进行补齐。
不得跨轨迹或跨重置拼接历史。其语义迁移效果仍需实验验证；形状兼容不代表触觉一定有效。

### 绝对动作与训练残差的区别

磁盘中的标签是绝对目标。为保持 Tabero 既有预处理方式，
训练时会从**动作序列中的每一个未来目标**减去当前观测状态的前 6 个分量。
夹爪位置不做差分。归一化统计量也在同样的变换之后计算。
策略输出端的逆变换顺序为：

1. 采样模型内部带填充维度的归一化动作序列。
2. 使用本次实验仅由训练集计算的统计量进行反归一化。
3. 对维度 `0:6` 加回本次推理所使用的状态。
4. 返回前 7 维，作为绝对目标。

旋转向量分量相减不是 SO(3) 旋转复合运算。不要将未经反归一化的输出直接发送给机器人，
不要重复加回状态，也不要把输出当成关节角。
如果控制器接收夹爪总开口宽度，桥接程序必须将模型输出的单指位置转换为总开口，
且**只转换一次**：`width = 2 * finger_m`。

### 真机侧仍需实现的安全措施

本次修改不添加 ROS 节点、机器人连接或执行器控制命令。
输出适配器会检查形状以及数值是否有限，但**它不是安全控制器，也不能证明模型已具备部署条件**。
执行任何真机动作之前，桥接程序还必须落实以下检查与保护：

- 确认位姿坐标系、工具/TCP 变换和控制器约定。
- 检查相机、状态和触觉历史的时间戳是否新鲜、是否同步。
- 限制位置工作空间，以及平移和 SO(3) 旋转的单步变化量与变化速率。
- 限制每指夹爪位置为 `[0,0.0425]` 米，并设置力、速度限制及碰撞保护。
- 设置明确的使能/持续按住使能机制、可随时触及的停止方式，以及输入缺失或过期时的看门狗保护。
- 对有限预测做显式边界投影并记录原值和触发项；NaN/Inf、实测状态或系统故障必须停止。

按数据集的 10 Hz 频率，50 个预测动作对应 5 秒，但当前同步部署每份观测只使用`action[0]`，
不会将整段动作直接开环执行。实际重规划频率由模型推理延迟决定，仍需通过现场 shadow 测量。

## 数据局限：目前不能认定为真机复现成功

交接记录指出，29 条轨迹中有 28 条在启动阶段存在较大的动作位置跳变，
并且旋转向量在 π 附近存在分支跳变。本任务尚未修正或重新审计这些问题。
审计脚本只进行测量，不裁剪数据、不展开旋转分支，也不修改标签。
应先厘清标签语义，再进行长时间训练或真机执行。

当前损失是仅针对动作维度的**流匹配速度均方误差（MSE）**，并不是直接的位姿 MSE、
SO(3) 误差或机器人成功率，也不能作为触觉提升性能的直接证据。
后续评估还需包括反归一化后的位置与夹爪误差、旋转测地距离误差、
仅 RGB/状态与加入触觉的消融比较，以及具备安全保护的真机试验。

## 手动操作命令

通过 SSH 操作时，建议使用已安装的 `tmux`，避免连接断开导致手动启动的任务退出。
先自行创建会话，再在会话内激活环境：

```bash
tmux new -s tabero-smoke
cd /home/yanghaojun/Tabero-VTLA
source /data/yanghaojun/envs/tabero-smoke/bin/activate
export HF_HOME=/data/yanghaojun/cache/huggingface
export OPENPI_DATA_HOME=/data/yanghaojun/cache/openpi
export PYTHONUNBUFFERED=1
TABERO_CONFIG=pi0_lora_tacfield_local_tactile_lora_smoke
# 如果要运行冻结触觉编码器的基线，改为选择：
# TABERO_CONFIG=pi0_lora_tacfield_local_smoke
```

先按 Ctrl-b，再按 d，即可分离会话；使用 `tmux attach -t tabero-smoke` 重新连接。
下文使用完整 Python 路径的命令，即使未激活环境也可以执行。
测试和审计时，应在对应命令前单独设置 `JAX_PLATFORMS=cpu`，
不要在整个 shell 中全局导出该变量，以免误将后续训练限制到 CPU。

### 1. 重装或复现依赖环境（已完成，通常可以跳过）

```bash
cd /home/yanghaojun/Tabero-VTLA
GIT_LFS_SKIP_SMUDGE=1 \
UV_PROJECT_ENVIRONMENT=/data/yanghaojun/envs/tabero-smoke \
UV_CACHE_DIR=/data/yanghaojun/cache/uv \
UV_PYTHON_INSTALL_DIR=/data/yanghaojun/envs/python \
uv sync --frozen --python 3.11
```

依赖包选择和过时的工作区锁文件已修复。
PyAV 固定为 14.2.0，以使用适用于 CPython 3.11/Linux 的预编译 wheel，
避免要求安装系统 FFmpeg 开发包。参见 [PyAV 发布文件](https://pypi.org/project/av/14.2.0/#files)。

### 2. 回归测试与只读数据审计

以下回归测试可以重复运行，不会训练模型。本次验证选用的完整测试集合为：

```bash
JAX_PLATFORMS=cpu HF_HUB_OFFLINE=1 WANDB_MODE=disabled \
/data/yanghaojun/envs/tabero-smoke/bin/python -m pytest \
  src/openpi/training/tabero_smoke_test.py src/openpi/transforms_test.py \
  src/openpi/shared/normalize_test.py src/openpi/models/lora_test.py \
  src/openpi/models/tactile_encoder_test.py \
  -k 'not tokenize' -q
```

```bash
JAX_PLATFORMS=cpu /data/yanghaojun/envs/tabero-smoke/bin/python -m pytest \
  src/openpi/training/tabero_smoke_test.py -q

JAX_PLATFORMS=cpu /data/yanghaojun/envs/tabero-smoke/bin/python \
  scripts/prepare_tabero_smoke.py \
  --config="$TABERO_CONFIG" \
  --output-dir "/data/yanghaojun/outputs/${TABERO_CONFIG}-audit"
```

审计会解码并统计全部视频帧，检查形状、有限值、索引和夹爪范围，
报告动作跳变与历史序列一致性，并且仅使用训练轨迹计算统计量。
不读取本版未使用的深度和腕部力/力矩列。
脚本将 `data_audit.json`、`norm_stats.json` 和 `split_provenance.json`
写入输出/资源目录，不写入原始数据集。
继续操作前先检查这些文件；修改数据划分或配置时，需要重新计算相应统计量。
资源目录为 `/data/yanghaojun/outputs/assets/<config>/local/tabero_lerobot_compact_v1`。
即使之前已为基线运行过准备脚本，也应使用**新配置**再次运行，以生成新配置独立目录中的资源。
程序不会自动读取基线的资源目录。相同的预处理和数据划分应产生相同的统计量。

统计量生成后，在 CPU 上分别检查训练集和验证集的一个 batch，不创建模型：

```bash
/data/yanghaojun/envs/tabero-smoke/bin/python scripts/check_tabero_batch.py --config="$TABERO_CONFIG"
```

预期结果：动作张量为 `[2,50,32]`，其中位置 `7:` 全为零；触觉前缀为 `[2,9,396]`；
状态为 `[2,32]`；图像张量为 `[2,224,224,3]`；没有触觉后缀。
第三路图像是带掩码的占位输入，并不是另一台触觉相机。
检查数值范围，注意是否出现极端的归一化数值。
只有动作的前 7 维参与监督；部署时经过逆变换后返回的动作序列为 `[50,7]`。

### 3. 下载 Tabero 预训练参数，而不是 pi0_base

```bash
HF_HOME=/data/yanghaojun/cache/huggingface \
/data/yanghaojun/envs/tabero-smoke/bin/python scripts/download_tabero_checkpoint.py \
  --output-dir /data/yanghaojun/checkpoints/tabero-pretrained
```

下载脚本会解析并记录一个不可变的 Hugging Face 版本，
只下载模型参数和原始资源，跳过旧优化器状态。
这是使用新优化器、从已有权重开始的微调，不是通过 `--resume` 恢复发布者的原始实验。

### 4. 单 GPU 冒烟训练（由用户手动启动；助手执行需另获明确授权）

先重新运行 `nvidia-smi`，选择一张获准使用的显卡。
本次准备工作记录的**最近一次检查**中，4 张卡都有任务：GPU 0/1 各剩余约 26 GiB，
GPU 2/3 的显存几乎占满。不要终止无关任务，也不要假定显卡可用性一直不变。
首次运行优先选择空闲卡；如果必须共享显卡，应降低显存分配比例，并事先确定可使用的资源额度。
显存分配比例针对的是显卡总显存，不是当前剩余显存。

```bash
nvidia-smi
read -r -p "输入已确认允许使用且空闲的 GPU 编号: " TABERO_GPU
: "${TABERO_GPU:?必须先选择可用 GPU}"
TABERO_RUN=real_fr3_tactile_lora_r16_smoke_001
mkdir -p /data/yanghaojun/outputs/logs
set -o pipefail

CUDA_VISIBLE_DEVICES="$TABERO_GPU" JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_MEM_FRACTION=0.70 \
OPENPI_DATA_HOME=/data/yanghaojun/cache/openpi \
HF_HOME=/data/yanghaojun/cache/huggingface \
/data/yanghaojun/envs/tabero-smoke/bin/python -u scripts/train.py \
  "$TABERO_CONFIG" --exp-name="$TABERO_RUN" \
  --batch-size=2 --num-train-steps=100 --no-wandb-enabled \
  2>&1 | tee "/data/yanghaojun/outputs/logs/${TABERO_RUN}.log"
```

使用新的实验名称，不需要添加 `--overwrite`。首次运行时可能会下载 tokenizer。
在重新评估显存限制之前，不要在繁忙的共享显卡上启动任务。
首次获准运行时，应先检查 batch 和权重恢复日志，再判断损失是否合理。
日志中应出现 `Strictly restored ... (... LoRA)` 以及 `Trainable parameters` / `Trainable paths`。
新配置首次从发布权重开始微调时，还应出现
`Initialize 16 explicitly allowed new adapter leaves`，以及
`Trainable tactile LoRA: 779008 parameters in 16 leaves`。
触觉编码器的基础 `kernel` 和 `bias` 不应出现在可训练参数路径中。
恢复完整检查点或续训时，已有触觉适配器应被保留，不能重置为零。
严格恢复报错意味着需要停止并排查检查点与配置不匹配的问题；
不要为了让命令运行而关闭严格加载，也不要改回 pi0_base。

使用上述实验名称时，输出文件位于：

```text
/data/yanghaojun/outputs/checkpoints/pi0_lora_tacfield_local_tactile_lora_smoke/real_fr3_tactile_lora_r16_smoke_001/
  metrics.jsonl
  99/params/
  99/assets/
  99/train_state/
```

上述命令还会将终端输出保存到 `/data/yanghaojun/outputs/logs/<run>.log`。
只有在训练创建 `metrics.jsonl` 后，才能使用 `tail -f` 持续查看它。
如果显存不足，使用新的实验名称并设置 `--batch-size=1`；
不要静默切换到 CPU，也不要终止其他用户的 GPU 任务。
检查审计结果中的标签不连续警告之前，不要延长为长时间训练。

### 5. 连接机器人之前，先进行离线推理

使用 `create_trained_policy`，传入与训练相同的本地冒烟配置，以及新检查点自身的 `assets`，
不要使用发布者的旧归一化统计量。先用一条已录制的观测调用 `policy.infer`，
检查输出是否为 `[50,7]` 的有限绝对目标，并验证归一化与反归一化的往返一致性。
本次修改不包含服务端或机器人客户端的启动操作。

新配置推理时必须使用
`config.get_config("pi0_lora_tacfield_local_tactile_lora_smoke")`，
并且 rank、alpha 与训练保持一致。**不要**使用冻结触觉的基线配置重建该检查点：
基线架构没有触觉适配器，因此不会应用训练后的触觉 LoRA。
正常保存的完整检查点同时包含基础权重和适配器；
此 JAX 推理流程不需要手动合并或导出 LoRA，也不需要重新下载基础模型。

## 2026-09-03：NaN 后的手动重启与 GPU 数量变化

本节命令由用户执行。助手没有启动新的 GPU 训练。此前 3000-step 实验在约 260 步出现 NaN，
没有到达首次保存点；不要对那个空实验目录使用 `--resume`。
新增逐步有限值检查：无效更新不提交，保存故障 batch、诊断 JSON 和故障前的正常状态后退出。
这属于故障防护与定位，**尚未证实 NaN 根因已修复**。

启动脚本为 `scripts/run_tabero_recovery.sh`，默认从已存在的
`real_fr3_tactile_lora_r16_20260902_195449/99/params` 载入权重，重新建立优化器和学习率计划。
这是继续已有 Tabero + 触觉 LoRA 权重，而不是从 pi0 基础权重训练，也不是精确续跑旧优化器。
可用第二个位置参数指定另一份已确认正常的 `params` 路径。

终端存在续行崩溃问题，因此下列命令应**逐行单独粘贴、回车执行**，不要整块多行粘贴。
建议先进入 `tmux new-session -s tabero-recovery` 创建的会话。

```bash
cd /home/yanghaojun/Tabero-VTLA
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv
read -r -p "请输入确认可用的 GPU 编号（如 1 或 1,3，不带空格）: " TABERO_GPUS
TABERO_RUN="real_fr3_recovery_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES="$TABERO_GPUS" bash scripts/run_tabero_recovery.sh "$TABERO_RUN"
```

脚本自己指定 Python 环境和缓存，不依赖终端中的旧 `TABERO_CONFIG` / `TABERO_INIT_PARAMS`。
只选择有权使用、显存足够的空闲 GPU；不会自动抢卡或停止其他进程。
全局 batch 为卡数的两倍：1/2/3/4 卡对应 2/4/6/8，均设置 3000 次更新、FSDP=1（数据并行）、
峰值学习率 1e-5、warmup 100、每 250 步验证、约每 100 步保存。
保留最近检查点及 500 步周期检查点，正常完成的末次目录为 `2999`。
卡数改变会改变全局 batch 和总样本暴露量，不是完全相同的训练实验。
仍为 **26 条训练、3 条验证（4、14、24）**，并非全部 29 条参与梯度更新。
验证覆盖 693 帧中的 692/692/690/688 帧，取决于卡数（丢弃不完整 batch）。

仅检查命令、不启动模型或写输出，可在启动前执行：

```bash
CUDA_VISIBLE_DEVICES="$TABERO_GPUS" bash scripts/run_tabero_recovery.sh --dry-run "$TABERO_RUN"
```

启动时会打印完整 `Log:`、`Metrics:` 路径以及可直接复制到另一个终端的 `tail` 命令。
也可以在另一个终端逐行执行；该终端不会自动继承原终端的实验名变量：

```bash
read -r -p "请输入本次实验名（日志文件名去掉 .log）: " TABERO_RUN
tail -n 50 -F "/data/yanghaojun/outputs/logs/${TABERO_RUN}.log"
```

在日志查看终端按 Ctrl+C 只结束查看。查看验证记录：

```bash
rg 'Validation step|Rejected non-finite|FloatingPointError|Traceback' "/data/yanghaojun/outputs/logs/${TABERO_RUN}.log"
```

`metrics.jsonl` 位于 `/data/yanghaojun/outputs/checkpoints/pi0_lora_tacfield_local_tactile_lora_smoke/<实验名>/`。
若再次数值失败，同目录还会生成 `numerical_failure.json` 和 `failed_batch.npz`；请保留它们用于定位，
不要不断重启或把 NaN 检查关闭。Loss 下降不等于已经可以安全部署真机。

## 2026-09-03：训练后的离线动作评估

入口：`scripts/eval_tabero_offline.py`；便捷启动：`scripts/run_tabero_offline.sh`。
只加载本地检查点并读取录制数据，不连接机器人、不启动服务、不训练、不改数据或权重。
默认使用已完成的 `real_fr3_recovery_20260903_112020/2999`，匹配 rank-16 触觉 LoRA 配置，
使用该检查点自己的 `assets`。严格恢复会拒绝多余、缺少或形状不匹配的参数，防止触觉适配器被忽略。

### 手动启动

下面仍须**逐行单独粘贴执行**。本脚本仅使用一张 GPU；请先确认空闲和使用权限。
脚本会指定 Python 环境和缓存，不必激活环境，也不必再次下载权重。

```bash
cd /home/yanghaojun/Tabero-VTLA
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv
read -r -p "请输入一张可用 GPU 的编号: " TABERO_EVAL_GPU
TABERO_EVAL="offline_2999_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES="$TABERO_EVAL_GPU" bash scripts/run_tabero_offline.sh "$TABERO_EVAL"
```

默认：验证 episode 4、14、24 的全部 **693 个观测帧**（不丢尾 batch），每帧预测 `[50,7]`，
采样去噪 10 步。第一帧包含模型编译开销，可能比后续明显慢，不要仅因短暂无进度就终止。
每 25 个观测打印一次进度。总耗时与所选 GPU/负载有关，尚未实测完整模型推理。
若仅想限制检查范围，使用新实验名并在启动命令末尾加 `--max-frames-per-episode=20`，
这会在每条轨迹均匀选择至多 20 帧；抽样结果不能冒充全验证集结果。
`--episodes 4 14 24` 可选择验证子集，训练轨迹编号会被拒绝。
`--stride=2` 则在各轨迹每隔一帧推理。所有比较必须保持选择规则、seed、去噪步数一致。

默认输出：`/data/yanghaojun/outputs/offline_eval/<TABERO_EVAL>/`。已有目录会被拒绝，不自动覆盖。
完整解释器入口也支持 `--checkpoint=/绝对路径/检查点目录`、`--output-dir=/新目录`，
便捷脚本末尾的这两个参数可覆盖默认值。

### 输出与查看

- `eval.log`：进度、最终首动作指标及保持当前状态基线。
- `summary.json`：总体、逐 episode、逐预测偏移量的误差（mean/RMSE/p50/p95/max）和异常统计。
- `predictions.csv`：每个观测/预测偏移的 7D 预测、标签、误差与有效标记。`valid=0` 的补齐行误差留空。
- `predictions.npz`：未裁剪的预测、标签、状态、有效掩码、episode/frame 和同步后的推理时间；无需 pickle。
- `episode_000004.png`、`episode_000014.png`、`episode_000024.png`：首动作与记录目标/状态的对比图。
- `manifest.json`：检查点、归一化文件哈希、源码哈希、选择范围、随机种子和完成/失败状态。

在另一个终端先设置同一个 `TABERO_EVAL` 实验名，再查看日志：

```bash
read -r -p "请输入本次离线评估实验名: " TABERO_EVAL
tail -n 30 -F "/data/yanghaojun/outputs/offline_eval/${TABERO_EVAL}/eval.log"
```

`manifest.json` 的 `status=complete` 且存在 `summary.json` 表示评估成功完成；
`checks_passed_no_inference` 只表示 CPU 预检查完成，不包含模型结果。
遇到非有限值或错位立即停止，不替换坏值、不跳过样本；`failure.json` 定位故障 episode/frame，
之前完成的预测留在 CSV。模型加载等更早阶段的错误记录在 `manifest.json` 和日志。

### 指标如何理解

`overall.first_action` 是每次观测下预测的第一个动作与同帧记录目标的误差；
`overall.valid_chunk` 是整段未来预测去除 episode 末尾 padding 后的误差。
位置是 xyz 欧氏距离（毫米）；旋转是 SO(3) 最短旋转角（度），不是轴角分量相减；
夹爪是单指绝对位置差（毫米），不是总开口差。
`hold_current_state_baseline` 将当前状态重复到各个未来偏移，是判断“仅保持不动”能有多小误差的参照。
该基线不需要模型，也不是另一份训练结果。

同一目标可能在多个重叠预测窗口出现，因此整段指标不是独立样本统计或任务成功率。
首动作曲线是使用每帧真实录制观测产生的独立预测，不是执行模型动作后的轨迹。
本脚本不证明触觉贡献；触觉消融、不同 checkpoint 的配对比较和真机闭环评估是后续工作。
原始 action 启动突变没有裁掉，报告中的真实标签跳变应结合采集背景判断。

推理时间使用整个 `policy.infer()` 调用的墙钟时间，包含预/后处理和 GPU 同步；
第一帧编译时间单列，后续帧报告分位数。视频解码、网络、控制器和真实传感器延迟不在其中。
夹爪越界、相邻预测位置变化超过 50 mm 仅为诊断项，不能代替工作空间、速度、碰撞和急停检查。
**没有任何误差阈值会在此自动批准真机执行。**
