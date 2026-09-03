# 独立训练 RGB＋state 基线，与触觉模型比较

本次只准备代码和 CPU 测试；训练、加载权重推理均由用户执行，不连接机器人。

## 实验定义

| 项目 | 无触觉组 | 可选的匹配触觉组 |
| --- | --- | --- |
| 配置 | `pi0_lora_tabero_rgb_state` | `pi0_lora_tabero_rgb_state_touch` |
| 输入 | 前视 RGB、腕部 RGB、state、原任务文本 | 相同输入＋marker history |
| 触觉分支 | 不创建编码器，不生成触觉 token，不向模型传入任何触觉字段 | 原 TCN＋rank-16 触觉 LoRA |
| 初始化 | 原始已下载 Tabero `49999/params` 的共享参数 | 同一份原始 Tabero 参数；新触觉 LoRA 零输出初始化 |
| 训练 | 26 条轨迹，5962 帧；3000 更新、全局 batch 4、seed 42 | 相同 |
| 验证 | episode 4、14、24，不参与训练 | 相同 |
| 优化 | backbone/action-expert LoRA；其余参数冻结 | 相同共享 LoRA＋触觉 LoRA |
| 学习率 | warmup 100，peak 1e-5，decay 3000 至 1e-6 | 相同 |
| 动作 | 保持绝对 xyz＋axis-angle＋单指米制位置，7D 监督、50 步 chunk、32D 内部宽度 | 相同 |

无触觉组不是把原始坐标清零，也不是只在推理时屏蔽触觉。训练和推理都没有触觉分支。仍然以 **Tabero 而不是 pi0_base** 为共同起点；因此它并非“从未接受触觉预训练的模型”。

原始 Tabero checkpoint 多出来的 16 个触觉编码器 kernel/bias 叶子只在无触觉组初始化时按明确白名单忽略；共享参数缺失、形状错误、未知额外参数、误用带本地触觉 LoRA 的 checkpoint 都会被严格检查。训练后评估仍使用严格全量恢复，不会默默删掉模型参数。

### 与已经完成的触觉模型有何差别

现有 `real_fr3_recovery_20260903_112020/2999` 是从本地 **已训练 100 步的触觉 smoke99** 再训练 3000 步得到的。因此：

- 先只训练新的无触觉模型，与现有触觉评估比较：可以做，但属于**探索性比较**，初始化历史和训练预算不完全一致。
- 想做更严格的训练对照：还需手动训练可选的匹配触觉组，两组从同一个原始 Tabero checkpoint 开始，同 batch、步数、seed、优化器、评估设置。优先两组使用相同 GPU 数量。
- 不从现有 `2999` 或 `99` 初始化无触觉模型，否则共享权重已经接受过本地触觉条件下的训练。
- 单次 seed、三条验证轨迹不足以证明统计显著性；两组可训练参数也相差触觉 LoRA 部分。本实验比较的是完整训练方案，不是严格等参数量的信息消融。

本轮不裁剪开头、不修改旋转表示、不改动作标签。已有启动跳变与旋转分支风险仍在；若以后处理，两组必须同时采用同版本数据/变换并重新训练。结果不能作为直接真机执行许可。

## 1. 启动无触觉训练

工作站交互式 Bash 存在反斜杠续行崩溃问题。**下面每个代码块只有一行，逐条执行，不要一次粘贴多条命令，不要添加反斜杠续行。**

```bash
cd /home/yanghaojun/Tabero-VTLA
```

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv
```

```bash
read -r -p "输入允许使用的空闲 GPU 编号（例如 0,1）: " TABERO_BASE_GPUS
```

支持 1、2、4 张卡；不支持 3 张卡，因为全局 batch 固定为 4。优先两张空闲 A6000。单卡 batch 4 的显存未实测；不要为使用更多 GPU 擅自增大全局 batch。GPU 查询不等于独占预约。

```bash
TABERO_BASE_RUN="rgb_state_$(date +%Y%m%d_%H%M%S)"
```

先检查命令（不会创建训练目录、加载模型或启动 GPU）：

```bash
CUDA_VISIBLE_DEVICES="${TABERO_BASE_GPUS:?请先选择GPU}" bash scripts/run_tabero_baseline.sh --dry-run "${TABERO_BASE_RUN:?}"
```

确认后，以下命令才会真正训练：

```bash
CUDA_VISIBLE_DEVICES="${TABERO_BASE_GPUS:?请先选择GPU}" bash scripts/run_tabero_baseline.sh "${TABERO_BASE_RUN:?}"
```

脚本使用 `/data/yanghaojun/envs/tabero-smoke/bin/python`，不依赖当前已激活环境；固定原始 Tabero 权重，不覆盖旧实验。已有数值失败保护继续生效。正常完成后的最终 checkpoint 是循环索引 `2999`（3000 次更新）。

**无需重新下载权重或运行数据 preparation。** 两组复用已完成 checkpoint 的训练集归一化资产，启动脚本校验源文件 SHA256；无触觉组仅使用并保存 state/actions 两项。复用的是统计，不是 `2999` 的学习权重。若统计缺失或 hash 不符，先调查，不要绕过校验。

## 2. 查看训练日志

同一个终端变量仍有效时：

```bash
tail -n 50 -F "/data/yanghaojun/outputs/logs/${TABERO_BASE_RUN:?}.log"
```

新终端先设置 `TABERO_BASE_RUN` 为实际训练名，或直接填入日志完整路径。训练默认前台运行；需要断开 SSH 时，先在自己创建的 tmux 会话中启动。不要关闭训练终端。

逐条指标保存在 `/data/yanghaojun/outputs/checkpoints/pi0_lora_tabero_rgb_state/<训练名>/metrics.jsonl`；实际参数保存在同目录 `run_config.txt`。训练期验证每次覆盖 692/693 帧，离线评估覆盖全部 693 帧。

## 3. 训练完成后离线评估

先确认日志成功结束，并出现最终 `2999` checkpoint。之后重新选择一张空闲 GPU；不要把训练使用的多卡列表直接传入单卡评估。

```bash
read -r -p "输入离线评估使用的一张空闲 GPU 编号: " TABERO_EVAL_GPU
```

```bash
TABERO_BASE_CKPT="/data/yanghaojun/outputs/checkpoints/pi0_lora_tabero_rgb_state/${TABERO_BASE_RUN:?}/2999"
```

```bash
TABERO_BASE_EVAL="offline_${TABERO_BASE_RUN:?}_2999"
```

```bash
CUDA_VISIBLE_DEVICES="${TABERO_EVAL_GPU:?}" bash scripts/run_tabero_offline.sh "${TABERO_BASE_EVAL:?}" --config=pi0_lora_tabero_rgb_state --checkpoint="${TABERO_BASE_CKPT:?}" --seed=42 --num-denoise-steps=10
```

输出目录是 `/data/yanghaojun/outputs/offline_eval/<评估名>`，包含 `summary.json`、`predictions.npz/csv`、三张相同布局的轨迹图和 `eval.log`。已存在目录会被拒绝；重新评估请用新名字。

```bash
tail -n 30 -F "/data/yanghaojun/outputs/offline_eval/${TABERO_BASE_EVAL:?}/eval.log"
```

## 4. 与现有触觉结果做探索性比较

无触觉评估完成后运行，**此命令只读取结果，使用 CPU，不加载模型**：

```bash
JAX_PLATFORMS=cpu /data/yanghaojun/envs/tabero-smoke/bin/python scripts/compare_tabero_offline.py --touch=/data/yanghaojun/outputs/offline_eval/offline_2999_20260903_125340 --no-touch="/data/yanghaojun/outputs/offline_eval/${TABERO_BASE_EVAL:?}" --output-dir="/data/yanghaojun/outputs/offline_eval/compare_${TABERO_BASE_EVAL:?}"
```

终端及 `comparison.txt` 给出整体表格，`comparison.json` 还包含逐 episode 结果。指标包含首动作、有效完整 chunk 的位置 mm、SO(3) 旋转角度和单指夹爪 mm，以及两模型预测之间的差异。

`no_touch_minus_touch` 的均值为正，表示触觉组误差较低；为负，表示无触觉组误差较低。脚本检查逐帧标签/state/掩码、采样设置与共享归一化一致，但**不会把这种检查声称为训练协议完全匹配**。旧评估没有逐帧 RGB 文件哈希，还须保证原始视频和任务文本未被修改。

## 5. 可选：从共同起点重训触觉组

这不是启动无触觉训练的前置条件，也不会自动运行。若要做匹配对照，使用相同 GPU 数量和独立训练名：

```bash
TABERO_TOUCH_RUN="rgb_state_touch_matched_$(date +%Y%m%d_%H%M%S)"
```

```bash
CUDA_VISIBLE_DEVICES="${TABERO_BASE_GPUS:?}" bash scripts/run_tabero_baseline.sh "${TABERO_TOUCH_RUN:?}" rgb_state_touch
```

离线评估改用 `--config=pi0_lora_tabero_rgb_state_touch`，checkpoint 为 `/data/yanghaojun/outputs/checkpoints/pi0_lora_tabero_rgb_state_touch/<触觉训练名>/2999`。评估选帧、seed、去噪步数保持不变。然后把比较命令的 `--touch` 换成新触觉评估目录；不要根据哪组表现好临时选择不同的 checkpoint 步数。
