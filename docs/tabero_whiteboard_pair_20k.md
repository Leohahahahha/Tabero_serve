# 白板擦除动作定义配对训练（含力/力矩预测）

本实验使用同一批 39 条白板擦除轨迹训练两个独立模型：

- `pi0_lora_tabero_whiteboard_next_state_force_20k`：动作监督目标是机械臂下一帧实际状态；
- `pi0_lora_tabero_whiteboard_sent_command_force_20k`：动作监督目标是当前帧同步记录的遥操作绝对指令。

两者都从 published Tabero 49999 参数开始新优化器，使用 RGB、腕部 RGB、state 和 `[9,198,2]` 触觉，联合预测 `7D action + 6D wrist_wrench`，训练 20,000 步，每 4,000 步保留 checkpoint。全局 batch 4、seed 42、500 步 warmup、`1e-5 -> 1e-6` cosine、每 1,000 步验证和 W&B 标量设置相同。W&B 图像上传关闭。

6D 顺序固定为 `Fx,Fy,Fz,Tx,Ty,Tz`；前三维单位 N，后三维单位 N·m，坐标系为 K。训练时由独立的 `wrist_wrench` 序列列拼到 7D 动作后面。前 7 维采用 XYZ + SO(3) 相对姿态 + 绝对单指夹爪；后 6 维不做姿态 delta、单位换算或坐标变换。模型使用原始 Tabero 力分支权重 `0.1`，padding 维仍不计损失。W&B/本地日志会分别出现 `action_loss` 和 `tactile_loss`；这里的 `tactile_loss` 实际是后 6 维 wrist-wrench 预测损失。

数据元数据只声明 wrench 来源为同步的 `robot.force + robot.torque`；N/N·m 和 K 坐标系由用户在 2026-09-15 补充确认为外部契约，审计 provenance 会明确记录这一来源。模型推理输出会拆成 `actions [50,7]` 与 `wrist_wrench [50,6]`，机器人控制仍只能消费经过安全检查的 7D 动作，不能把预测力直接当作真机力控指令。

## 固定配对划分

随机种子 42 一次性抽出验证 episode `(1,7,17)`。两套数据都使用这 3 条验证轨迹，共 1,001 帧；每次评估读取 250 个完整 batch，即相同的 1,000 帧。其余 36 条、11,125 帧用于训练。两个模型分别计算 train-only 13D target normalization，禁止互相复用统计。两套导出的 `wrist_wrench` 逐行完全相同，因此实验变量仍主要是前 7 维动作标签来源。

## 标签风险

`sent-command` 导出中 39/39 条轨迹都有超过 50 mm 的相邻指令跳变，共 161 处，最大 451.7 mm。episode 1 的典型记录在机械臂 z 约 0.142 m 时把命令写为 0.600 m，随后跳回约 0.148 m。这可能与指令饱和、遥操作门控或 reset 有关，但没有原始控制日志时不能确认根因。

训练代码不会删除、裁剪或修复这些标签。实验结果只能解释为“当前两个导出标签流的对比”，不能把性能差异完全归因于理想化的实际状态与发送指令定义。

## 启动前

所有命令单行执行，避免工作站交互式 Bash 的反斜杠续行问题。

```bash
cd /home/yanghaojun/Tabero-VTLA
```

```bash
source /data/yanghaojun/envs/tabero-smoke/bin/activate
```

确认 W&B 已登录：

```bash
wandb login
```

检查实时 GPU 状态并选择 1、2 或 4 张卡；全局 batch 始终是 4：

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv
```

先做纯命令 dry-run，不会准备数据或训练：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_tabero_whiteboard_pair_20k.sh --dry-run whiteboard_action_force_ab_20260915
```

## 无人值守顺序训练

先创建持久终端：

```bash
tmux new -s tabero_whiteboard_pair
```

在 tmux 中重新执行上面的 `cd` 和 `source`，然后把 GPU 编号替换成启动时真正空闲的卡：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_tabero_whiteboard_pair_20k.sh whiteboard_action_force_ab_20260915
```

脚本依次执行：next-state 数据审计与统计、sent-command 数据审计与统计、next-state 20k 训练、检查 next-state `20000/params`、sent-command 20k 训练。任一步非零退出都会阻止后续步骤。

按 `Ctrl-b` 后按 `d` 离开 tmux。SSH 断开不会结束 tmux 中的脚本。重新查看：

```bash
tmux attach -t tabero_whiteboard_pair
```

## 状态、日志和输出

```bash
cat /data/yanghaojun/outputs/whiteboard_pairs/whiteboard_action_force_ab_20260915/status.txt
```

```bash
tail -f /data/yanghaojun/outputs/logs/whiteboard_action_force_ab_20260915_pair.log
```

```bash
tail -f /data/yanghaojun/outputs/logs/whiteboard_action_force_ab_20260915_next_state.log
```

```bash
tail -f /data/yanghaojun/outputs/logs/whiteboard_action_force_ab_20260915_sent_command.log
```

checkpoint 分别位于：

```text
/data/yanghaojun/outputs/checkpoints/pi0_lora_tabero_whiteboard_next_state_force_20k/whiteboard_action_force_ab_20260915_next_state/{4000,8000,12000,16000,20000}
/data/yanghaojun/outputs/checkpoints/pi0_lora_tabero_whiteboard_sent_command_force_20k/whiteboard_action_force_ab_20260915_sent_command/{4000,8000,12000,16000,20000}
```

W&B 中两个 run 都属于 `tabero-vtla` project，并共享 group `whiteboard_action_force_ab_20260915`。训练时重点同时看训练曲线 `action_loss`、`tactile_loss` 和验证曲线 `validation/action_loss`、`validation/tactile_loss`，不能只看加权总 loss。

## 中断恢复

使用原来的 pair name 和当前允许使用的相同 GPU 集合：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_tabero_whiteboard_pair_20k.sh --resume whiteboard_action_force_ab_20260915
```

恢复时，已有 `20000/params` 的模型会跳过；未完成模型必须同时存在数值 checkpoint 和 `wandb_id.txt`，否则脚本拒绝猜测或覆盖。
