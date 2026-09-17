# state/action 首尾差异排查（2026-09-03）

**结论：episode 4 开头约 423 mm 的差异在训练前的 action 标签中已经存在。连续 11 帧沿用上一条轨迹的最后一个目标，而真实机器人已经复位。不是绘图、模型输出反归一化或简单错一帧造成的。** 现有证据高度支持旧目标跨 episode 残留；还缺原始 action 事件与采集/转换脚本，不能进一步断定残留发生在采集缓存还是转换时的时间匹配/前向填充。

轨迹末尾是另一个问题：episode 4 最后一帧真实目标与反馈只有 3.63 mm 差异，模型预测与真实目标差 32.64 mm，姿态误差 107.17°。不能把所有预测误差都归因于开头的脏标签，也不能由此认定训练代码完全没有其他问题。

## 输入与验证范围

- [原始反馈附件](/home/yanghaojun/.codex/attachments/7a3207e0-6628-4ea2-b07c-332c8eb1a530/robot_state_events.jsonl)：10014 条记录，全部 `type=robot_state, ok=true`；没有下发 action，没有 episode ID。
- [数据集转换元信息](/data/yanghaojun/datasets/tabero_lerobot_compact_v1/meta/tabero_conversion.json)：29 条输出轨迹，7D 绝对 XYZ、旋转向量、单指位置；原始目录 `/media/enine/Extreme SSD/hj/test` 在当前机器不存在。
- [已完成评估数组](/data/yanghaojun/outputs/offline_eval/offline_2999_20260903_125340/predictions.npz)：693 个观测锚点，每个 50 步 action chunk。
- 对全部 29 条 Parquet 读取 state/action；所有 XYZ state 都能在附件中找到 float32 精度内的反馈。episode 4 进一步验证连续时间对应、完整姿态与夹爪。
- 对评估的 3 条轨迹、693 个锚点、全部 34650 个 chunk 槽位逐项验证目标：与所属 episode 的 Parquet 一致；边界外重复最后目标，valid mask 正确。没有混入下一条 episode。
- 仅运行 CPU 数据分析及实际变换类的数值往返验证；没有训练、模型推理、修改数据/权重或机器人通信。

## episode 4 开头的直接证据

帧索引从 0 开始；下表位置单位为 mm。

| 帧 | state XYZ | recorded action XYZ | action 与 state 距离 |
|---|---|---|---:|
| 0 | [306.392, 0.291, 414.350] | [688.212, -29.003, 233.827] | 423.359 |
| 10 | [306.394, 0.291, 414.354] | [688.212, -29.003, 233.827] | 423.359 |
| 11 | [306.392, 0.291, 414.350] | [308.562, 0.072, 411.316] | 3.737 |
| 221（末帧） | [594.263, 72.390, 242.906] | [595.049, 71.161, 246.231] | 3.632 |

- **episode 4 第 0～10 帧 action 的全部 7 个数，精确等于 episode 3 最后一帧 action。** 第 11 帧目标突然跳变 420.10 mm，回到当前机器人附近。
- 前 11 帧反馈相对第一帧的最大位移只有 **0.00533 mm**。机器人没有执行那段 423 mm 的“下一帧移动”。
- episode 4 对应附件第 **1229～1450 行**，连续 222 条。位置最大误差 0.0000301 mm；按 `pose[3:]` 为 xyzw 四元数比较，姿态最大误差 0.00000713°；`gripper_width / 2` 与 state 夹爪值完全一致。
- 原始第 1228 行还在上一轨迹末端；隔了 **27.7195 s**，第 1229 行已回到初始位姿。缺口中没有复位过程记录，不能把两行相邻当作 0.1 s 的运动；但复位后的目标仍等于上一条末端目标这一模式很明确。
- episode 4 转换报告 `compacted=false`、缺失候选帧数为 0，不能用压缩时间轴解释其开头差异。

首帧预测 XYZ=[307.267,-1.122,405.833] mm。它与当前反馈只有 **8.68 mm** 差异，但与那个旧目标相差 **418.91 mm**。原图开头约 400 mm 的误差峰值，主要反映了异常目标标签。

整条 episode 4 的预测/目标平均位置误差为 **37.15 mm**；前 11 帧为 **391.37 mm**；其余 211 帧为 **18.68 mm**。这里只做分段诊断，没有删除标签或修改正式评估分数。

## 不是单条轨迹的偶发问题

- 29 条中 **28 条**首帧 action/state 位置差 >50 mm，27 条 >100 mm。
- **14 条**首帧 action 的全部 7 个数精确等于上一条已导出轨迹的末帧：3、4、5、8、9、12、14、16、18、19、20、24、27、28。
- 28 条大偏差轨迹的首段 action 恒定区共 291 帧。恒定和阈值仅作为排查线索，不能单凭它们自动判定所有帧无效。
- 其他轨迹开头不等于上一条已导出的末尾，可能与未导出的操作、暂停或不同时间匹配有关；现有文件不能确定原因。

逐条统计见 [summary.json](/home/yanghaojun/Tabero-VTLA/diagnostics/state_action_20260903/summary.json)。

## 为什么 action 不等于 state 的下一帧

当前 [data_loader.py](/home/yanghaojun/Tabero-VTLA/src/openpi/training/data_loader.py:160) 从同一条记录开始查询 action 列，时间偏移为 `[0,0.1,...,4.9]` s，构成 50 步监督序列。代码中没有 `actions = state[1:]` 的构造。

如果 action 是控制器目标，它与下一次采样的实际反馈本来就不保证相等。位置/阻抗控制的实际响应、记录时刻和异步查询都可能带来差异。若希望训练目标严格定义为 `state[t+1]`，必须在数据构建时明确做这种标签定义；这会改变模仿的对象，需要同时考虑真实机器人执行接口，不能只为让曲线重合而移位。

对 episode 4 正常运动区（锚点 25～211）做位置匹配：

| 比较 | 平均位置差 |
|---|---:|
| action[t] vs state[t] | 13.107 mm |
| action[t] vs state[t+1] | 9.684 mm |
| action[t] vs state[t+2] | 6.374 mm |
| action[t] vs state[t+3] | 4.565 mm |
| action[t] vs state[t+4] | 6.069 mm |

另外两条验证轨迹也在 +3 帧附近最接近。这是约 0.3 s 的**综合跟踪/对齐滞后现象**，不是经过 action 发送时间戳证实的控制器延迟，不能据此直接全局平移 3 帧。开头即使改成与 state[t+1] 比较，首帧仍相差 423.359 mm。

## 模型和绘图代码检查结果

原图蓝线是数据集 action，绿线是当前反馈，橙线是每个观测独立推理得到的第一个 action。它不是闭环执行后得到的机器人轨迹。绘图 [tabero_offline.py](/home/yanghaojun/Tabero-VTLA/src/openpi/policies/tabero_offline.py:200) 直接取这些数组并乘 1000 转为 mm，没有额外时间移位。

当前训练 [config.py](/home/yanghaojun/Tabero-VTLA/src/openpi/training/config.py:438) 使用：

1. 第一至第六维：`action[t+k] - state[t]`，夹爪保持绝对位置。
2. Normalize 后送入模型；不是相邻 action 相减。
3. 输出经过 Unnormalize，再加回锚点 state，恢复绝对 action。

用实际 `DeltaActions → Normalize → Unnormalize → AbsoluteActions` 和该 checkpoint 的统计量，对全部已存监督 chunk 做往返验证：XYZ 最大误差 **0.0000152 mm**，完整 action 分量最大误差约 2.38e-7。没有发现足以解释 423 mm 差异的归一化/逆变换问题。此检查不证明模型参数或其他所有代码均无问题。

**姿态表示另有需要处理的训练风险。** episode 4 的 state 第 12→13 帧旋转向量分量跳变约 6.283 rad，但真实 SO(3) 旋转只有 0.086°。当前 DeltaActions 直接减旋转向量分量，不是计算相对旋转。全数据共有 638 帧满足“action/state 旋转向量差 >3 rad，但实际姿态差 <10°”，episode 4 占 29 帧。这会把物理上接近的姿态变成数值上很远的监督，可能增加训练难度；不能仅据此证明所有姿态预测尖峰的原因，也不能解释 XYZ 开头的旧目标残留。

原评估姿态误差已使用 SO(3) 最短角，因此末帧 107.17° 是真实预测姿态差，不是单纯的 ±π 画图假象。

## 末尾应分开解释

episode 4 收尾抬升阶段，目标先移动、实际反馈随后跟进，第 212 帧 action/state 差 47.46 mm，末帧已经收敛至 3.63 mm。末帧预测/目标仍差 32.64 mm，属于这次离线模型输出误差。

夹爪末帧单指目标为 42.50 mm，实际为 40.066 mm，差 2.434 mm；原始 `gripper_width/2` 精确支持这个实际值。这里比较的是命令目标与实际位置，不是读同一个反馈再移一帧。未取得原始命令/夹爪控制实现，不能断言是标定、限位或其他具体机制。

## 建议修复顺序

1. 先定位旧目标来源：检查 episode reset/start 是否清空或重新初始化 `last_action`/目标 pose；检查 exporter 的 previous/nearest/forward-fill 是否跨 episode 使用事件，以及 reset 命令是否未被记入动作日志。
2. 采集记录应带上 episode ID、单调时间、命令有效标记，区分“已发送且当前有效的目标”和 UI/遥操作中保留的旧变量。新 episode 首个有效目标到来前，不能默认继承上一条目标作为训练标签。
3. 旧数据按原始 action 时间戳与 episode 边界重建同步；只在证实标签无效后裁剪或排除对应观测/action chunk，避免历史窗口和未来 chunk 穿过无效边界。原始文件保留，产出新的数据集版本；按同一训练划分重算统计量、重训后评估。
4. 位置标签清理之后，单独处理旋转向量分支问题。连续化或相对 SO(3) 表示必须配套训练/推理逆变换及既有权重兼容性验证，不能只改一行减法。

要定位到具体采集/转换代码行，仍需同次采集的 **action/teleop 事件日志、原始 episode 4 Parquet、采集脚本及 LeRobot 转换脚本**；其中原始 episode Parquet 与转换脚本可以先把“转换前就错了”与“转换时才错”分开。

## 复现与产物

[分析脚本](/home/yanghaojun/Tabero-VTLA/diagnostics/state_action_20260903/analyze.py)、[逐帧数值](/home/yanghaojun/Tabero-VTLA/diagnostics/state_action_20260903/episode_000004_frames.csv)、[分段对比图](/home/yanghaojun/Tabero-VTLA/diagnostics/state_action_20260903/episode_000004_boundaries.png)。分析脚本包括数据/评估 chunk 一致性断言；变换往返数值由单独 CPU 探针验证。

```bash
MPLCONFIGDIR=/tmp/tabero-state-action-matplotlib PYTHONDONTWRITEBYTECODE=1 /data/yanghaojun/envs/tabero-smoke/bin/python /home/yanghaojun/Tabero-VTLA/diagnostics/state_action_20260903/analyze.py
```

![episode 4 首尾证据](/home/yanghaojun/Tabero-VTLA/diagnostics/state_action_20260903/episode_000004_boundaries.png)
