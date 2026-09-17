# FR3 推理主机重写交接文档

本文用于在推理主机上重新实现 Tabero FR3 真机推理与控制程序。基线代码为
`fix/fr3-tactile-shadow-deploy` 分支的提交 `53370e642c72ac468194ef773bb20cf8ba8d0403`。
新实现必须先复现这里记录的输入输出契约、安全行为和诊断能力，再调整推理调度。

当前 checkpoint 已经能够产生动作并驱动机械臂，但五次 K=2 抓取/装配均失败。日志证明机器人
确实移动并跟随了客户端命令，因此重写目标不是简单地“把模型输出发给机械臂”，而是建立可验证的
观测、推理、调度、保护和闭环评估链路。

## 1. 当前基线和权威文件

| 项目 | 当前唯一基线 |
|---|---|
| GitHub | `Leohahahahha/Tabero_serve` |
| 分支 | `fix/fr3-tactile-shadow-deploy` |
| 本文基线提交 | `53370e6` |
| 模型配置 | `pi0_lora_tabero_v3_touch_20k` |
| checkpoint | 完整的 `20000/` 目录，必须同时包含 `params/` 和 `assets/` |
| 数据资产 | `local/tabero_lerobot_compact_v3` |
| 服务入口 | `scripts/serve_tabero.py` |
| 现有客户端入口 | `examples/fr3_deploy/run.py` |
| 运行配置 | `examples/fr3_deploy/config.json` |
| 转换契约 | `examples/fr3_deploy/tabero_conversion.json` |
| 传感器适配 | `examples/fr3_deploy/observations.py`、`dmtac_w_ipc.py` |
| 单位、安全和几何 | `examples/fr3_deploy/core.py` |
| 网络协议 | `examples/fr3_deploy/transport.py` |

当前两个必须核对的哈希为：

```text
tabero_conversion.json SHA256:
1541807cb3e51b3c0843bd27972f1172e1027584833329ea50f45c80dece1ffb

checkpoint 20000 v3 norm_stats.json SHA256:
3179b7ee7553cac64f3ac8a40897ed620b78b2ee354f9a297534f88cc66a0aed
```

不要使用旧配置 `pi0_lora_tacfield_local_tactile_lora_smoke` 加载 checkpoint 20000。旧配置查找 v1
归一化资产，会造成模型、统计量与数据预处理不一致。也不要把旧异步客户端作为新实现的时序基线。

## 2. 推荐的进程边界

相机、触觉和控制客户端都可以在推理主机运行，但模型服务与 ROS2 客户端仍建议使用两个进程和
两个 Python 环境：

```mermaid
flowchart LR
    ZED[ZED ROS2] --> O[观测采样器<br/>ROS/Jazzy Python]
    D405[D405 ROS2] --> O
    TAC[DM-Tac W ROS2] --> O
    FR3S[FR3 /getstate] --> O
    O --> C[同步控制客户端<br/>.venv-fr3]
    C -->|msgpack-numpy WebSocket| P[策略服务<br/>JAX/Python 3.11 .venv]
    P -->|metadata + actions 50x7| C
    C --> G[TargetGuard + 调度器]
    G -->|/pose + /move_gripper| FR3[franka_server]
    HB[/tabero/enable] --> C
    O --> L[观测与时序日志]
    P --> L
    G --> L
```

- 模型服务环境保留仓库的 Python 3.11/JAX 依赖。
- ROS 客户端使用 ROS2 Jazzy 对应的系统 Python，通过 `--system-site-packages` 访问 `rclpy`。
- ROS 客户端只需 `numpy`、`scipy`、`requests`、`opencv-python`、`websockets`、`msgpack` 和
  `packages/openpi-client`，不需要安装 JAX 或 PyTorch。
- 两个进程在同一主机时继续使用 `ws://127.0.0.1:8000`，避免不必要的跨机网络延迟。

建议把新客户端拆成以下职责，模块名可以变化，但边界不要混在一个循环中：

| 模块 | 职责 |
|---|---|
| `contracts` | shape、dtype、单位、哈希、协议版本和 metadata 校验 |
| `sensor_node` | ROS 订阅、时间戳校验、RGB 解码/裁剪、触觉历史构造 |
| `policy_client` | WebSocket 连接、超时、序列化、动作 chunk 校验 |
| `robot_client` | `/getstate`、`/pose`、`/move_gripper`；运动请求不自动重试 |
| `guard` | 工作空间、目标幅度、SO(3)、跟踪误差、速度和夹爪保护 |
| `scheduler` | K=1/K=2 时序、结果年龄、持续使能和重规划 |
| `run_logger` | JSONL 事件、传感器快照索引、延迟分解和任务结果 |
| `main` | 状态机、warmup、shadow/execute 选择、退出和 hold |

## 3. 模型输入契约

模型请求必须包含以下字段，不能只保证数组 shape 相同：

| 字段 | 必须满足的语义 |
|---|---|
| `image` | ZED 前视 RGB `uint8`；原图 `[540,960,3]`，裁剪 ROI `xyxy=[350,0,740,520]`，输出 `[520,390,3]` |
| `wrist_image` | D405 腕部 RGB `uint8 [480,640,3]` |
| `state` | 有限值 `[x,y,z,rx,ry,rz,finger_m]`；XYZ 为米，姿态为 rotvec/轴角弧度，夹爪为单指绝对位置 |
| `prompt` | `Align the black circular component with the receiving hole on the gray circular component and insert it to complete the assembly.` |
| `tactile_marker_motion` | 连续内存 `float32 [9,198,2]`；第 0 帧固定参考网格，第 1～8 帧为从旧到新的触觉 marker 坐标 |

### 3.1 ROS2 话题和 QoS

| 数据 | 当前话题 | 消息/QoS |
|---|---|---|
| ZED 前视 | `/zed/zed_node/rgb/color/rect/image/compressed` | `CompressedImage`，RELIABLE，depth 1 |
| D405 腕部 | `/camera/d405/color/image_raw` | `Image`，RELIABLE，depth 1 |
| 左触觉 | `/dmtac/left/packed_frame` | `Image` packed，BEST_EFFORT，depth 1 |
| 右触觉 | `/dmtac/right/packed_frame` | `Image` packed，BEST_EFFORT，depth 1 |
| 持续使能 | `/tabero/enable` | `Bool`，RELIABLE，depth 1 |

DM-Tac packed schema 由 `dmtac_w_ipc.py` 的 `packed_layout_metadata("shear_depth")` 提供。
程序只读取每侧 `float32 [240,320,2]` shear，在 9×11 网格上采样。每帧先放左侧 99 点，
再放右侧 99 点，当前 marker 坐标为：

```text
fixed_reference_xy + shear_scale * sampled_shear_xy
```

`shear_scale=1.0`。历史长度为 8；启动时复制首个有效当前帧补齐，采样间隔超过 150 ms 时清空历史。
不能用全零数组替代缺失触觉，不能交换左右侧、xy 通道或历史方向。

### 3.2 HTTP 状态转换

`POST /getstate` 必须返回：

```json
{
  "pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
  "gripper_width": 0.085,
  "stamp": {"to_sec": 0.0}
}
```

其中 pose 是 `XYZ + XYZW quaternion`，`gripper_width` 是两指总宽度米制值。转换为模型 state 时：

```python
state[:3] = pose[:3]
state[3:6] = Rotation.from_quat(pose[3:7]).as_rotvec()
state[6] = gripper_width / 2.0
```

四元数转 rotvec 和总宽转单指只做一次。禁止读取旧接口的 0～1 `gripper_pos` 作为米制宽度。

### 3.3 时间戳

- 相机、触觉和机器人状态都必须使用采集时间戳，不能拿接收时刻伪装成采集时刻。
- 各发布主机与推理主机必须通过同一时间源对时。
- 当前 `max_sensor_age_sec=0.25`，多模态 `max_sensor_skew_sec=0.10`。
- 重复或倒退的图像时间戳不刷新健康状态。
- 采样线程以 10 Hz 构造同步观测；观测缓存自身超过 150 ms 也视为过期。

## 4. 模型服务和输出契约

服务启动时应先发送 metadata，客户端在发送第一份观测前校验：

- `deployment_protocol == "tabero_fr3_absolute_v1"`
- `config == "pi0_lora_tabero_v3_touch_20k"`
- `asset_id == "local/tabero_lerobot_compact_v3"`
- `action_representation == "absolute_xyz_axis_angle_single_finger_m"`
- `action_dim == 7`、`action_horizon == 50`、`dataset_fps == 10`
- `use_tactile == true`、触觉 shape/dtype/layout 完全一致
- `predicts_wrench == false`
- conversion SHA 和 norm-stats SHA 与本节开头记录一致

WebSocket 使用 msgpack-numpy。请求是第 3 节的 observation dict；响应必须至少包含有限值
`actions [50,7]`。策略服务已经完成输入归一化、输出反归一化和绝对动作恢复，客户端禁止再次
反归一化、再次加当前 state 或套用遥操作的增益/坐标映射。

每个动作的含义为：

```text
[absolute_x_m, absolute_y_m, absolute_z_m,
 absolute_rotvec_x_rad, absolute_rotvec_y_rad, absolute_rotvec_z_rad,
 absolute_single_finger_position_m]
```

发送到机器人时使用：

```python
pose_xyz_xyzw = np.r_[action[:3], Rotation.from_rotvec(action[3:6]).as_quat()]
total_gripper_width_m = 2.0 * action[6]
```

模型内部训练使用相对 XYZ 和 SO(3) 相对旋转，但策略输出适配器返回的是上述绝对目标。

## 5. 控制状态机和安全行为

建议新实现显式维护以下状态：

```text
STARTUP -> WAIT_SENSORS -> WARMUP -> SHADOW_READY
                                  -> WAIT_ENABLE -> ARMED -> INFER -> GUARD -> COMMAND
                                                        ^                    |
                                                        +---- REPLAN <-------+
任意状态 -- fault/timeout/enable lost/SIGINT --> STOPPING -> best-effort HOLD -> EXIT
```

第一轮推理只用于 JAX 编译，输出必须丢弃。execute 只能在 warmup 完成后收到新的、持续更新的
true 心跳；false、250 ms 内没有新心跳或运行中丢失心跳都要退出。

### 5.1 有限预测做饱和

当前顺序应保持：

1. XYZ 投影到 `[0.30,-0.30,0.08]`～`[0.75,0.30,0.45]` 工作空间。
2. 相对当前实测位置的目标距离最多 50 mm。
3. 相对当前实测姿态的 SO(3) 最短旋转最多 0.35 rad。
4. 单指位置投影到 `[0,0.0425] m`。
5. 相对上一条已提交命令按实际周期限速；`dt` 最多按 100 ms 计算，卡顿不能产生追赶大步。

当前速度上限为平移 `0.02 m/s`、旋转 `0.10 rad/s`、总夹爪宽度 `0.02 m/s`，即正常
100 ms 周期最多移动 2 mm、旋转 0.01 rad、总宽改变 2 mm。

### 5.2 实测或系统故障直接停止

以下情况不能通过裁剪继续运行：

- state、action、时间戳出现 NaN/Inf 或 shape 错误；
- 实测 XYZ 已经离开工作空间，或实测夹爪宽度非法；
- 机械臂相对上一条命令的跟踪误差超过 30 mm 或 0.20 rad；
- 任一传感器过期、跨模态 skew 超过 100 ms、触觉 schema/history 错误；
- 推理结果年龄超过 350 ms；
- enable 心跳失效；
- WebSocket/HTTP 超时、拒绝或协议错误。

`/pose` 和 `/move_gripper` 是两个非原子 HTTP 请求。运动请求 timeout 后不能判断服务端是否已经执行，
所以不能自动重试。退出时的当前位置 `/pose` hold 只是尽力保持，不是急停；底层 watchdog、碰撞保护
和硬件停止必须独立存在。

## 6. K=1、K=2 与推理频率

数据集和动作 chunk 的时间语义是 10 Hz，即相邻动作相隔 100 ms。当前同步实现支持：

- K=1：每次新观测推理后只执行 `action[0]`，闭环最新，但约 192 ms 推理导致实际约 5 Hz。
- K=2：执行 `action[0]`，等待 100 ms，重新读取实测 state 并运行全部保护，再执行 `action[1]`；
  随后重新采样和推理。

五次真机 K=2 的实际控制频率为 `5.97～6.18 Hz`。在 3,255 个可执行 chunk 中，第二步成功执行
2,620 次，即 `80.5%`；634 个 chunk 因第二步过期等原因被截断。K=2 提高了动作吞吐，但降低了
视觉和触觉重规划频率。现有日志没有同场景 K=1/K=2 成对试验，不能断言 K=2 是五次失败的根因。

重写初期继续保留 K=1/K=2，保持每步 100 ms 时间语义和 350 ms 时效门限。不要直接放开 K=5 或
K=15：以当前延迟，后续动作会过期；若放宽门限强行执行，则会长时间使用旧观测，接触和抓取阶段
几乎失去高层反馈。若需要接近 10 Hz，应先减少端到端推理延迟，或另行设计带明确时间戳、执行进度
和新旧 chunk 连续性检查的流水线方案。

## 7. 目前部署中已经确认的问题

### 7.1 五次 K=2 真机结果

下表来自 `sync_k2_execute_003.jsonl`～`007.jsonl`。五次的 config、checkpoint、asset、prompt、
normalization SHA 和 conversion SHA 完全一致；物体位置、光照和相机画面没有保存在日志中，无法证明
现场视觉条件完全相同。

| 运行 | 控制频率 | 控制步 | 累计路径 | 净位移 | 相对最近专家终点的位置差 | 单指最小值 | 结束原因 |
|---|---:|---:|---:|---:|---:|---:|---|
| 003 | 6.18 Hz | 1098 | 617 mm | 337 mm | 124 mm | 31.45 mm | 人工停止 |
| 004 | 6.18 Hz | 1482 | 727 mm | 353 mm | 148 mm | 31.64 mm | 时长结束 |
| 005 | 6.10 Hz | 1018 | 576 mm | 325 mm | 188 mm | 40.63 mm | 人工停止 |
| 006 | 5.99 Hz | 1437 | 656 mm | 352 mm | 182 mm | 32.76 mm | 时长结束 |
| 007 | 5.97 Hz | 840 | 511 mm | 311 mm | 104 mm | 32.02 mm | ZED/状态 skew 102 ms |

已经确认的结论：

- 机械臂不是“没有执行”。5,875 条 pose 命令全部返回成功，五次都移动了 31～35 cm 净距离。
- 底层跟踪不是主要故障。位置跟踪 p95 为 5.0～5.8 mm，最大 8.3 mm，远低于 30 mm 停止阈值；
  姿态跟踪最大 1.37°。
- 策略没有稳定到达专家任务终点。末态相对最近专家终点仍差 104～188 mm 和 10～20°；几乎相同
  的起点最终 XYZ 最多相差约 111 mm。
- 轨迹发生闭环漂移。沿途最近训练 state 的位置差 p95 为 25～53 mm，末态最近任意训练 state 的
  姿态差为 8～20°。
- `3,653/5,875 = 62.2%` 控制步触发限幅，主要是平移速度限幅。实际执行路径经常不是原始模型
  目标的原幅度轨迹。
- 005 的单指位置始终为 40.63～42.50 mm，几乎没有闭合；训练示范每条轨迹的最小单指值为
  27.52～34.64 mm。其它运行闭合更多仍失败，所以夹爪不闭合是一个确定的失败模式，不是唯一原因。
- 007 因前视 ZED 成为最老数据、跨模态 skew 达到 102 ms 而安全退出；它不能解释其它四次失败。

### 7.2 离线误差小但真机失败

checkpoint 20000 的离线评估只使用留出的专家轨迹 4、14、24，共 690 个 anchor。它把每个专家
观测输入模型，再把预测与该专家轨迹未来动作比较，不会让模型看到自己执行错误后产生的新画面，
也不统计是否对准、抓住或插入物体。

| 离线指标 | 位置 | SO(3) 旋转 | 单指夹爪 |
|---|---:|---:|---:|
| 模型第一步平均误差 | 4.623 mm | 0.163° | 0.442 mm |
| 保持当前位置基线第一步平均误差 | 3.627 mm | 0.126° | 0.081 mm |
| 模型完整有效 chunk 平均误差 | 38.622 mm | 1.226° | 1.405 mm |
| 模型完整有效 chunk P95 | 101.286 mm | 3.124° | 5.019 mm |

第一步只有 100 ms，专家本身移动较小，因此“什么也不做”的基线平均误差反而更小。4.623 mm
不能证明模型学会了任务。完整 chunk 的长时误差明显更大，而且仍属于专家观测上的 teacher-forced
测试。真机一旦偏离，后续图像、状态和触觉都离开训练分布，误差会逐步累积。

因此当前矛盾的直接原因是离线指标解释错误；真机任务失败的最终根因仍未完全定位。已观察到闭环漂移、
高限幅率、6 Hz/10 Hz 时间不匹配和一次夹爪不闭合。相机画面/物体位姿/光照、触觉数值语义、TCP
或标定偏差仍是假设，因为现有 JSONL 没有保存传感器 payload。

### 7.3 数据和表示问题的当前状态

- v1 跨 episode 的错误动作标签已经在 v3 修复；新实现只允许 v3 资产。
- v3 磁盘中仍有 114 帧 rotvec 分量跨分支约 `2π`，但最大真实相邻旋转只有 1.225°；训练和部署
  比较姿态时使用 SO(3)，不能按 rotvec 分量直接相减。
- v3 有 12 条 episode 共压缩了 18 个缺失采样间隔；这会造成训练时间轴与真实动态不完全一致。
- 2,726/6,626 个 v3 一步专家目标超过当前 2 mm/100 ms 平移限制，训练速度分布与部署 guard 不匹配。
- 这五次 K=2 的起始姿态已经对齐示范起点：最近 episode 起点的位置差 0.17～1.47 mm、姿态差
  0.09～0.53°。早期运行的 30～40° 起始姿态偏移不能用于解释本次五次失败。

## 8. 新实现必须补充的诊断能力

现有日志能检查动作与控制，却无法回看模型究竟看到了什么。新实现至少增加：

1. **观测包留档**：在 shadow 全量或按配置抽样保存前视 RGB、腕部 RGB、完整
   `tactile_marker_motion`、state 和每路原始时间戳；JSONL 保存相对文件名和 SHA256。
2. **端到端延迟分解**：分别记录采集年龄、同步等待、图像处理、序列化、WebSocket 排队、GPU 推理、
   反序列化、guard、HTTP 往返和控制器确认时间。
3. **可复现推理**：记录推理 seed/采样器参数；支持同一冻结观测重复 N 次，量化 flow-matching 随机性。
4. **动作三层日志**：保留 `raw_action`、`bounded_action`、`rate_limited_action` 和每个触发的限制。
5. **实际执行进度**：记录 observation id、chunk id、action index、计划发送时刻、真实发送时刻、结果年龄、
   命令前后 state 和跟踪误差。
6. **夹爪闭环**：同时记录模型单指目标、发送的总宽、实测总宽、接触阶段和 HTTP 返回；避免只知道
   请求成功却不知道物体是否被夹住。
7. **任务阶段和结果**：至少标记 reach、pre-grasp、close、lift/transport、align、insert，以及人工确认的
   success/failure。离线和真机报告都按阶段统计，不能只给全程平均误差。
8. **首个异常快照**：传感器 skew、非有限输出、严重饱和、跟踪失败或人工停止时保存最近一组完整观测。

观测文件可能很大，应使用每次运行的独立目录、固定上限和 manifest，不能把二进制数组直接塞进每条
JSONL。日志中不得记录凭据。

## 9. 重写验收顺序

### 阶段 A：纯契约测试

- 用当前测试 fixture 验证 RGB 通道/裁剪、带 row padding 的 ROS Image、DM-Tac packed endian/offset、
  触觉网格和历史顺序。
- 验证 quaternion/rotvec 的往返、`±π` SO(3) 最短距离、总宽/单指转换只发生一次。
- 对 metadata、conversion SHA、norm SHA、checkpoint asset、错误 shape/dtype/NaN 建立失败测试。
- 对全部安全边界建立边界内、恰好边界和边界外测试。
- 模拟 HTTP timeout，确认 pose/gripper 不重试且状态机退出。

### 阶段 B：保存观测回放

- 先让旧客户端和新客户端读取同一批已保存观测，不连接机器人。
- 对比发送给策略的四个输入字段，要求数组、dtype、裁剪、左右顺序、历史和 prompt 一致。
- 固定推理 seed 后比较 `actions [50,7]`；若协议或预处理未改变，应逐项一致或达到明确数值容差。
- 对同一观测重复推理，先量化随机波动，再讨论跨时刻策略变化。

### 阶段 C：现场 shadow

- 默认 K=1，至少完整覆盖一次任务时长；不发送 `/pose` 或 `/move_gripper`。
- 确认所有 sensor age/skew、metadata、shape 和 hash 检查通过。
- 回看保存的 ZED/D405 图像和触觉数组，人工确认物体、裁剪、颜色、相机方向及触觉变化符合训练语义。
- 统计 action0、action1、完整 chunk、相邻推理跳变和限幅率，并按任务阶段展示。
- 新旧客户端在同一保存观测上的差异解释清楚后，才能停用旧客户端。

### 阶段 D：受保护执行

- 先保持 K=1、小于 10 秒、持续人工使能和硬件停止可达；确认命令、实测状态和画面方向一致。
- 再进行 K=2 shadow；只有结果年龄、第二步利用率、轨迹连续性和任务阶段方向均可接受时，才做 K=2
  短执行。
- 不通过放宽 workspace、跟踪、时效或 K 值来掩盖失败。
- 正式比较必须固定起始姿态、物体位姿、相机、光照、checkpoint、seed/采样设置和安全限制，分别记录
  K=1 与 K=2；没有成对试验不能归因于执行步数。
- 最终用任务成功率和阶段成功率验收。下一动作平均误差只能作为诊断指标。

五次 K=2 已经连续任务失败。在完成观测留档、回放和阶段诊断前，不建议继续重复长时间 execute。

## 10. 推理主机操作基线

### 10.1 代码和模型

```bash
git clone --branch fix/fr3-tactile-shadow-deploy git@github.com:Leohahahahha/Tabero_serve.git ~/Tabero_serve
```

如果目录已存在：

```bash
cd ~/Tabero_serve && git fetch origin && git switch fix/fr3-tactile-shadow-deploy && git pull --ff-only origin fix/fr3-tactile-shadow-deploy
```

开始重写前从该基线创建独立分支，并保留旧客户端用于回放对照：

```bash
cd ~/Tabero_serve && git switch -c codex/fr3-inference-rewrite
```

完整 checkpoint 建议仍放在：

```text
/home/enine/tabero_deploy/checkpoints/20000/
├── params/
└── assets/local/tabero_lerobot_compact_v3/norm_stats.json
```

tokenizer 放在：

```text
$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model
```

### 10.2 模型服务

```bash
cd ~/Tabero_serve
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false taskset -c 8-15 .venv/bin/python scripts/serve_tabero.py --config pi0_lora_tabero_v3_touch_20k --checkpoint /home/enine/tabero_deploy/checkpoints/20000 --conversion examples/fr3_deploy/tabero_conversion.json --host 127.0.0.1 --port 8000
```

启动后必须核对 config、checkpoint、asset id 和两个 SHA。首个请求会触发较长 JAX 编译；客户端会丢弃
该次输出。

### 10.3 ROS2 客户端环境

```bash
cd ~/Tabero_serve
source /opt/ros/jazzy/setup.bash
source /home/enine/ros2_hj/install/setup.bash
export ROS_DOMAIN_ID=0
unset ROS_LOCALHOST_ONLY
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
unset ROS_STATIC_PEERS
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///home/enine/cyclonedds.xml
```

先确认话题和频率，再运行客户端：

```bash
ros2 topic list | grep -E 'zed|d405|dmtac|tabero'
```

### 10.4 新实现的第一轮运行

重写完成后的第一轮必须是 shadow。若仍保留原 CLI，可使用：

```bash
taskset -c 4-7 .venv-fr3/bin/python examples/fr3_deploy/run.py --robot-url http://172.31.179.19:5000 --policy-url ws://127.0.0.1:8000 --actions-per-inference 1 --seconds 60 --log deployment_logs/rewrite_k1_shadow_001.jsonl
```

K=1 shadow 通过后再运行 K=2 shadow：

```bash
taskset -c 4-7 .venv-fr3/bin/python examples/fr3_deploy/run.py --robot-url http://172.31.179.19:5000 --policy-url ws://127.0.0.1:8000 --actions-per-inference 2 --seconds 60 --log deployment_logs/rewrite_k2_shadow_001.jsonl
```

不要在重写后的首次运行中加入 `--execute`。execute 应在阶段 A～C 的记录可审查后单独开启。

## 11. 交付清单

新推理代码交接时应同时提供：

- 源码和锁定依赖；
- 协议版本和模型/转换哈希；
- 契约与状态机单元测试；
- 一套不含机器人写操作的保存观测回放测试；
- 一份 K=1 shadow 和一份 K=2 shadow 的 JSONL、观测 manifest 和统计报告；
- 所有配置的最终展开值；
- 已知限制、尚未验证假设和回退到 `53370e6` 的方法；
- 明确的任务成功判定及受保护 execute 操作记录。

更完整的现有操作说明见 `docs/tabero_fr3_deployment.md`；历史问题和面试复盘见
`docs/tabero_interview_incident_log.md`。
