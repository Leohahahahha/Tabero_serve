# Tabero 真机部署：FR3 + ZED + D405 + DM-Tac W

代码入口：`scripts/serve_tabero.py`（模型主机）和 `examples/fr3_deploy/run.py`（机器人主机）。
默认是 shadow 模式：读取真实传感器、调用模型、记录预测，但不发送运动或夹爪指令。
这份实现依据本次提供的采集、转换和 HTTP 客户端代码；本地验证仅使用 CPU 和模拟接口，尚未连接真机。

## 1. 模型输出不能原样传给 `/pose`

| 数据 | 当前训练/部署约定 |
|---|---|
| 输入 `state` | `[x,y,z,rx,ry,rz,finger_m]`，位置为米，姿态为旋转向量（轴角，弧度） |
| 输入图像 | HWC、RGB、uint8；前视原图 540×960，裁剪 `[350,0,740,520]`；腕部 480×640 |
| 输入触觉 | 必需的 `tactile_marker_motion`，float32 `[9,198,2]`；参考网格 + 8帧历史，每帧左99点后右99点 |
| `policy.infer()["actions"]` | `[50,7]`，反归一化且恢复绝对坐标后的目标 |
| `/pose` | `{"arr":[x,y,z,qx,qy,qz,qw]}` |
| `/move_gripper` | `{"gripper_width": 2*finger_m}`，两指总开口宽度 |

因此，单步动作的接口转换是：

```python
pose = np.r_[action[:3], Rotation.from_rotvec(action[3:6]).as_quat()]
width_m = 2.0 * action[6]
```

以上是接口转换公式；正式执行还要经过 `TargetGuard` 的目标检查和限速。
`rx,ry,rz` 不是 Euler 角、角速度或四元数的前三项；最后一维也不是四元数的 w。
夹爪例如模型输出 `0.030`，HTTP 应发送总宽度 `0.060 m`，不是 0–255 指令。

服务端调用项目已有 `create_trained_policy()`，完成输入归一化、输出反归一化和
`AbsoluteActions` 变换。客户端不要再次反归一化、再加当前状态，或再乘遥操作的
`delta_linear_gain` / `delta_angular_gain` / `R_map`。训练标签已是机械臂基坐标系的目标。
本实现假定部署继续使用相同 FR3 基坐标系、末端/TCP 定义、夹具和相机安装。
移动相机、修改 TCP 或更换机器人基准后，不能只替换 IP 就认为观测仍与训练一致。

## 2. 两台主机如何分工

```text
机器人主机：ZED/D405/DM-Tac ROS2 -> 10 Hz 观测与触觉历史 -> WebSocket 模型主机
                                                        <- 绝对动作 chunk
机器人主机：目标检查、限速、使能和时效检查 -> 原 franka_server HTTP -> FR3
```

模型也可以与机器人客户端运行在同一台机器上，但建议使用两个 Python 环境：
模型保持本项目 Python 3.11/JAX 环境；机器人客户端使用与 ROS2 对应的 Python。
附件 IPC 注释中的 ROS 进程为 Python 3.12，不能把该 `rclpy` 直接放进 3.11 环境。

部署前**停止原来的 `franka_http_bridge_node.py` 控制进程**，保留底层 `franka_server`、
相机和 DM-Tac 发布节点。客户端自行读取 `/getstate`，不依赖原桥接节点发布状态。
只松开原 VR 使能按钮不足以保证独占控制：原代码还有夹爪控制和 A 键复位路径。
独占 HTTP 控制权需要由现场进程管理保证；现有 HTTP 协议没有控制权租约接口。

## 3. 复制代码与模型

模型主机要复制**当前修改后的整个仓库**，不能仅 clone 上游仓库：本地动作适配器、
触觉 LoRA 配置及参数加载逻辑都是当前 checkpoint 的依赖。
另复制训练完成的 step 目录，保留 `params/`、`assets/` 及 checkpoint 元数据。
部署不需要复制原始数据、视频或训练优化器状态；复制整个 step 目录最省事。

本次已有触觉 LoRA 模型的 step 路径为：

```text
/data/yanghaojun/outputs/checkpoints/pi0_lora_tacfield_local_tactile_lora_smoke/real_fr3_recovery_20260903_112020/2999
```

服务端参数传 `2999`，不是 `2999/params`。归一化统计从该 step 的
`assets/local/tabero_lerobot_compact_v1/norm_stats.json` 加载，禁止临时重新计算统计。

`examples/fr3_deploy/tabero_conversion.json` 是当前训练数据转换元数据的原样副本，
包含裁剪、触觉网格、单位及数据来源。模型与客户端都读取此文件，并检查 SHA256 一致。
该校验只能证明两端用了同一份元数据，不能自动证明它对应某个任意 checkpoint。

如模型主机离线，还要从当前机器复制：

```text
/data/yanghaojun/cache/openpi/big_vision/paligemma_tokenizer.model
```

放到新主机的 `$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model`。
没有本地缓存时，项目 tokenizer 会尝试下载。

## 4. 模型主机启动

在复制后的仓库目录安装当前锁定环境（已有相同环境可跳过）：

```bash
uv sync --frozen
```

以下路径 `/path/to/...` 按新主机实际路径替换；选择一张空闲 GPU：

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false OPENPI_DATA_HOME=/path/to/openpi-cache .venv/bin/python scripts/serve_tabero.py --config pi0_lora_tacfield_local_tactile_lora_smoke --checkpoint /path/to/2999 --conversion examples/fr3_deploy/tabero_conversion.json --host 0.0.0.0 --port 8000
```

该配置匹配当前 rank-16 触觉 LoRA checkpoint；不要换成发布版基础配置。
脚本严格校验参数树，缺失或多余权重不会自动随机补齐或忽略。
WebSocket 复用项目现有服务协议，没有认证或 TLS；绑定 `0.0.0.0` 的示例用于受控局域网。

若部署另行训练完成的 RGB+state 基线，改用 `--config pi0_lora_tabero_rgb_state`
及该基线自己的 checkpoint，并在机器人端加 `--no-tactile`。这不是把触觉模型的输入置零。

## 5. 机器人主机准备

复制 `examples/fr3_deploy/` 和 `packages/openpi-client/`；也可以直接复制当前仓库。
先 source 现场 ROS2 和相机/触觉驱动工作空间，再用 ROS2 对应的 Python 创建环境：

```bash
python3 -m venv --system-site-packages .venv-fr3
.venv-fr3/bin/python -m pip install -r examples/fr3_deploy/requirements.txt
.venv-fr3/bin/python -m pip install -e packages/openpi-client
.venv-fr3/bin/python -c 'import rclpy; from sensor_msgs.msg import Image; from std_msgs.msg import Bool'
```

不需要在机器人客户端安装 JAX、PyTorch 或 DM-Tac SDK。SDK/ROS 驱动仍由原采集环境运行。
已将你补充的 `dmtac_w_ipc.py` 原样放在部署目录，只调用其布局函数，不创建 mmap 或 SDK 进程。

修改 `examples/fr3_deploy/config.json`：

- `robot_url`：现场原 `franka_server` 地址；默认沿用附件 `http://192.168.1.10:5000`。
- `policy_url`：模型主机地址，例如 `ws://192.168.1.20:8000`。
- `front_topic`、`wrist_topic`：默认使用跨机ZED compressed话题和D405 raw RGB话题。
- 触觉默认 `/dmtac/left/packed_frame` 和 `/dmtac/right/packed_frame`，模式为 `shear_depth`。
- `limits.workspace_min/max`：默认复制附件中的范围，**应按实际夹具和操作空间设置**，不是现场碰撞模型。
- `prompt`：默认保留当前训练集的原始 task 文本。当前数据使用泛化的采集描述，并非精确任务指令；
  不要假设改成另一条自然语言任务就能获得该任务能力。

QoS按采集链路分别设置：ZED/D405为RELIABLE/depth1，DM-Tac packed保持
BEST_EFFORT/depth1，使能心跳为RELIABLE/depth1。DM-Tac packed发布端本身是
BEST_EFFORT；只把订阅端改成RELIABLE会造成QoS不兼容。

`robot_url`和`policy_url`可通过启动参数`--robot-url`、`--policy-url`覆盖，避免把
某台现场主机的地址提交到Git。最终生效配置会写入本次JSONL日志。

schema 3 的每侧数据为 `8UC1`、height=1、width=step=921600；前 614400 字节是
little-endian float32 `[240,320,2]` shear，其后为 depth。客户端通过附件布局函数解码，
不会把打包字节或完整 shear 展平数组直接塞进模型。
如驱动仅发布 shear 图像，可设置 `tactile_format: "shear"` 并将左右话题改为真实的 `32FC2` 话题。

触觉在独立的 10 Hz 采样线程更新：左右各抽取 9×11 网格，先左后右拼接，
当前坐标 = 固定参考网格 + shear_scale × shear，随后构造参考帧 + 8 帧历史。
启动时按转换器规则复制第一帧补齐历史；采样断档会清空旧历史。
不减去首次接触读数，不把无接触触觉写成全零坐标。

所有传感器和 HTTP state 必须带采集时间戳。代码核对各主机的 Unix 时间，
因此机器人服务器、相机发布主机和客户端需对时；不能比较不同主机的 monotonic 时间。
默认观测最老采集时间不得超过 250 ms，多模态时间差不超过 100 ms。
图像中重复的 header.stamp 不会刷新时效；缺失/停滞的时间戳会被拒绝。

触觉模型不会在marker缺失时自动补零。客户端采样后、WebSocket发送前和模型adapter入口都会
验证`float32 [9,198,2]`及有限值；服务端还会通过metadata与conversion文件核对shape、dtype、
参考帧+8帧历史布局和left-then-right顺序。任一不一致都会停止，不会继续推理。

## 6. 先运行 shadow

模型服务就绪后，在机器人主机仓库目录运行：

```bash
.venv-fr3/bin/python examples/fr3_deploy/run.py --seconds 30 --log deployment_logs/shadow_001.jsonl
```

若现场机器人HTTP地址为`172.31.179.19:5000`、模型主机为`192.168.1.20`，可直接覆盖：

```bash
.venv-fr3/bin/python examples/fr3_deploy/run.py \
  --robot-url http://172.31.179.19:5000 \
  --policy-url ws://192.168.1.20:8000 \
  --seconds 30 \
  --log deployment_logs/shadow_001.jsonl
```

该模式会读取真实状态、双相机及双触觉，首次调用用于模型编译并丢弃输出。
正式预测逐步写入 JSONL。每个新推理结果只写一条`inference_chunk`记录，包含唯一`chunk_id`、
模型看到的状态和完整`50×7`动作块；每个控制周期写一条`control_tick`记录，包含`action[0]`、
实际选中的`action[k]`、当前状态、限速目标和 HTTP payload。`distances`同时给出`action[0]`相对
观测/当前状态、`action[k]`相对当前状态/`action[0]`、当前状态相对观测状态以及与上一控制周期
目标之间的位置、SO(3)最短旋转角和单指开度差。结合`chunk_switched_since_previous_tick`可以区分
模型首步异常、chunk内部未来轨迹、shadow不运动以及新旧chunk切换不连续。
连接服务后还会写一条`policy_metadata`，记录服务实际使用的config、checkpoint、
`norm_stats_sha256`和`conversion_sha256`。这可以确认两次shadow是否加载了同一组checkpoint本地
归一化资产；哈希一致只证明文件一致，仍需结合训练数据来源确认该资产是否属于当前模型和数据转换。
触觉模型还会记录marker的shape、dtype及左右当前帧相对参考网格的平均/最大位移，便于确认真机触觉正在变化。
位置/姿态跳变、夹爪越界等预测会记录 `ok: false` 和原因；shadow 继续观察，不发送控制请求。
观测断流、模型通信错误和推理过期仍会终止程序。

检查日志里的坐标、姿态和夹爪趋势与实际场景相符，且不是持续被目标检查拒绝。
当前模型此前的离线结果存在较大姿态误差、夹爪超范围及 episode 起始标签问题；
部署接口写好并不意味着该 checkpoint 已具备真机任务成功率。
若 shadow 持续拒绝，先处理数据/模型误差，不要直接放宽跳变阈值来让它运动。

## 7. 手动启用有界执行

客户端使用独立的 `/tabero/enable` Bool 心跳。控制期间必须持续收到 true，
false 或超过 250 ms 无更新会结束本次运行；恢复心跳不会自动恢复已经退出的程序。
可以把 `enable_topic` 改为原 VR 原始 `/vr/right_controller/grip_button`，
前提是该原始节点持续发布按钮状态，且原 HTTP 遥操作桥已停止。
单次锁存的 true 消息不符合要求。

先在终端 A 启动短时执行：

```bash
.venv-fr3/bin/python examples/fr3_deploy/run.py --execute --seconds 10 --log deployment_logs/execute_001.jsonl
```

看到 `Waiting for fresh true heartbeat` 后再启用。若使用独立心跳话题，
可在终端 B 做人工监督的短时测试：

```bash
ros2 topic pub --rate 20 /tabero/enable std_msgs/msg/Bool '{data: true}'
```

这个命令会持续发布 true，不是按住才生效的实体使能按钮；停止终端 B 的发布进程才会让心跳过期。
正式操作应接到现场持续监测的使能输入。终端 A 的 Ctrl+C、时长到期或检查失败也会退出。
所有日志使用新文件名；脚本拒绝覆盖旧日志。

## 8. 时序、限速与停止的实际含义

模型输出 50 步，但默认只允许使用观测后前 5 个时刻的目标（最多 0.5 秒）。
推理线程和控制循环分开，控制目标最多以 10 Hz 发送，并持续用最新观测重新推理。
推理结果返回时按观测年龄选择索引：例如观测后 200 ms 的控制 tick 使用 `actions[2]`，
已经过去的 `actions[0:2]` 被丢弃，不会把整个 chunk 一口气发出去。
返回时年龄超过 350 ms、chunk 用完、控制 tick 超时或传感器陈旧都会结束执行。
这里是带延迟补偿的短 chunk 执行，并不声称实现了完整 RTC 或力控制。

默认初始限制：模型位置目标与实测差异不超过 5 cm、姿态差异不超过 0.35 rad；
限速后最多 2 mm/控制步、0.01 rad/控制步、夹爪总宽变化 2 mm/控制步。
姿态误差按 SO(3) 最短角度计算，避免轴角接近 ±π 时把同一物理姿态误判为巨幅旋转。
实际状态跟随误差另有限制，防止被物体阻挡后目标持续积累。
这些是客户端目标限制，不代替底层控制器的速度、加速度、力矩、碰撞或安全限制。

`/pose` 和 `/move_gripper` 是两个独立 HTTP 请求，不是原子操作。
任一失败立即退出且不重试，因为 timeout 时无法确定服务器是否已经执行。
如果已经尝试过运动请求，退出时会读取新的实测状态，尽力发送一次当前位置保持目标。
这只是一条普通 HTTP `/pose`，**不是硬件急停**；网络中断、排队请求、客户端崩溃时均不能保证生效。
已有夹爪运动也可能继续完成。提供的接口没有可确认的控制器停止/夹爪中止路由，
因此没有虚构 `/stop`；现场仍须保留底层 watchdog 和可用的硬件停止方式。
客户端不会自动 `/clearerr`、`/jointreset` 或开合夹爪来“恢复”运行。

## 9. 已做验证与未验证范围

CPU 测试覆盖接口单位、XYZW 四元数、±π 姿态、网格/历史顺序、真实 IPC pack/decode、
RGB 字节布局、前视裁剪、延迟索引、shadow 零写入、使能丢失、过期推理、
危险首目标、部分 HTTP 失败、实测状态时间戳及迁移后 checkpoint 统计加载。
还复用了已有 policy 的正反变换回归测试。

```bash
JAX_PLATFORMS=cpu .venv/bin/python -m pytest examples/fr3_deploy/test_deployment.py scripts/serve_tabero_test.py src/openpi/policies/tabero_offline_test.py -q
```

没有在本任务中加载训练模型做推理、占用 GPU、连接 ROS2 现场话题或发送真机指令。
另一台主机的网络、ROS Python ABI、实际发布话题、控制器 HTTP 行为和任务效果仍需现场验证。
