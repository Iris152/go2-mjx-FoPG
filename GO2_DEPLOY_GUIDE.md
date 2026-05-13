# GO2 Deploy Guide

这份说明配合新建的 `go2_unitree_sdk2_deploy.py` 使用，不会改动你现有训练和导出文件。

## 1. 这份 runner 做了什么

- 直接使用 `unitree_sdk2py` 读写 `rt/lowstate` / `rt/lowcmd`
- sim2sim 默认使用 `mujoco_menagerie/unitree_go2/scene_mjx.xml` 这份训练同源模型，同时保留 `rt/lowstate` / `rt/lowcmd` 的 Unitree 低层通信结构
- 复现你当前的两阶段策略链路：
  - baseline: `40 -> 12`
  - forward residual: `52 -> 12`
- 内置 joint remap：
  - 策略顺序：`FL, FR, RL, RR`
  - Unitree 低层顺序：`FR, FL, RR, RL`
- 实机模式下默认通过 `MotionSwitcher` 检查并释放 Go2 主运控服务 MCF，避免 `sport_mode` 等高层服务和低层 `LowCmd` 抢控制权
- 内置站立缓启动、策略渐入、姿态保护、退出时回站立再切 passive

## 2. 运行前准备

你需要在能运行你当前 ONNX 推理的 Python 环境里执行。当前系统自带的 `python3` 没有这些包，所以要先激活你自己的环境。

至少需要：

- `numpy`
- `onnxruntime`
- `unitree_sdk2py`

如果要先跑 sim2sim，需要先启动本目录的 `launch_unitree_mujoco_python_sim.py`。它默认不再打开 Unitree 官方 GO2 XML，而是打开训练同源的 menagerie/MJX XML。

## 3. 先在 menagerie 训练同源仿真里验证

### 3.1 启动 Terminal B：menagerie DDS 仿真

在 `MJX` 目录下运行：

```bash
python launch_unitree_mujoco_python_sim.py \
  --backend menagerie \
  --network lo \
  --domain_id 1
```

默认使用：

- XML: `mujoco_menagerie/unitree_go2/scene_mjx.xml`
- control mode: `position_servo`
- servo gain: `kp=230, kd=0.5`
- DDS: 发布 `rt/lowstate`，订阅 `rt/lowcmd`

如果只想无窗口检查链路：

```bash
python launch_unitree_mujoco_python_sim.py \
  --backend menagerie \
  --network lo \
  --domain_id 1 \
  --headless \
  --max_time 10
```

### 3.2 启动 Terminal A：策略 runner

另开一个终端，在 `MJX` 目录下运行：

```bash
python go2_unitree_sdk2_deploy.py \
  --network lo \
  --domain_id 1 \
  --auto_start \
  --auto_policy
```

如果你想手动分两步确认：

```bash
python go2_unitree_sdk2_deploy.py \
  --network lo \
  --domain_id 1
```

默认行为：

1. 连接 `rt/lowstate`
2. 等你按一次 Enter，开始站起
3. 站稳后再按一次 Enter，开始策略

如果只是想先看第一阶段原地 trot：

```bash
python go2_unitree_sdk2_deploy.py \
  --mode trot \
  --network lo \
  --domain_id 1
```

### 3.3 可选：回到 Unitree 官方 XML 做对照

这一步不是当前主线，只用于确认旧问题是否仍然来自 XML/执行器语义差异：

```bash
python launch_unitree_mujoco_python_sim.py \
  --backend unitree \
  --network lo \
  --domain_id 1
```

## 4. 再上真机

先按 Unitree 官方流程处理：

- 机器人上电后进入 `zero-torque`
- 手柄 `L2 + R2` 进入 `debug mode`
- 电脑网口设置为 `192.168.123.222/24`
- 确认连接机器狗的网卡名，比如 `enp5s0`

`go2_unitree_sdk2_deploy.py` 默认 `--release_mcf auto`：当 `--domain_id 0` 时会先调用 SDK2 `MotionSwitcher.CheckMode()` 查询当前运动服务；如果发现 `sport_mode` / `ai_sport` / `advanced_sport` 等服务仍然激活，会调用 `ReleaseMode()` 释放，直到 MCF 关闭后才进入低层控制。这个逻辑对应 Unitree `go2_stand_example.cpp` 的做法。

然后执行：

```bash
python go2_unitree_sdk2_deploy.py \
  --network enp5s0 \
  --domain_id 0
```

第一次真机不要加 `--auto_start --auto_policy`，先手动触发更稳。

如果只是做本地仿真，`--domain_id 1` 下会自动跳过 MCF 释放；如果实机上你已经手动确认 MCF 关闭，也可以显式加 `--release_mcf never`，但不建议作为默认做法。

## 5. 建议的验证顺序

1. menagerie DDS 仿真中只做站立，不开策略
2. menagerie DDS 仿真中开启 `trot` 模式
3. menagerie DDS 仿真中开启 `forward` 模式
4. 真机只做站立
5. 真机短时开启 `trot`
6. 真机短时开启 `forward`

## 6. 常用参数

只开 baseline：

```bash
python go2_unitree_sdk2_deploy.py --mode trot --network lo --domain_id 1
```

调小策略增益：

```bash
python go2_unitree_sdk2_deploy.py \
  --network lo \
  --domain_id 1 \
  --policy_kp 25 \
  --policy_kd 0.5
```

调长策略渐入时间：

```bash
python go2_unitree_sdk2_deploy.py \
  --network lo \
  --domain_id 1 \
  --policy_ramp_duration 2.0
```

手动控制 MCF 释放策略：

```bash
# 默认：domain_id=0 时释放，domain_id=1 时跳过
python go2_unitree_sdk2_deploy.py --network enp5s0 --domain_id 0 --release_mcf auto

# 强制释放，用于排查
python go2_unitree_sdk2_deploy.py --network enp5s0 --domain_id 0 --release_mcf always

# 跳过释放，仅在你已经确认 MCF 关闭时使用
python go2_unitree_sdk2_deploy.py --network enp5s0 --domain_id 0 --release_mcf never
```

## 7. 现在最值得先看的风险点

- joint remap 是否正确
- MCF 是否已经释放，不能让 `sport_mode` 等高层服务同时控制机器人
- `stand_kp/kd` 和 `policy_kp/kd` 是否偏硬
- IMU 倾角保护是否会过早触发
- 你当前策略是固定前进，不是手柄速度跟随

## 8. 和官方仓库的关系

官方参考：

- `unitree_rl_mjlab`: <https://github.com/unitreerobotics/unitree_rl_mjlab>
- `unitree_mujoco`: <https://github.com/unitreerobotics/unitree_mujoco>

这份 runner 没有直接复用 `unitree_rl_mjlab` 的 Go2 deploy controller，原因是你当前策略的 observation schema 和它默认的 Go2 deploy schema 不同。
