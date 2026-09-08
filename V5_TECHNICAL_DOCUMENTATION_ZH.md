# NavRL Safe UAV V5 技术文档

> 文档状态：与当前 V5 代码、配置和实验产物一致  
> 编写日期：2026-09-01  
> 项目根目录：`/home/tianxuli/Desktop/NavRL-Safe-UAV-v5`  
> Isaac Lab 任务 ID：`Isaac-UAV-NavRL-V5-GPU-Direct-v0`

## 目录

1. [项目概述](#1-项目概述)
2. [设计目标与系统边界](#2-设计目标与系统边界)
3. [系统总体架构](#3-系统总体架构)
4. [代码组织与模块职责](#4-代码组织与模块职责)
5. [运行环境与依赖](#5-运行环境与依赖)
6. [Isaac Sim 场景与无人机动力学](#6-isaac-sim-场景与无人机动力学)
7. [场景生成与逐回合随机化](#7-场景生成与逐回合随机化)
8. [传感器边界与坐标系](#8-传感器边界与坐标系)
9. [GPU 静态障碍物感知](#9-gpu-静态障碍物感知)
10. [GPU 动态障碍物感知](#10-gpu-动态障碍物感知)
11. [策略观测空间](#11-策略观测空间)
12. [NavRL Actor-Critic 网络](#12-navrl-actor-critic-网络)
13. [动作定义与控制执行](#13-动作定义与控制执行)
14. [奖励、终止与统计指标](#14-奖励终止与统计指标)
15. [PPO 算法与训练稳定机制](#15-ppo-算法与训练稳定机制)
16. [训练结果与最终模型](#16-训练结果与最终模型)
17. [独立森林泛化测试](#17-独立森林泛化测试)
18. [GPU 性能与内存](#18-gpu-性能与内存)
19. [测试和验证](#19-测试和验证)
20. [安装、训练、评估和可视化](#20-安装训练评估和可视化)
21. [可视化实现细节](#21-可视化实现细节)
22. [数据无泄漏与安全屏蔽审计](#22-数据无泄漏与安全屏蔽审计)
23. [已知限制](#23-已知限制)
24. [故障排查](#24-故障排查)
25. [面向 V6 和真实部署的接口](#25-面向-v6-和真实部署的接口)
26. [附录：关键参数和张量形状](#26-附录关键参数和张量形状)

---

## 1. 项目概述

V5 是一个独立的 Isaac Lab 无人机安全导航版本。它建立在 V2 的高速直接速度控制环境之上，保留 V4 的 NavRL 结构化策略输入思想，但移除了 ROS2、PX4 和 safety shield，并将感知链改写为适合数百至数千并行环境的 GPU 张量实现。

V5 的主要目标是回答以下问题：

1. 在不向策略泄漏 Isaac 障碍物真值的前提下，能否使用传感器深度返回构造静态和动态障碍物观测；
2. 能否让感知、环境、CNN/MLP、PPO rollout 和参数更新尽可能留在 GPU 上；
3. 在 40 个静态障碍物和 15 个动态障碍物的逐回合随机场景中，能否获得稳定的高成功率；
4. 已训练策略能否零样本迁移到不同形状和密度的几何森林场景。

V5 的策略数据路径为：

```text
Isaac/Warp 深度射线返回
        │
        ├── 静态路径：运动掩码剔除 → CUDA 占据体素 → 3D 虚拟射线
        │                                      ↓
        │                              S_static [N,36,7]
        │                                      ↓
        │                                  CNN → 128
        │
        └── 动态路径：时序深度残差 → GPU NMS → 3D 关联 → 速度滤波
                                               ↓
                                     S_dynamic [N,5,10]
                                               ↓
                                           MLP → 64

目标/速度内部状态 [N,8] ────────────────────────┐
                                               ↓
                                  特征拼接 [N,200]
                                               ↓
                                      共享 MLP 256-256
                                        ┌──────┴──────┐
                                        ↓             ↓
                                  Beta Actor       Critic Value
                                        ↓
                                  直接根速度控制
```

这里的“深度”指 Isaac `MultiMeshRayCaster`/Warp 返回的稀疏深度射线命中，不是 RGB 图像，也不是直接读取障碍物的位置、速度、尺寸或语义 ID。

---

## 2. 设计目标与系统边界

### 2.1 V5 实现的功能

- Isaac Sim/Isaac Lab 中真实 `cf2x.usd` Crazyflie 资产与 PhysX collider；
- 40 个活动静态障碍物和 15 个活动动态障碍物；
- 每个并行环境、每个 episode 独立随机化；
- 传感器深度到 GPU 体素地图、3D 虚拟射线和动态轨迹的完整策略观测链；
- 静态 CNN、动态 MLP、内部状态融合和 Beta 分布 PPO；
- 基于真实 PhysX 接触的碰撞终止；
- TensorBoard/W&B 日志、best checkpoint 和 early stopping；
- Isaac Sim 全局第三方视角、真实无人机、目标、轨迹及障碍物实时同步显示。

### 2.2 V5 明确不包含的功能

| 功能 | V5 状态 | 说明 |
|---|---:|---|
| ROS2 | 未使用 | 感知和策略在 Isaac 进程内运行 |
| PX4/SITL | 未使用 | 不模拟 PX4 飞控闭环 |
| Safety shield | 已删除/不调用 | 训练与评估均不替换 PPO 动作 |
| Velocity obstacle 投影 | 未使用 | 无动作安全投影 |
| Deadlock recovery override | 未使用 | 无脱困动作覆盖 |
| YOLO | 未使用 | 无语义目标检测器 |
| DBSCAN/KD-tree | 未使用 | 动态候选不通过 CPU 聚类产生 |
| 完整 Kalman filter | 未使用 | 使用关联后的指数速度滤波，不维护 Kalman 协方差 |
| RGB 图像输入 | 未使用 | 策略链只使用深度射线返回 |
| 原始深度直接输入 PPO | 未使用 | 深度先转换为虚拟射线和动态状态 |
| 完整体素地图输入 PPO | 未使用 | 策略仅接收压缩后的 `[36,7]` 距离矩阵 |
| Isaac 障碍物真值输入 PPO | 禁止 | 真值只用于场景更新、接触终止和评估标签 |

### 2.3 与 NavRL 的关系

V5 参考 NavRL 的策略侧观测契约：静态虚拟射线、有限数量的动态障碍物状态、无人机目标/速度状态、CNN/MLP 融合和 Beta 策略。V5 没有声称其动态感知数值上等价于 NavRL 部署端的 U-depth/DBSCAN/Kalman 节点。

原始 CPU 部署链包含连通域、KD-tree、Python 对象列表等不适合 4096 环境训练的步骤。V5 以时序深度残差、角域 NMS、GPU 3D 关联和张量速度滤波替代这些步骤，并保留策略接口 `[N,5,10]`。

---

## 3. 系统总体架构

V5 按照“仿真—感知—观测—策略—控制”分层：

```text
┌─────────────────────────────────────────────────────────────┐
│ Isaac Sim / PhysX                                           │
│ Crazyflie + StaticObstacle_* + DynamicObstacle_* + ground   │
└───────────────────────┬─────────────────────────────────────┘
                        │ MultiMeshRayCaster hit positions
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ GpuNavRLPerception                                          │
│ 1. metric depth and ray direction                           │
│ 2. ego-compensated temporal residual                        │
│ 3. motion NMS / top-k / 3D tracking                         │
│ 4. exclude dynamic hits from static map                     │
│ 5. frame-token occupancy integration                        │
│ 6. body-heading-aware 3D voxel ray casting                  │
└───────────────────────┬─────────────────────────────────────┘
                        │ structured observations
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ NavRLActorCritic                                            │
│ static CNN + dynamic MLP + internal state + shared trunk    │
│ Beta(alpha,beta) actor heads + scalar critic                │
└───────────────────────┬─────────────────────────────────────┘
                        │ action in [0,1]^3
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ Direct velocity plant                                      │
│ centered action → goal-frame velocity → world velocity      │
│ → write_root_velocity_to_sim                               │
└─────────────────────────────────────────────────────────────┘
```

关键解耦关系如下：

- 场景模块负责产生和移动 collider，不负责构造 PPO 输入；
- 感知模块只接受深度射线、无人机位姿和时间步信息；
- 环境模块负责将感知输出和内部状态组织为 observation dict；
- 网络模块不知道体素的构造方式，也不知道 Isaac 障碍物对象；
- PPO 模块只处理网络输出、rollout、优势和损失；
- 控制模块只解释策略动作，不修改感知结果。

---

## 4. 代码组织与模块职责

```text
NavRL-Safe-UAV-v5/
├── configs/default.yaml
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── play.py
│   ├── random_agent.py
│   ├── benchmark_scaling.py
│   ├── validate_domain_randomization.py
│   └── run_persistent_dense_training.sh
├── source/navrl_uav_v5/navrl_uav_v5/
│   ├── assets/drone.py
│   ├── tasks/navrl_navigation/
│   │   ├── navrl_env_cfg.py
│   │   ├── navrl_env.py
│   │   └── agents/
│   │       ├── static_obstacle_encoder.py
│   │       ├── navrl_actor_critic.py
│   │       ├── navrl_ppo.py
│   │       ├── rsl_rl_ppo_cfg.py
│   │       └── stability_runner.py
│   └── utils/
│       ├── gpu_navrl_perception.py
│       ├── navrl_dynamic.py
│       ├── navrl_scene.py
│       └── obstacles.py
└── tests/
    ├── test_gpu_navrl_perception.py
    └── test_v5_architecture.py
```

主要文件职责：

| 文件 | 运行时职责 |
|---|---|
| `navrl_env_cfg.py` | 仿真频率、场景、传感器、障碍物、观测、奖励和终止参数 |
| `navrl_env.py` | 场景创建、episode reset、动态物体更新、观测、奖励、终止和动作执行 |
| `gpu_navrl_perception.py` | GPU 体素、虚拟射线、时序运动检测、NMS、关联和速度滤波 |
| `obstacles.py` | 预生成带 PhysX collider 的静态/动态模板 |
| `navrl_dynamic.py` | 动态障碍物观测编码等工具；其中历史 guard/reward 工具不在 V5 主环境路径调用 |
| `static_obstacle_encoder.py` | 静态距离矩阵 CNN |
| `navrl_actor_critic.py` | 结构化观测编码、Beta actor 和 critic |
| `navrl_ppo.py` | ValueNorm、GAE/PPO 更新和硬 KL guard |
| `stability_runner.py` | 最佳模型、稳定性状态和早停 |
| `train.py` | 训练入口、恢复训练、W&B/TensorBoard 配置 |
| `evaluate.py` | 冻结模型的确定性多 episode 评估 |
| `play.py` | Isaac GUI 可视化、轨迹、全局视角和 Fabric/PhysX 变换同步 |

注意：仓库仍保留 `curriculum.py`、`navrl_scene.py` 中的通用采样函数，以及 `navrl_dynamic.py` 中的历史 action guard/reward 工具。代码检索确认当前 V5 主运行路径没有调用 curriculum、`apply_navigation_command_guard()` 或通用 `navrl_reward()`；实际奖励直接定义在 `navrl_env.py::_get_rewards()`。

---

## 5. 运行环境与依赖

当前验证机器的工具链快照如下：

| 项目 | 版本/规格 |
|---|---|
| 操作系统 | Ubuntu 22.04.3 LTS |
| Kernel | 6.8.0-136-generic |
| GPU | NVIDIA GeForce RTX 4070，12,282 MiB |
| NVIDIA Driver | 580.173.02 |
| Python | 3.11.15 |
| Isaac Sim | 5.1.0.0 |
| Isaac Lab 源码 | v2.3.2，commit `37ddf626...` |
| Isaac Lab Python 包 | 0.54.2 |
| Isaac Lab RL | 0.4.7 |
| PyTorch | 2.7.0+cu128 |
| CUDA Runtime | 12.8 |
| RSL-RL | 3.1.2 |
| Gymnasium | 1.2.1 |
| NumPy | 1.26.0 |
| TensorBoard | 2.21.0 |

依赖策略：

- Isaac Sim、Isaac Lab、PyTorch、CUDA 和 RSL-RL 使用已验证的 `isaaclab` Conda 环境；
- 项目自身以 editable、`--no-deps` 模式安装；
- 不让上游 Python 元数据重新安装或替换 Isaac Sim 已验证的 PyTorch/CUDA 组合；
- 文档中的依赖版本来自 `environment_info.txt`。该文件标题和旧安装路径沿用了早期版本名称，应把内容视为机器工具链快照，而不是 V5 包名定义。

---

## 6. Isaac Sim 场景与无人机动力学

### 6.1 时间尺度

| 参数 | 数值 |
|---|---:|
| PhysX 仿真步长 | `0.01 s`，100 Hz |
| Decimation | `2` |
| 策略/控制频率 | 50 Hz |
| Episode 时长 | 12 s |
| 最大策略步数 | 600 |
| 默认并行环境数 | 4096 |
| 环境间距 | 22 m |

### 6.2 Crazyflie 资产

V5 使用官方 `cf2x.usd` 资产和真实 PhysX collider，而不是显示用的代理方块。碰撞传感器绑定到：

```text
/World/envs/env_.*/Drone/body
```

无人机的 body 原点在 reset 时位于地面上方 `0.10 m`，避免 collider 在生成瞬间与地面互穿，同时呈现从地面起飞的行为。

V5 使用高层速度控制抽象：资产关闭重力，PPO 输出被解释为目标根速度并通过 `write_root_velocity_to_sim()` 写入 PhysX。该模型适合高速策略训练，但不等价于真实电机、桨叶、姿态内环和气动力学。

### 6.3 飞行空间

| 约束 | 数值 |
|---|---:|
| X 范围 | `[-10, 10] m` |
| Y 范围 | `[-10, 10] m` |
| 起点高度 | `0.10 m` |
| 目标高度 | `[0.80, 1.60] m` |
| 软高度上限 | `1.80 m` |
| 硬终止高度 | `[0.05, 2.20] m` |
| 起点—目标水平距离 | `[6, 15] m` |
| 成功半径 | `0.50 m` |

静态障碍物高度为 `2.60–5.00 m`，全部高于 `2.20 m` 的硬高度终止线。因此策略不能通过飞越林冠逃避障碍物，必须在低空进行水平绕行。

---

## 7. 场景生成与逐回合随机化

### 7.1 为什么使用模板池

PhysX 启动后直接修改 collider 尺寸并不稳定。V5 在 scene setup 时预生成不同形状和尺寸的模板，episode reset 时选择模板子集并重新放置，而不是重建 USD stage。

### 7.2 静态障碍物

| 项目 | 数值 |
|---|---:|
| 模板池 | 48 |
| 每个 episode 活动数量 | 40 |
| 随机种子基值 | 53 |
| 形状 | cuboid 与 cylinder 混合 |
| cuboid 宽/长 | `0.5–1.5 m` |
| cylinder 半径 | `0.25–0.75 m` |
| 高度 | `2.60–5.00 m` |
| 最小间隔 | `0.50 m` |
| 起点/目标净空 | `1.25 m` |

静态模板在 reset 时按每个环境独立选择，并放置在带随机扰动的格点布局中。未激活模板移动至地图下方，不参与有效场景。

### 7.3 动态障碍物

| 项目 | 数值 |
|---|---:|
| 模板池 | 24 |
| 每个 episode 活动数量 | 15 |
| 随机种子基值 | 401 |
| 形状/尺寸类别 | 8 类 cuboid/高柱体组合 |
| 代表宽度 | `0.25, 0.50, 0.75, 1.00 m` |
| 速度 | `0.45–1.25 m/s` |
| 局部活动范围 | `4 × 4 × 1.5 m` |
| 目标到达阈值 | `0.25 m` |
| 动态最小间隔 | `0.20 m` |
| 与静态障碍物净空 | `0.15 m` |

动态物体是带 PhysX collider 的 kinematic actor。每个物体朝当前目标移动，到达阈值后重新采样目标。策略不知道这些 actor 的真值位置、目标或速度。

### 7.4 Reset 顺序

一次 episode reset 的关键顺序是：

1. 选择新的活动静态模板并更新位置；
2. 选择新的动态模板，采样初始位置、目标和速度；
3. 将不活动模板移动到 `z=-100 m` 附近；
4. 在静态布局确定后采样 UAV 起点和目标，并检查 `1.25 m` collider 净空；
5. 将所有刚体变换写入 Isaac/PhysX；
6. 清除被 reset 环境的深度历史、占据时间戳和动态 tracker 状态；
7. 重新开始观测和控制。

该顺序避免上一 episode 的深度、体素和速度轨迹污染新场景。

---

## 8. 传感器边界与坐标系

### 8.1 深度传感器

环境使用 `MultiMeshRayCasterCfg`，目标 prim 为：

```text
{ENV_REGEX_NS}/StaticObstacle_.*
{ENV_REGEX_NS}/DynamicObstacle_.*
```

两类目标都启用 transform tracking。静态对象在单个 episode 内不动，但 reset 会重新放置，因此也必须刷新每个环境的 mesh transform。

传感器配置：

| 参数 | 数值 |
|---|---:|
| 水平 FOV | 360° |
| 水平角分辨率 | 10° |
| 水平射线数 `Nh` | 36 |
| 垂直 FOV | -30° 到 +30° |
| 垂直角分辨率 | 10° |
| 垂直通道数 `Nv` | 7 |
| 最大深度 | 5.0 m |
| 对齐方式 | body/base |

Isaac 原始射线布局在感知模块中显式重排为策略约定的 `[Nh,Nv] = [36,7]`。

### 8.2 坐标约定

V5 使用右手坐标系：

```text
世界/机体系：+X 前，+Y 左，+Z 上
方位角 φ：绕 +Z
俯仰/仰角 θ：向上为正
```

机体系射线方向为：

```math
d_b(\phi,\theta)=
[\cos\theta\cos\phi,\ \cos\theta\sin\phi,\ \sin\theta]^T
```

虚拟射线根据无人机当前 yaw 旋转到世界/地图方向：

```math
d_w = R_z(\psi) d_b
```

策略内部目标状态使用“固定目标坐标系”：该坐标系的 X 轴在 episode 开始时沿起点到目标的水平连线，之后不跟随 UAV yaw 改变。这样目标方向、速度和动态物体状态使用一致参考系。

---

## 9. GPU 静态障碍物感知

实现位于 `utils/gpu_navrl_perception.py`，主要类为 CUDA 占据地图、虚拟射线器和统一 `GpuNavRLPerception`。

### 9.1 深度到命中点

从 Isaac 获得每条射线的起点和世界命中点后，计算度量深度：

```math
r = \|p_{hit}-p_{origin}\|_2
```

无效值、NaN 或 infinity 被映射到最大距离 `5.0 m`。原始深度只在感知模块内部使用，不进入 PPO。

### 9.2 动态命中剔除

动态 detector 首先在当前深度帧上产生 motion mask。被判断为动态的射线不会写入静态占据地图，减少移动物体在静态地图中留下“鬼影”的风险。

### 9.3 占据体素地图

每个环境维护独立 int32 张量：

```text
grid shape = [N, 80, 80, 20]
map size   = 20 × 20 × 5 m
voxel size = 0.25 m
origin     = (-10, -10, 0) m，环境局部坐标
```

世界点先减去对应 `env_origin`，再转换为体素索引：

```math
i = \left\lfloor\frac{p_{local}-o_{map}}{v}\right\rfloor
```

其中 `v=0.25 m`。索引必须同时满足 X/Y/Z 边界才能写入。

V5 使用“帧时间戳体素”而不是每步把 2 GiB 网格清零：

- 每帧为环境分配一个新的 int32 frame token；
- 当前命中体素通过 batched scatter 写入该 token；
- 查询时只有 `grid[cell] == current_token` 才视为占据；
- 上一帧的体素自然过期；
- episode reset 时只清除被 reset 环境的状态。

因此当前实现是“逐帧体素化占据层”，不是跨帧累计的 SLAM 或持久全局地图。

### 9.4 3D 虚拟射线

虚拟射线参数：

```text
azimuth   = 0°, 10°, ..., 350°
elevation = -30°, -20°, ..., +30°
range samples = 0.25, 0.50, ..., 5.00 m
```

所有环境、方位角、仰角和距离采样由广播张量一次生成。采样点被转换为体素索引，并通过 batched gather 查询。每条射线返回第一个占据体素对应的距离。

原始输出规则：

```text
命中：首个占据体素距离
无命中：max_distance + no_hit_offset = 5.1 m
```

归一化规则：

```math
s_{norm}=\frac{clip(s_{raw},0,5.0)}{5.0}\in[0,1]
```

所以无命中在策略输入中为 `1.0`，但调试时仍可使用 `5.1 m` 的原始值区分“最大距离处命中”和“无命中”。最终静态观测：

```text
S_static.shape = [N, 36, 7]
S_static[i,a,e] = 第 i 个环境在方位 a、仰角 e 的最近静态障碍物归一化距离
```

---

## 10. GPU 动态障碍物感知

V5 不从 Isaac 直接读取动态物体位置、速度和尺寸作为策略输入。动态状态完全由连续深度帧和 UAV 自运动估计产生。

### 10.1 自运动补偿的时序残差

设上一帧深度为 `d_{t-1}`，当前深度为 `d_t`，传感器原点位移为 `Δp`，射线单位方向为 `u`。静态世界假设下，上一帧深度在当前时刻的近似预测为：

```math
\hat d_t^{static}=d_{t-1}-u^T\Delta p
```

残差：

```math
e_t=|d_t-\hat d_t^{static}|
```

仅在当前、上一帧深度均有效且已有历史时参与检测。运动阈值为：

```text
e_t >= 0.035 m
```

该补偿主要处理传感器平移。它不是完整光流、稠密 scene flow 或相机旋转重投影。

### 10.2 角域膨胀与 NMS

二值 motion mask 先经过 `3×3` max-pool 膨胀。候选分数为：

```math
score=e_t\cdot max(1-d_t/d_{max},0.05)
```

随后再次使用 `3×3` max-pool 进行角域局部极大值抑制（NMS），每个环境选择 top-5 候选。

NMS 的作用是抑制同一运动区域周围多个相邻射线的重复峰值，使有限的五个 track 槽更可能代表不同物体。它不是基于真实 2D bounding box IoU 的 YOLO NMS。

### 10.3 3D 位置和尺寸近似

候选射线命中点提供表面位置。根据水平角分辨率 `Δφ=10°` 和深度 `d` 估算角域宽度：

```math
w=clip\left(2d\tan(\Delta\phi/2),0.20,1.50\right)
```

候选尺寸近似为：

```text
[width, width, max(width, 0.50)]
```

候选中心从表面命中点沿射线向障碍物内部移动 `0.5 × width`。

因此 V5 的宽、高是快速角分辨率近似，不是真值 bounding box，也不是 DBSCAN 点云包围盒。其误差与距离、遮挡、目标方向、射线分辨率和运动残差质量相关，代码没有一个可对所有场景成立的固定厘米误差保证。

### 10.4 3D 数据关联

当前候选位置与上一帧 track 位置使用 batched `torch.cdist()` 计算欧氏距离。最近邻距离不超过 `1.25 m` 时视为匹配，否则初始化为新 track。

每个环境最多输出五个 track：

```text
track_position [N,5,3]
track_velocity [N,5,3]
track_size     [N,5,3]
track_valid    [N,5]
```

### 10.5 速度估计与滤波

对匹配 track：

```math
v_{meas}=clip\left(\frac{p_t-p_{t-1}}{\Delta t},-5,5\right)
```

速度滤波：

```math
v_t=0.70v_{t-1}+0.30v_{meas}
```

未匹配的新 track 初始速度为零。该滤波器是指数平滑/简化 alpha-beta 风格速度滤波，并非完整 Kalman filter：它不维护状态协方差、过程噪声、观测噪声或显式预测—校正矩阵。

### 10.6 静态与动态感知的关系

两类感知共享同一深度传感器，但输出和网络分支分离：

```text
同一深度帧
  ├─ motion mask ──→ 动态 top-k tracks ──→ [N,5,10]
  └─ 剔除 motion ──→ 静态体素/射线 ─────→ [N,36,7]
```

这不是把静态和动态障碍物混成一个向量；它们只在策略特征融合层汇合。

---

## 11. 策略观测空间

环境返回结构化 observation：

```python
obs = {
    "static_obstacles":  ...,  # [N, 36, 7]
    "internal_state":    ...,  # [N, 8]
    "dynamic_obstacles": ...,  # [N, 5, 10]
}
```

Actor 和 critic 使用相同的三组观测，不采用 asymmetric critic。

### 11.1 静态障碍物 `[N,36,7]`

- 数值范围 `[0,1]`；
- `0` 表示非常近；
- `1` 表示最大范围或无命中；
- 不在环境中提前 flatten，保持二维角域结构供 CNN 使用。

### 11.2 内部状态 `[N,8]`

| 索引 | 含义 | 表达/归一化 |
|---:|---|---|
| 0–2 | 目标单位方向 | 在固定起点—目标坐标系中 |
| 3 | 水平目标距离 | 除以 `15 m` 并裁剪到 `[0,1]` |
| 4 | 有符号垂直目标距离 | 除以 `2.15 m` 并裁剪到 `[-1,1]` |
| 5–7 | UAV 速度 | 世界速度旋转到目标坐标系，除以 `2 m/s` 并裁剪 |

内部状态计算和射线实现相互独立。

### 11.3 动态障碍物 `[N,5,10]`

每个有效 track 的 10 维编码：

| 索引 | 含义 |
|---:|---|
| 0–2 | 障碍物相对单位方向，目标坐标系 |
| 3 | 水平距离 |
| 4 | 有符号垂直距离 |
| 5–7 | 估计速度，目标坐标系 |
| 8 | 宽度类别：`width / 0.25 - 1` |
| 9 | 高度/形态标量 |

无效 track 行全部为零。当前动态 detector 没有语义类别，环境传入的 `perceived_columns` 全为 false，因此第 9 维来自估计尺寸的 Z 分量，而不是 Isaac 的 cylinder/cuboid 标签。

### 11.4 信息隔离

以下数据不进入 actor 或 critic：

- `object_pos_w`、`object_quat_w`；
- 障碍物真值速度、目标或尺寸；
- prim 路径、mesh ID、颜色或语义标签；
- raw depth；
- 完整体素网格；
- ContactSensor 输出；
- success/collision/timeout/OOB 标签。

---

## 12. NavRL Actor-Critic 网络

### 12.1 静态 CNN

输入：

```text
[B,36,7] → unsqueeze → [B,1,36,7]
```

网络：

```text
Conv2d(1, 8, kernel=3, padding=1) + ReLU
Conv2d(8,16, kernel=3, stride=(2,1), padding=1) + ReLU
Conv2d(16,32,kernel=3, stride=2, padding=1) + ReLU
AdaptiveAvgPool2d((4,2))
Flatten: 32 × 4 × 2 = 256
Linear(256,128) + ReLU + LayerNorm(128)
```

自适应池化避免把卷积 flatten 尺寸硬编码到特定输入分辨率。

### 12.2 动态 MLP

```text
[B,5,10] → flatten [B,50]
Linear(50,128) + LeakyReLU + LayerNorm
Linear(128,64) + LeakyReLU + LayerNorm
```

### 12.3 融合与共享主干

```text
static embedding  128
internal state      8
dynamic embedding  64
----------------------
policy input       200

Linear(200,256) + LeakyReLU + LayerNorm
Linear(256,256) + LeakyReLU + LayerNorm
```

### 12.4 Beta Actor

共享主干输出分别进入 `alpha` 和 `beta` 三维 head：

```math
\alpha=softplus(h_\alpha)+1+\epsilon
```

```math
\beta=softplus(h_\beta)+1+\epsilon
```

这保证 `α,β>1`，使初始分布更稳定。训练时从 Beta 分布采样，确定性推理时采用均值：

```math
a=\frac{\alpha}{\alpha+\beta}\in(0,1)^3
```

### 12.5 Critic

Critic 与 actor 共享三个 encoder 和 256 维主干，使用独立 `Linear(256,1)` value head。环境没有为 critic 提供额外真值信息。

---

## 13. 动作定义与控制执行

PPO 动作为三维 Beta 变量：

```text
a ∈ [0,1]^3
```

先中心化：

```math
u=2a-1\in[-1,1]^3
```

再映射为固定目标坐标系中的速度：

```text
vx_goal = 2.0 × u0 m/s
vy_goal = 2.0 × u1 m/s
vz      = 1.0 × u2 m/s
```

水平速度从目标坐标系旋转回世界坐标，随后写入 root velocity：

```text
policy action
  → center/scale
  → goal-frame velocity
  → world-frame velocity
  → write_root_velocity_to_sim
```

V5 没有 action shield、最近安全速度投影或规则型恢复器。因此在训练和评估中：

```text
executed action == bounded PPO action
```

这一点使策略性能可归因于感知、奖励和 PPO 本身，但也意味着碰撞不会被外部安全层兜底。

---

## 14. 奖励、终止与统计指标

### 14.1 实际奖励函数

以下是 `navrl_env.py::_get_rewards()` 的实际运行项。

#### 进度奖励

```math
r_{progress}=3.0(d_{t-1}-d_t)
```

#### 朝向目标的速度奖励

```math
r_{velocity}=0.20(v_t^T\hat g_t)
```

#### 静态安全奖励

对 36×7 个静态射线：

```math
r_{static}=0.08\;mean\left[\log\frac{clip(d_s,r_c,d_{max})}{d_{max}}\right]
```

其中碰撞半径 `r_c=0.30 m`。

#### 动态安全奖励

对最多五个有效动态 track 的表面距离：

```math
r_{dynamic}=0.08\;mean\left[\log\frac{clip(d_{surface},r_c,R_{sense})}{R_{sense}}\right]
```

动态感知范围 `R_sense=4.0 m`。

#### 动作和光滑性惩罚

```math
r_{action}=-0.005\|u_t\|_2^2
```

```math
r_{smooth}=-0.03\|u_t-u_{t-1}\|_2^2
```

#### 高度惩罚

期望上限取：

```math
z_{corridor}=min(max(z_{start},z_{goal})+0.20,1.80)
```

仅对超过该高度的部分惩罚：

```math
r_{altitude}=-4.0\;max(z-z_{corridor},0)^2
```

#### 时间与事件奖励

```text
每步时间惩罚       -0.01
到达目标           +25
PhysX 碰撞         -25
越界               -25
```

总奖励为上述项之和。

### 14.2 终止条件

| 事件 | 条件 |
|---|---|
| 成功 | 到目标距离 `<=0.50 m` |
| 碰撞 | ContactSensor 净接触力 `>0.10`，且已过 reset grace period |
| 越界 | X/Y 超过地图边界，或 `z<0.05 m`，或 `z>2.20 m` |
| 超时 | 600 个策略步仍未出现其他终止事件 |

起飞后的前 10 个策略步约 `0.20 s` 忽略短暂地面接触，避免初始滑擦被误判；之后持续接触仍是碰撞。成功与越界同一帧发生时，成功逻辑优先抑制越界标签。

### 14.3 训练日志指标

每个 episode 记录：

- return、episode length；
- success、collision、timeout、out_of_bounds；
- final goal distance、path length、maximum altitude；
- minimum perceived static distance；
- mean valid dynamic tracks；
- 各奖励分项；
- PPO value/surrogate/entropy、梯度范数、KL、学习率和吞吐率。

---

## 15. PPO 算法与训练稳定机制

### 15.1 PPO 超参数

| 参数 | 数值 |
|---|---:|
| 并行环境 | 4096（默认） |
| 每环境 rollout 长度 | 32 |
| 每迭代 transition | 131,072 |
| `gamma` | 0.99 |
| GAE `lambda` | 0.95 |
| PPO clip | 0.10 |
| 学习率 | `2e-5`，fixed |
| Learning epochs | 2 |
| Mini-batches | 8 |
| Value loss coefficient | 1.0 |
| Entropy coefficient | `1e-3` |
| Desired KL | 0.005 |
| Max gradient norm | 1.0 |
| 保存间隔 | 250 iterations |
| 最大迭代 | 10,000 |

### 15.2 ValueNorm 与 GAE

Value target 使用带去偏的指数移动 ValueNorm：

```text
beta = 0.995
epsilon = 1e-5
```

超时 bootstrap 在反归一化 value 单位中处理，然后再形成 GAE。Critic 使用 `delta=10` 的 Huber loss，降低少量极端 return 对更新的影响。

### 15.3 Beta PPO 和 KL 保护

PPO 使用 Beta 分布的精确 log probability 和 KL。硬 KL guard 阈值为：

```math
KL_{hard}=1.5\times desired\_kl=0.0075
```

每次更新至少应用一个 policy minibatch；一旦采样 KL 超过硬阈值，可提前停止本轮后续策略更新，限制分布漂移。

优化器包含 feature、actor 和 critic 参数组，使用同一 Adam 学习率。由于 Isaac CUDA observation buffer 会复用，rollout 存储前显式 clone observation，防止后续仿真步覆盖历史样本。

### 15.4 最佳模型和早停

`StabilityOnPolicyRunner` 监控 `Episode/success`：

| 参数 | 数值 |
|---|---:|
| Rolling window | 25 |
| 最小有效提升 | 0.002 |
| Warmup | 50 iterations |
| 无提升 patience | 500 iterations |
| 最大允许下降 | 0.08 |
| 退化 patience | 50 iterations |

产生的关键文件：

```text
model_best.pt
model_early_stop.pt
model_final.pt 或 model_interrupted.pt
stability_state.json
events.out.tfevents.*
env.yaml
agent.yaml
```

`train.py` 在正常完成时保存 `model_final.pt`；在 SIGINT/SIGTERM 被 Isaac 转换为退出时，`finally` 分支保存 `model_interrupted.pt`，便于继续训练。

---

## 16. 训练结果与最终模型

### 16.1 最终低空高障碍训练段

最终选定运行目录：

```text
logs/rsl_rl/uav_v5_navrl_gpu/
2026-09-01_12-20-38_v5_lowalt_tall_s40_d15_s46_e256_20260901_122034_seg1/
```

该训练段从已有最佳权重继续，在低起飞高度、较高静态障碍物和 40/15 障碍物配置下微调。TensorBoard 标量覆盖迭代 `2911–3766`。

稳定性结果：

| 项目 | 数值 |
|---|---:|
| 最佳 25 点滚动训练成功率 | `0.985506`（98.55%） |
| 最佳迭代 | 3266 |
| 早停迭代 | 3766 |
| 早停时滚动成功率 | `0.978403`（97.84%） |
| 早停原因 | 连续 500 iterations 无超过 0.002 的有效提升 |
| 单批次最高 `Episode/success` | 1.0 |
| 单批次最高 return | 146.572，迭代 3146 |
| 日志内最高 episode altitude | 1.509 m |

必须区分：`Episode/success=1.0` 是某一批完成 episode 的即时均值；98.55% 是 25 个训练记录的 rolling metric；两者都不是固定独立测试集上的置信区间。

### 16.2 最佳 checkpoint

```text
logs/rsl_rl/uav_v5_navrl_gpu/
2026-09-01_12-20-38_v5_lowalt_tall_s40_d15_s46_e256_20260901_122034_seg1/
model_best.pt
```

SHA-256：

```text
cd45da49d613f05dcef73a3ed16ea643a27995d34c9460b9796fc33579aa6949
```

在交付、评估和远程可视化前应使用 `sha256sum` 校验，避免误用较早 checkpoint。

---

## 17. 独立森林泛化测试

另有独立工程 `/home/tianxuli/Desktop/NavRL-Safe-UAV-v5-forest-eval`，使用同一个冻结 checkpoint 对程序化几何森林进行零样本评估。该测试不更新权重，不属于 V5 原训练分布。

森林模板池包含 96 个真实 kinematic PhysX collider：

- 48 个直立树干；
- 16 个倾斜树干；
- 16 个倒木/低枝；
- 16 个岩石。

800 个未见 seed episode 的结果：

| 静态 | 动态 | Episodes | 成功率 | 碰撞率 | 超时率 | OOB | 平均最小静态净空 | 最大高度 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 | 0 | 200 | 100.0% | 0.0% | 0.0% | 0.0% | 0.704 m | 1.445 m |
| 45 | 0 | 200 | 100.0% | 0.0% | 0.0% | 0.0% | 0.466 m | 1.710 m |
| 60 | 0 | 200 | 100.0% | 0.0% | 0.0% | 0.0% | 0.438 m | 1.627 m |
| 45 | 15 | 200 | 96.0% | 3.5% | 0.0% | 0.5% | 0.370 m | 1.895 m |

45 静态 + 15 动态场景的成功率 Wilson 95% 区间为 `92.31%–97.96%`。

几何同步验证：

- 最大 active object PhysX 位置误差：`0.0 m`；
- 最大四元数误差：`2.3842e-7`；
- reset 后模板位姿变化率：100%；
- 强制 Crazyflie—树干碰撞测试在第 21 步检测到接触。

该结果证明模型对几何形状和密度有较强泛化，但仍不是照片级森林测试：未覆盖叶片透明、风、变形植被、复杂地形、真实相机噪声和照明域偏移。

---

## 18. GPU 性能与内存

### 18.1 体素内存

单环境体素数量：

```text
80 × 80 × 20 = 128,000 cells
```

int32 每 cell 4 bytes：

```text
128,000 × 4 = 512,000 bytes ≈ 0.488 MiB/env
```

4096 环境仅体素时间戳约 `2000 MiB`。这还不包括深度、射线采样、track、PhysX、网络、rollout 和优化器内存。

### 18.2 环境 step benchmark

RTX 4070 12 GiB 上、不含 PPO 更新的实测：

| 环境数 | ms/策略步 | env-steps/s | 体素地图 | PyTorch 峰值 |
|---:|---:|---:|---:|---:|
| 256 | 13.44 | 19,045 | 125 MiB | 247 MiB |
| 512 | 15.26 | 33,553 | 250 MiB | 485 MiB |
| 1024 | 18.94 | 54,065 | 500 MiB | 956 MiB |
| 2048 | 21.06 | 97,227 | 1000 MiB | 1902 MiB |
| 4096 | 36.48 | 112,276 | 2000 MiB | 3795 MiB |

4096 环境、32 步、一次 PPO smoke test 处理 131,072 transitions，用时 5.62 s，即包含采集和优化约 23,307 transitions/s。

### 18.3 主要瓶颈

1. `[N,80,80,20]` int32 体素网格的线性显存增长；
2. `[N,Nh,Nv,Nrange]` 虚拟射线采样和 gather 的中间张量；
3. Isaac/Warp 多 mesh ray casting 和 transform tracking；
4. 4096×32 rollout storage 和 PPO backward；
5. GUI 模式中的 GPU 渲染与 Fabric→CPU 可视化同步。

训练脚本启用 TF32、cuDNN benchmark 和高精度 matmul 策略。最优环境数应通过吞吐与显存共同决定，不是越大越好。

---

## 19. 测试和验证

### 19.1 当前自动测试结果

执行：

```bash
cd /home/tianxuli/Desktop/NavRL-Safe-UAV-v5
/home/tianxuli/.conda/envs/isaaclab/bin/python -m pytest -q tests
```

当前结果：

```text
11 passed in 0.66s
```

测试覆盖：

- 体素坐标转换和边界；
- 前方障碍命中；
- 无障碍 no-hit；
- 正/负 elevation；
- UAV yaw 旋转；
- map boundary；
- 动态 detector 和 track shape；
- structured observation shape；
- CNN/Actor-Critic 输出 shape；
- 不允许 action shield/真值观测进入 V5 架构。

旧 `V5_IMPLEMENTATION_REPORT.md` 记录为 9 个 simulator-free tests；当前仓库实际测试集已扩展到 11 项，本文件采用本次重新运行结果。

### 19.2 Isaac 集成验证

已完成的集成检查包括：

- 32 环境有限步随机 rollout；
- 观测 shape `[32,36,7]`、`[32,8]`、`[32,5,10]`；
- rollout 中出现有效动态 track；
- 32 环境 PPO 单迭代；
- 4096 环境 PPO 单迭代且无 OOM；
- domain randomization 与感知 reset 同步检查；
- 可视化中渲染变换与 PhysX 变换误差断言。

### 19.3 建议的回归门槛

任何感知、场景或网络改动后，至少按以下顺序验证：

1. `pytest -q tests`；
2. 32 环境、50 步 `random_agent.py`；
3. 32 环境 domain randomization 检查；
4. 32 环境 PPO smoke test；
5. 512/1024 环境 benchmark；
6. 再开始长时训练；
7. 用完全未见 seed 做独立评估。

---

## 20. 安装、训练、评估和可视化

以下命令假定 Isaac Lab 位于 `/home/tianxuli/Desktop/IsaacLab`。

### 20.1 安装 V5 扩展

```bash
cd /home/tianxuli/Desktop/IsaacLab
./isaaclab.sh -i /home/tianxuli/Desktop/NavRL-Safe-UAV-v5/source/navrl_uav_v5
```

若上游元数据会替换已验证的 CUDA/PyTorch 版本，使用：

```bash
/home/tianxuli/.conda/envs/isaaclab/bin/python -m pip install --no-deps -e \
  /home/tianxuli/Desktop/NavRL-Safe-UAV-v5/source/navrl_uav_v5
```

### 20.2 单元测试

```bash
cd /home/tianxuli/Desktop/NavRL-Safe-UAV-v5
/home/tianxuli/.conda/envs/isaaclab/bin/python -m pytest -q tests
```

### 20.3 有限环境 rollout

```bash
cd /home/tianxuli/Desktop/IsaacLab
./isaaclab.sh -p /home/tianxuli/Desktop/NavRL-Safe-UAV-v5/scripts/random_agent.py \
  --headless --device cuda:0 --num_envs 32 --steps 50
```

### 20.4 训练

```bash
cd /home/tianxuli/Desktop/IsaacLab
./isaaclab.sh -p /home/tianxuli/Desktop/NavRL-Safe-UAV-v5/scripts/train.py \
  --headless --device cuda:0 --num_envs 4096 \
  --max_iterations 10000 --run_name v5_sensor_only
```

W&B：

```bash
./isaaclab.sh -p /home/tianxuli/Desktop/NavRL-Safe-UAV-v5/scripts/train.py \
  --headless --device cuda:0 --num_envs 4096 \
  --max_iterations 10000 --logger wandb \
  --wandb_project navrl-safe-uav-v5 --run_name v5_wandb
```

### 20.5 从 checkpoint 恢复

```bash
./isaaclab.sh -p /home/tianxuli/Desktop/NavRL-Safe-UAV-v5/scripts/train.py \
  --headless --device cuda:0 --num_envs 4096 \
  --resume /absolute/path/model_best.pt \
  --learning_rate 1e-5 --reset_optimizer_on_resume \
  --max_iterations 2000 --run_name v5_finetune
```

如果希望保留外部评估得到的旧最佳模型，可同时传入：

```text
--initial_best_metric 0.96
--initial_best_checkpoint /absolute/path/model_best.pt
```

### 20.6 确定性评估

```bash
./isaaclab.sh -p /home/tianxuli/Desktop/NavRL-Safe-UAV-v5/scripts/evaluate.py \
  --headless --device cuda:0 \
  --checkpoint /absolute/path/model_best.pt \
  --num_envs 64 --episodes 500 --seed 1001 \
  --output /absolute/path/evaluation.json
```

评估输出包含成功、碰撞、超时、越界、平均 return、最高高度和运行边界声明。

### 20.7 全局第三方视角循环播放

```bash
cd /home/tianxuli/Desktop/IsaacLab
DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority \
./isaaclab.sh -p /home/tianxuli/Desktop/NavRL-Safe-UAV-v5/scripts/play.py \
  --device cuda:0 --checkpoint /absolute/path/model_best.pt \
  --num_envs 1 --single-episode --loop-episodes \
  --third-person-viewport --real-time --playback-speed 1.0 \
  --inter-episode-delay 2.0
```

如果需要每次重复同一场景，添加：

```text
--repeat-same-scene --seed 2027
```

如果希望每轮看到不同随机场景，不添加 `--repeat-same-scene`。

### 20.8 GPU benchmark

```bash
./isaaclab.sh -p /home/tianxuli/Desktop/NavRL-Safe-UAV-v5/scripts/benchmark_scaling.py \
  --headless --device cuda:0 --num_envs 1024 \
  --warmup_steps 20 --benchmark_steps 100
```

---

## 21. 可视化实现细节

`play.py` 的可视化不改变 RL 逻辑。其显示内容包括：

- 真实 `cf2x.usd` Crazyflie；
- 正常工作的 PhysX collider，但隐藏青色 collider guide；
- 红色目标球；
- 洋红色完整飞行轨迹；
- 静态和动态障碍物；
- 固定全局第三方相机。

全局相机相对 env-0 原点：

```text
eye    = (16, -19, 22) m
target = (0, 0, 2) m
```

### 21.1 为什么要同步 Fabric 变换

GPU PhysX/Fabric 场景通过 tensor API 更新对象位置时，普通 USD xform 不一定自动刷新到 viewport。若不手工同步，会出现“画面中的障碍物”和“传感器/碰撞中的障碍物”不在同一位置，视觉上像无人机穿过物体。

V5 每帧将 env-0 的以下真值变换只用于渲染同步：

```text
drone body_pos_w/body_quat_w
static object_pos_w/object_quat_w
dynamic object_pos_w/object_quat_w
```

同步写入 Fabric world-position/world-orientation attributes。该数据不进入 policy observation。

### 21.2 渲染—物理误差断言

第一次更新及每 50 帧验证：

```text
position tolerance    = 1e-4 m
quaternion tolerance  = 1e-4
```

四元数比较考虑 `q` 与 `-q` 表示同一旋转。误差超限时直接抛出异常，避免继续展示错误结果。

### 21.3 Episode 播放模式

- `--single-episode`：禁止终止后立即自动 reset，保留一个完整场景到终止；
- `--loop-episodes`：完整 episode 终止后暂停，然后 reset 并循环；
- `--repeat-same-scene`：每轮 reset 前重新设置 seed，复现同一场景；
- `--playback-speed 0.5`：半速播放；
- `--real-time`：以环境 50 Hz 控制周期限速；
- trajectory 每移动 `0.025 m` 添加一段线，episode reset 时清除。

---

## 22. 数据无泄漏与安全屏蔽审计

### 22.1 允许使用的真值

Isaac 真值仅用于：

1. 创建和移动 kinematic 障碍物；
2. 通过 ContactSensor 确定客观碰撞终止；
3. success/OOB 等评估标签；
4. GUI 中同步实际 PhysX pose，保证画面正确；
5. 生成起点/目标时进行 collider 净空检查。

### 22.2 禁止进入策略的真值

Actor/critic 不接收障碍物真值位置、速度、尺寸、形状、ID、类别或目标。环境代码通过 `GpuNavRLPerception.update()` 统一产生策略障碍物输入。

### 22.3 Safety shield 审计

V5 主环境不导入或调用：

```text
apply_navigation_command_guard
velocity-obstacle projection
deadlock recovery action override
```

即使 `navrl_dynamic.py` 为历史兼容保留了通用 guard 函数，架构测试和运行路径检索都确认其未被执行。训练、评估和可视化没有可打开 shield 的命令行选项。

---

## 23. 已知限制

### 23.1 感知限制

- 使用稀疏射线深度，不是实际 RGB-D 相机成像模型；
- 没有相机噪声、畸变、曝光、运动模糊、缺失深度或 rolling shutter；
- 动态检测以时序深度差为基础，对低相对速度、同向运动或完全遮挡目标可能漏检；
- ego compensation 主要处理平移，不是完整的 SE(3) 深度重投影；
- `3×3` 角域 NMS 和 top-5 可能合并相邻目标或丢弃第六个目标；
- 尺寸是角分辨率近似，不是真实 3D bounding box；
- 速度滤波不是完整 Kalman，不提供不确定度；
- 体素地图只表示当前帧静态命中，不是长期占据/未知/自由空间概率地图。

### 23.2 控制和动力学限制

- 直接根速度控制不包含 PX4、姿态内环和执行器延迟；
- 关闭重力的高层 plant 不能复现真实 Crazyflie 推力饱和和失速；
- 没有通信延迟、控制丢包、传感器时间戳不同步；
- 没有 safety shield，现实部署前必须增加独立安全层。

### 23.3 场景和评估限制

- 原训练障碍物仍是规则 cuboid/cylinder；
- 动态物体运动模式相对简单；
- 几何森林不包含叶片、风、变形、复杂地面或光照域偏移；
- 训练滚动成功率不能替代多 seed、固定协议的独立置信区间；
- 当前 collision label 依赖仿真 ContactSensor，现实系统需碰撞预防而不是事后标签。

### 23.4 计算限制

- 稠密体素为每环境约 0.488 MiB，扩展到更大地图或更细分辨率会迅速增长；
- `torch.cdist` 在当前 5-track 规模很小，但更大 track 数需更高效关联；
- GUI 每帧把 env-0 pose 拷贝到 CPU 仅适合展示，不能用于大规模训练。

---

## 24. 故障排查

### 24.1 Isaac Sim 与 Isaac Lab 上游依赖冲突

症状：安装项目后 PyTorch、CUDA 或 Isaac 包被 pip 替换，出现 ABI/import 错误。

处理：

```bash
/home/tianxuli/.conda/envs/isaaclab/bin/python -m pip install --no-deps -e \
  /home/tianxuli/Desktop/NavRL-Safe-UAV-v5/source/navrl_uav_v5
```

不要在已验证环境中重新安装 torch/CUDA。先确认：

```bash
/home/tianxuli/.conda/envs/isaaclab/bin/python -c \
  "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

### 24.2 CUDA OOM

优先降低 `--num_envs`：4096 → 2048 → 1024。体素地图对环境数线性增长。关闭 GUI、浏览器和其他 CUDA 进程，并使用 `nvidia-smi` 检查显存。

### 24.3 训练自动停止

检查运行目录：

```text
stability_state.json
model_early_stop.pt
model_interrupted.pt
```

- `status=early_stopped`：稳定机制正常触发；
- 存在 `model_interrupted.pt`：进程收到退出或异常；
- 无保存且系统日志显示 OOM kill：降低并行环境；
- 持久训练脚本可记录 SIGINT/SIGTERM/SIGHUP 来源和 `nvidia-smi dmon`。

### 24.4 动态 track 全为零

检查：

- 连续两帧是否存在有效深度；
- dynamic actor 是否在传感器 5 m 范围内；
- motion residual 是否超过 `0.035 m`；
- reset 后是否允许至少一帧建立历史；
- `Perception/mean_valid_dynamic_tracks` TensorBoard 指标；
- 是否误把动态 mesh 排除在 `MultiMeshRayCaster` target 之外。

### 24.5 画面中无人机穿过障碍物

先区分物理与渲染问题：

1. 查看 `[V5 RENDER SYNC]` 误差；
2. 确认位置/四元数误差都小于 `1e-4`；
3. 确认真实 Crazyflie collider prim 存在；
4. 查看 ContactSensor 是否触发；
5. 不要使用未同步 Fabric pose 的旧版 `play.py`。

### 24.6 GUI 太快或场景不停变化

- 使用 `--real-time --playback-speed 1.0`；
- 更慢使用 `--playback-speed 0.5`；
- 完整 episode 使用 `--single-episode`；
- 循环使用 `--single-episode --loop-episodes`；
- 固定同一场景再加 `--repeat-same-scene`。

---

## 25. 面向 V6 和真实部署的接口

V5 的优点是策略接口已与感知源解耦。只要维持以下 observation contract，后续可以替换传感器后端而不重写 PPO：

```text
static_obstacles  [B,36,7]
internal_state    [B,8]
dynamic_obstacles [B,5,10]
```

### 25.1 V6 可替换的感知后端

推荐 V6 保留 actor/critic，替换 `GpuNavRLPerception` 内部：

```text
真实 RGB-D / Isaac camera
  → 深度去噪与相机标定
  → SE(3) 运动补偿
  → GPU point cloud / occupancy
  → GPU connected components 或 voxel clustering
  → 3D bounding box
  → 带协方差的 Kalman/JPD association
  → 同样的 [B,36,7] 与 [B,5,10]
```

如果 V6 使用 DBSCAN/KD-tree，应采用 CUDA 实现或稀疏 voxel connected components，避免 Python 环境循环。

### 25.2 PX4/真实无人机控制边界

真实部署不应继续使用 `write_root_velocity_to_sim()`。建议：

```text
PPO [normalized action]
  → 速度/加速度限制
  → 独立 safety supervisor
  → ROS2/PX4 Offboard velocity setpoint
  → PX4 姿态/角速度/电机内环
```

应在 V5 直接 plant 与 PX4 SITL 之间加入：

- 控制延迟和低通；
- setpoint 丢失 failsafe；
- 最大加速度/jerk；
- 电池和推力裕度；
- collision imminent emergency brake；
- 独立于 RL 的高度和地理围栏。

### 25.3 Sim-to-real 验证顺序

1. Isaac camera 替换 ray depth，但维持观测维度；
2. 加入相机噪声、深度空洞和时延 randomization；
3. PX4 SITL/HIL 评估；
4. 在防护网、低速、无动态物体条件下真实测试；
5. 逐步增加静态密度和动态目标；
6. 每一步都保留独立 safety supervisor 和人工急停。

---

## 26. 附录：关键参数和张量形状

### 26.1 参数总表

| 类别 | 参数 | 数值 |
|---|---|---:|
| 仿真 | physics dt | 0.01 s |
| 仿真 | policy dt | 0.02 s |
| 仿真 | episode | 12 s / 600 steps |
| 场景 | 默认 envs | 4096 |
| 场景 | static active/pool | 40/48 |
| 场景 | dynamic active/pool | 15/24 |
| 场景 | static height | 2.60–5.00 m |
| 场景 | dynamic speed | 0.45–1.25 m/s |
| 地图 | size | 20×20×5 m |
| 地图 | voxel | 0.25 m |
| 射线 | azimuth | 360° / 10° |
| 射线 | elevation | 60° / 10° |
| 射线 | max/no-hit | 5.0/5.1 m |
| 动态 | motion threshold | 0.035 m |
| 动态 | association | 1.25 m |
| 动态 | smoothing | 0.70 old + 0.30 measured |
| 动态 | max tracks | 5 |
| 控制 | XY/Z max speed | 2.0/1.0 m/s |
| 碰撞 | body radius | 0.30 m |
| PPO | rollout | 32 steps/env |
| PPO | learning rate | 2e-5 |
| PPO | clip/KL | 0.10/0.005 |
| PPO | gamma/lambda | 0.99/0.95 |

### 26.2 单步张量流

以批量 `B=N` 为例：

```text
sensor ray hits                 [B, 7, 36, 3]（Isaac 原始角域约定）
metric depth                    [B, 7, 36]
reordered depth                 [B,36, 7]
motion mask                     [B,36, 7]
occupancy timestamps            [B,80,80,20]
virtual ray distances           [B,36, 7]
static CNN input                [B, 1,36,7]
static embedding                [B,128]
dynamic candidate/tracks        [B, 5,...]
dynamic observation             [B, 5,10]
dynamic flattened               [B,50]
dynamic embedding               [B,64]
internal state                  [B,8]
fused policy input              [B,200]
shared hidden                   [B,256]
alpha/beta                      [B,3] / [B,3]
action                          [B,3]
root velocity command           [B,6]（线速度 3 + 角速度 3）
value                           [B,1]
```

### 26.3 每个策略步的概念时序

```text
1. PPO 产生 [B,3] 动作
2. 环境更新动态 obstacle actor 目标和位姿
3. 动作转换为世界根速度并写入 Crazyflie
4. PhysX 执行 2 × 0.01 s 子步
5. MultiMeshRayCaster 获取新深度命中
6. GPU tracker 计算动态候选和速度
7. 动态 mask 从静态命中中剔除
8. 当前帧静态命中写入体素 token
9. 3D virtual ray caster 输出 [B,36,7]
10. 环境构造 [B,8] 和 [B,5,10]
11. 计算 reward、termination 和 episode metrics
12. observation/reward/done 写入 rollout
```

环境对同一仿真帧的观测、奖励和终止调用使用感知缓存，避免重复执行深度处理和射线遍历。

### 26.4 关键文件索引

- 环境配置：`source/navrl_uav_v5/navrl_uav_v5/tasks/navrl_navigation/navrl_env_cfg.py`
- 环境运行逻辑：`source/navrl_uav_v5/navrl_uav_v5/tasks/navrl_navigation/navrl_env.py`
- GPU 感知：`source/navrl_uav_v5/navrl_uav_v5/utils/gpu_navrl_perception.py`
- 场景模板：`source/navrl_uav_v5/navrl_uav_v5/utils/obstacles.py`
- 策略网络：`source/navrl_uav_v5/navrl_uav_v5/tasks/navrl_navigation/agents/navrl_actor_critic.py`
- PPO：`source/navrl_uav_v5/navrl_uav_v5/tasks/navrl_navigation/agents/navrl_ppo.py`
- 训练稳定机制：`source/navrl_uav_v5/navrl_uav_v5/tasks/navrl_navigation/agents/stability_runner.py`
- 训练配置：`source/navrl_uav_v5/navrl_uav_v5/tasks/navrl_navigation/agents/rsl_rl_ppo_cfg.py`
- 可视化：`scripts/play.py`
- 可审计配置摘要：`configs/default.yaml`

---

## 结论

V5 已实现一个高吞吐、sensor-only 的 NavRL 风格训练系统：障碍物策略输入来自 Isaac/Warp 深度射线，经 GPU 时序运动检测、当前帧体素化、3D 虚拟射线和结构化编码形成；PPO 不接收障碍物真值，也不依赖 ROS2、PX4 或 safety shield。最终低空高障碍训练段的最佳 25 点滚动训练成功率为 98.55%，冻结模型在独立 45 静态 + 15 动态几何森林测试中取得 96.0% 成功率。

该版本适合作为高速策略训练和几何泛化基线。它仍不是完整的现实部署栈；真实应用需要进一步替换相机模型、改进动态 3D 感知、引入 PX4/执行器动力学、时延与噪声随机化，并增加独立的安全监督器。
