# NavRL-Safe-UAV V6 正式训练技术基线

文档日期：2026-09-10  
目标分支：`v6-stereo-depth`  
服务器仓库：`/home/ubuntu/Desktop/NavRL-Safe-UAV-v5/NavRL-Safe-UAV-v5`  
Isaac Lab：`/home/ubuntu/Desktop/IsaacLab`（v2.3.2，commit `37ddf62`）

## 1. 状态与适用范围

本文是 V6 正式实验的配置事实来源，涵盖训练、仿真、机型、初始化、场景、传感器、观测和运行 gate。

V6 采用 `V6FrontDepthCameraEnv(NavRLGpuEnv)`：保留 V5 已验证的直接速度 plant、reward、termination、随机导航场景和 episode metrics，通过 perception hooks 将 V5 的 360° ray/voxel 感知替换为有限前视 RTX 深度和 GPU 时序动态跟踪。V5 默认 hook 保持原 ray/voxel 行为，避免破坏基线。

首个正式实验准确名称是 **simulated front-depth baseline**。当前只有一个 `TiledCamera` 深度 render product；`stereo_baseline_m=0.10` 是 M9 双目接口元数据，不代表已实现 left/right binocular reconstruction。

## 2. 事实来源

| 范围 | 文件 |
|---|---|
| V6 config、相机和 perception hook | `source/navrl_uav_v5/navrl_uav_v5/tasks/navrl_navigation/v6_camera_env.py` |
| 公共导航 mechanics | `source/navrl_uav_v5/navrl_uav_v5/tasks/navrl_navigation/navrl_env.py` |
| V5/公共环境参数 | `source/navrl_uav_v5/navrl_uav_v5/tasks/navrl_navigation/navrl_env_cfg.py` |
| 机型 | `source/navrl_uav_v5/navrl_uav_v5/assets/drone.py` |
| V6 runner | `source/navrl_uav_v5/navrl_uav_v5/tasks/navrl_navigation/agents/v6_rsl_rl_ppo_cfg.py` |
| PPO | `source/navrl_uav_v5/navrl_uav_v5/tasks/navrl_navigation/agents/rsl_rl_ppo_cfg.py` |
| static encoder | `source/navrl_uav_v5/navrl_uav_v5/tasks/navrl_navigation/agents/front_depth_encoder.py` |
| Actor/Critic | `source/navrl_uav_v5/navrl_uav_v5/tasks/navrl_navigation/agents/v6_actor_critic.py` |
| dynamic tracker | `source/navrl_uav_v5/navrl_uav_v5/utils/gpu_camera_motion_tracker.py` |
| runtime gate | `scripts/validate_v6_formal_gates.py` |
| scaling | `scripts/benchmark_scaling.py` |

当前 task ID 为 `Isaac-UAV-NavRL-V6-Front-Depth-M1-v0`；其中 `M1` 是历史遗留名称，不表示当前成熟度。为保持 M1–M7 命令和日志可追踪性，正式首轮暂不改名。

## 3. 软件环境

| 项目 | 配置 |
|---|---|
| OS | Ubuntu 22.04 |
| Isaac Sim | 5.1.0 |
| Isaac Lab | v2.3.2 / `37ddf62` |
| Python | 3.11 |
| PyTorch | 2.7.0+cu128 |
| RSL-RL | 3.1.2 |
| device | `cuda:0` |
| logger | TensorBoard |

依赖记录为 `environment_reproduced_20260908.txt`。编辑端不进入配置或运行路径；实验只使用上方服务器路径。

## 4. 训练配置

### 4.1 Runner

| 参数 | 值 |
|---|---:|
| seed | 42 |
| default envs | 64 |
| `num_steps_per_env` | 32 |
| `max_iterations` | 10,000 |
| `save_interval` | 250 |
| experiment | `uav_v6_front_depth` |
| run | `formal_camera_navigation` |
| clip actions | 1.0 |
| actor/critic observations | front depth + internal + dynamic |
| observation normalization | off；depth encoder 显式处理 |

命令行可以用 `--num_envs`、`--max_iterations` 和 `--learning_rate` 覆盖配置。one-iteration smoke 必须显式写 `--max_iterations 1`，不得把 smoke 日志混入正式 run。

### 4.2 PPO

| 参数 | 值 |
|---|---:|
| algorithm | `NavRLPPO` |
| learning rate | `2e-5`, fixed |
| epochs / mini-batches | 2 / 8 |
| gamma / lambda | 0.99 / 0.95 |
| PPO clip | 0.1 |
| value coefficient | 1.0 |
| entropy coefficient | `1e-3` |
| desired KL | 0.005 |
| hard KL multiplier | 1.5 |
| minimum updates before KL stop | 1 |
| max gradient norm | 1.0 |

动作使用三维 Beta distribution，concentration scale 1.0。best metric 为 `Episode/success`，window 25，minimum delta 0.002；early stop patience 500、warmup 50、maximum drop 0.08、degradation patience 50。

## 5. 仿真配置

| 参数 | 值 | 含义 |
|---|---:|---|
| physics `dt` | 0.01 s | 100 Hz |
| decimation | 2 | 两个 physics step/action |
| policy frequency | 50 Hz | `step_dt=0.02 s` |
| render interval | 2 | 与 control 对齐 |
| camera update period | 0.0 s | 每个可用 render 更新 |
| episode length | 12.0 s | 约 600 policy steps |
| env spacing | 22.0 m | 隔离 20×20 m 地图 |
| replicate physics | true | batched environments |
| physics material | friction 1.0/1.0, restitution 0.0 |
| light | dome, intensity 2000, RGB 0.75 |

## 6. 机型配置

机型不修改，继续使用 `isaaclab_assets.CRAZYFLIE_CFG.copy()`：

| 项目 | 值 |
|---|---|
| asset | Isaac Lab official Crazyflie |
| prim | `/World/envs/env_.*/Drone` |
| gravity | disabled |
| contact sensors | enabled |
| collision/navigation radius | 0.30 m |
| control abstraction | direct root velocity；无电机、气动和姿态内环 |
| V5 asset default Z | 0.10 m |
| V6 spawn/reset Z | 1.50 m（V6 config copy 独立覆盖） |

质量、惯量、关节、USD collider 等未覆盖字段严格继承服务器 `37ddf62` 的 `CRAZYFLIE_CFG`，不得猜测。V6 的 1.50 m 高度覆盖不会改变 V5 的 0.10 m ground-takeoff baseline。

## 7. 初始化与环境配置

### 7.1 起终点与飞行范围

| 参数 | 值 |
|---|---|
| map X/Y | `[-10,10]` m |
| operating Z | `[0.10,2.10]` m |
| start Z | 1.50 m |
| goal Z | `[1.55,1.90]` m |
| start-goal distance | `[6,15]` m |
| endpoint obstacle clearance | 1.25 m |
| soft ceiling | 2.10 m |
| termination Z | `[0.05,2.40]` m |
| goal threshold | 0.50 m |

每次 reset 随机采样 start/goal，拒绝与静态几何冲突的候选；机体 yaw 指向该 episode 的 start→goal，root/joint velocity 清零。相机和视觉 tracker 同步 reset，保证首帧不继承前一 episode 的 temporal state。

### 7.2 障碍物分布

| 项目 | 静态 | 动态 |
|---|---:|---:|
| template pool | 48 | 24 |
| active/episode | 40 | 15 |
| seed | 53 | 401 |
| prim | `StaticObstacle_*` | `DynamicObstacle_*` |
| speed | 0 | `[0.45,1.25]` m/s |
| local goal range | — | `(4,4,1.5)` m |

静态 cuboid/cylinder 的模板尺寸由 deterministic seed 生成，每个 episode 独立选择 active subset 并在抖动网格上重新布置。动态障碍独立随机 start、local goal、speed 和 active subset，并持续在目标间运动。inactive templates 放置在 Z=-100 m。

## 8. 动作、奖励与终止

### 8.1 动作

Policy action `[N,3]` clamp 到 `[0,1]`，再中心化到 `[-1,1]`：

- goal-frame XY 最大速度：2.0 m/s；
- goal-frame Z 最大速度：1.0 m/s；
- `goal_frame_to_world()` 转为世界速度；
- `_apply_action()` 直接 `write_root_velocity_to_sim()`。

不存在 safety shield、action projector、PX4 或 ROS2。

### 8.2 Reward

| component | weight/penalty |
|---|---:|
| goal progress | 3.0 |
| velocity toward goal | 0.20 |
| front-depth static safety | 0.08 |
| visual-track dynamic safety | 0.08 |
| action magnitude | -0.005 |
| action smoothness | -0.03 |
| altitude corridor excess² | -4.0 |
| time/step | -0.01 |
| success | +25 |
| collision | -25 |
| out of bounds | -25 |

static safety 只使用当前有限前视 depth，不再假设 360° 可见性。动态 safety 只使用 camera temporal tracker 的输出。障碍物 simulator truth 仅用于场景运动、客观碰撞/终止和 gate audit，不进入 policy observation 或 perception reward。

### 8.3 Termination

- success：goal distance ≤0.50 m；
- collision：contact history force 超过 0.10，且超过 reset grace 10 steps；
- out of bounds：超出 XY map 或 Z `[0.05,2.40]`；
- timeout：达到 12 s 且未发生前三类 termination，作为 truncation。

Episode logs 包含 return、length、success、collision、timeout、out-of-bounds、goal distance、path length、max altitude、minimum perceived static distance、mean valid dynamic tracks 和各 reward component。

## 9. 前视深度配置

| 参数 | 值 |
|---|---|
| sensor | `isaaclab.sensors.TiledCamera` |
| key | `front_depth_camera` |
| prim | `/World/envs/env_.*/Drone/body/front_depth_camera` |
| data | `distance_to_image_plane` |
| tensor | `[N,96,160,1]`, float, CUDA, metres |
| near/far | 0.10/5.0 m |
| clipping behavior | none |
| focal/focus/aperture | 12.0/5.0/24.0 |
| nominal horizontal FOV | 90°；几何以 intrinsic matrix 为准 |
| offset | `(0.05,0,0.05)`, identity quaternion, world convention |
| render products | 1 |

同时使用 `intrinsic_matrices [N,3,3]`、`pos_w [N,3]` 和 `quat_w_ros [N,4]` 做 backprojection 与 ego compensation。无返回像素保持 non-finite，由 encoder 的 validity mask 处理。

## 10. Observation 与 Actor/Critic

| observation | shape | 下游 |
|---|---|---|
| `front_depth` | `[N,96,160,1]` | `FrontDepthEncoder → [N,128]` |
| `internal_state` | `[N,8]` | 直接融合 |
| `dynamic_obstacles` | `[N,5,10]` | flatten + MLP `50→128→64` |

Depth encoder 先构造 proximity + validity 两通道 `[N,2,96,160]`，再用四层 stride-2 CNN、adaptive pool `3×5`、Linear/ReLU/LayerNorm 得到 128。融合为 `128+64+8=200`，shared trunk 为 `200→256→256`，actor 输出三维 alpha/beta，critic 输出 scalar value。

V5 checkpoint 不兼容；V6 必须从头训练。

Dynamic tracker 的流程为 previous-depth reprojection、ego-motion compensation、temporal residual、3×3 dilation、NMS/top-5、3D backprojection、nearest association、velocity smoothing 和 lifecycle。参数为 threshold 0.015 m、association 0.75 m、smoothing 0.70、missed frames 10、NMS kernel 9。暂时离开 FOV 的轨迹最多保留 10 帧，之后清零失效。

## 11. Scaling 基线

M7 使用 96×160、单 render product、warm-up 20、benchmark 100、profile 10：

| envs | env FPS | step/s | device used MiB | peak allocated/reserved MiB |
|---:|---:|---:|---:|---:|
| 1 | 48.08 | 48.08 | 2574.9 | 27.8/42 |
| 8 | 373.14 | 46.64 | 2580.9 | 34.5/48 |
| 32 | 1454.25 | 45.45 | 2638.9 | 94.8/154 |
| 64 | 2684.89 | 41.95 | 2694.9 | 180.8/278 |
| 128 | 4589.65 | 35.86 | 2814.9 | 350.1/564 |
| 256 | 7125.51 | 27.83 | 3050.9 | 691.8/1102 |

所有规模无 OOM，稳定段显存无持续增长。拐点从 64→128 开始，256 边际收益进一步减弱。该表是 clean camera workload；正式随机场景接入后必须重新测 64/128，首个正式训练默认 64。

## 12. Gate 清单

### 12.1 已实现并纳入自动 gate

- [x] action clamp、goal-frame 映射、direct velocity execution；
- [x] 非零且 finite reward；
- [x] success、physical contact collision、out-of-bounds、timeout；
- [x] episode metrics 与 `Episode/success`；
- [x] start/goal/static/dynamic episode randomization；
- [x] camera depth/static 128/dynamic `[5,10]`/internal 8 contract；
- [x] policy observation 不接受 obstacle simulator truth；
- [x] tracker ego translation compensation、association 和 lifecycle 单元回归；
- [x] one-iteration PPO 接线；
- [x] 1/8/32/64/128/256 camera scaling。

### 12.2 服务器最终 PASS gate

```bash
cd /home/ubuntu/Desktop/NavRL-Safe-UAV-v5/NavRL-Safe-UAV-v5
pytest -q

/home/ubuntu/Desktop/IsaacLab/isaaclab.sh -p scripts/validate_v6_formal_gates.py \
  --num_envs 8 --motion_steps 8 --headless

/home/ubuntu/Desktop/IsaacLab/isaaclab.sh -p scripts/train.py \
  --task Isaac-UAV-NavRL-V6-Front-Depth-M1-v0 \
  --num_envs 32 --max_iterations 1 --headless

/home/ubuntu/Desktop/IsaacLab/isaaclab.sh -p scripts/benchmark_scaling.py \
  --task Isaac-UAV-NavRL-V6-Front-Depth-M1-v0 \
  --num_envs 64 --warmup_steps 20 --benchmark_steps 100 \
  --profile_steps 10 --camera_read_repetitions 50 --headless

git status --short
```

PASS 要求：31+ tests；runtime gate 报告 action/reward/四类 termination/randomization/metrics/truth-leak 全通过；PPO 完成一次 update 且 checkpoint finite；64-env 完整场景无 OOM/持续显存增长；Git 状态为空。

## 13. 实验记录与尚未扩展项

每个正式 run 必须保存 Git commit、完整 CLI、env/runner config dump、seed、env count、GPU、resolution、render products、日志时间、checkpoint SHA-256、TensorBoard 和 deterministic evaluation 汇总。

以下不是首个 clean baseline 的阻塞项，但在宣称 stereo sim-to-real 前必须完成：left/right 相机与同步标定、真实 stereo matching、视差无效域/遮挡、深度噪声、纹理/光照/外参 randomization、旋转 ego-compensation 专项回归，以及真实相机 resolution/FPS/baseline 的重新 benchmark。
