# NavRL Safe UAV V5

V5 is an independent V2-based Isaac Lab task. It keeps V2's fast direct
root-velocity plant and replaces the old flat range observation with a
structured, sensor-only NavRL pipeline designed for large CUDA batches.

```text
Isaac/Warp depth-ray hits (sensor boundary)
        ↓
batched per-environment occupancy voxels on CUDA
        ↓
body-frame 3-D virtual ray casting [36 × 7]
        ↓                                  ↓
static CNN (128)             temporal depth residuals
                                      + GPU tracking
                                            ↓
                                  dynamic state [5 × 10]
        ↓                    internal/goal state [8]
        └──────────────────────────────┬───────────────
                                       ↓
                          NavRL CNN/MLP + Beta PPO
                                       ↓
                         direct root-velocity command
```

The PPO observation never contains obstacle transforms, velocities, sizes,
mesh IDs, semantic labels, the voxel grid, or raw depth. Ground-truth geometry
is private to scene motion and objective contact termination/evaluation.

## Deliberate V5 boundaries

- Control is the V2 direct velocity backend. There is no PX4 model.
- Perception runs in-process on PyTorch CUDA/Isaac Warp. There is no ROS2.
- There is no safety shield, velocity projection, deadlock override, or action
  replacement in training or evaluation. The executed command is exactly the
  bounded PPO command.
- V4's observation contract is retained: static virtual rays, eight internal
  states, five tracked dynamic states, CNN/MLP fusion, and a Beta actor.
- V4's single-environment CPU U-depth/DBSCAN/Kalman implementation is replaced
  with an ego-compensated temporal-depth detector, angular NMS, 3-D association,
  and alpha-beta filtering implemented as batched tensors. This is necessary
  for hundreds of parallel environments and is not a claim that GPU temporal
  differencing is numerically identical to the deployment CPU nodes.

## Environment

The registered task is `Isaac-UAV-NavRL-V5-GPU-Direct-v0`.

Defaults:

- 4096 parallel environments on the validated RTX 4070 (lower for smaller GPUs);
- 40 static and 15 active dynamic obstacles;
- every environment independently resamples its obstacle domain at every
  episode reset: 40 static templates are selected from a 48-shape pool and 15
  dynamic templates from a 24-template NavRL size/shape pool;
- every episode starts with the Crazyflie body origin 0.10 m above the ground;
  goals are sampled at 0.80--1.60 m, a NavRL-style endpoint-height corridor is
  reward-shaped, the soft ceiling is 1.80 m, and 2.20 m is a hard termination
  ceiling, preventing above-canopy shortcut policies;
- static obstacle templates are 2.60--5.00 m tall, so all static colliders
  remain above the hard ceiling and must be traversed horizontally;
- 100 Hz physics, 50 Hz policy;
- 20 × 20 × 5 m local map, 0.25 m voxels;
- 360° azimuth and ±30° elevation virtual scan, 5 m range;
- observation tensors `[N,36,7]`, `[N,8]`, and `[N,5,10]`.

The dense int32 occupancy timestamps use about 2.0 GiB at 4096 environments.
Old occupancy automatically expires through frame tokens, so the entire grid
is not cleared every step. Depth integration, ray traversal, motion detection,
tracking, CNN inference, rollout storage, and PPO updates remain on CUDA.

Runtime collider dimensions remain immutable after PhysX initialization.  V5
therefore randomizes obstacle dimensions by selecting a new subset from varied
pre-spawned templates, while positions, dynamic goals, and speeds are sampled
directly as CUDA tensors.  Inactive templates are moved below the map.  Start
and goal points are sampled only after the static layout and are rejected when
their collider clearance is below 1.25 m.  Isaac transforms are written before
the depth sensor, occupancy timestamps, motion differencer, and tracker are
reset, preventing stale observations from crossing episode boundaries.

## Install and validate

```bash
cd ~/Desktop/IsaacLab
./isaaclab.sh -i ~/Desktop/NavRL-Safe-UAV-v5/source/navrl_uav_v5

cd ~/Desktop/NavRL-Safe-UAV-v5
~/.conda/envs/isaaclab/bin/python -m pytest -q tests
```

Finite Isaac rollout:

```bash
cd ~/Desktop/IsaacLab
./isaaclab.sh -p ~/Desktop/NavRL-Safe-UAV-v5/scripts/random_agent.py \
  --headless --device cuda:0 --num_envs 32 --steps 50
```

Domain-randomization synchronization check:

```bash
cd ~/Desktop/IsaacLab
DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority \
~/.conda/envs/isaaclab/bin/python \
  ~/Desktop/NavRL-Safe-UAV-v5/scripts/validate_domain_randomization.py \
  --headless --device cuda:0 --num_envs 32
```

## Train

```bash
cd ~/Desktop/IsaacLab
./isaaclab.sh -p ~/Desktop/NavRL-Safe-UAV-v5/scripts/train.py \
  --headless --device cuda:0 --num_envs 4096 \
  --max_iterations 10000 --run_name v5_sensor_only
```

For W&B logging, append
`--logger wandb --wandb_project navrl-safe-uav-v5`.

GPU utilization should be tuned empirically. Benchmark 512, 1024, 2048, and
4096 environments; the best setting is the one with the highest
environment-steps/s without memory pressure, not necessarily the largest count.

## Evaluate and visualize

```bash
cd ~/Desktop/IsaacLab
./isaaclab.sh -p ~/Desktop/NavRL-Safe-UAV-v5/scripts/evaluate.py \
  --headless --device cuda:0 --checkpoint /absolute/path/model.pt \
  --num_envs 64 --episodes 500

./isaaclab.sh -p ~/Desktop/NavRL-Safe-UAV-v5/scripts/play.py \
  --device cuda:0 --checkpoint /absolute/path/model.pt --num_envs 1
```

Training and evaluation instantiate the same sensor-derived observation path;
there is no command-line option that can switch policy input to simulator truth
or enable an action shield.

## Important files

- `source/navrl_uav_v5/navrl_uav_v5/utils/gpu_navrl_perception.py`: batched
  occupancy, virtual rays, motion detection, and tracking.
- `source/navrl_uav_v5/navrl_uav_v5/tasks/navrl_navigation/navrl_env.py`: V2
  direct plant, scene, observation, reward, termination, and reset.
- `source/navrl_uav_v5/navrl_uav_v5/tasks/navrl_navigation/agents/`: structured
  NavRL actor/critic and PPO.
- `configs/default.yaml`: auditable runtime contract.
- `tests/`: deterministic sensor, coordinate, voxel, tracker, and architecture
  checks.
