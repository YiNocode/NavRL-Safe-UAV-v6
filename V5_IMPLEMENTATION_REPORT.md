# V5 implementation and validation report

## Scope

V5 is an independent package at `~/Desktop/NavRL-Safe-UAV-v5`. It uses the V2
direct root-velocity plant, the V4 structured NavRL actor/critic boundary, and
a new batched sensor-only perception implementation. ROS2, PX4, and all safety
shield modules, switches, reward terms, telemetry, and action overrides were
removed.

The runtime policy path is:

```text
Isaac/Warp ray depth -> CUDA occupancy voxels -> CUDA virtual rays -> CNN
                    \-> temporal residual/NMS/3-D tracker -> dynamic MLP
goal/velocity ---------------------------------------------> fusion -> PPO
```

Only ray hit tensors cross the simulator/perception boundary. Scene obstacle
transforms are used privately to move kinematic actors; contact is used only as
an objective terminal/evaluation label.

## GPU implementation

- One int32 timestamp voxel tensor per environment; no full-map clear per step.
- One batched scatter integrates all current static depth hits.
- One vectorized gather traverses every environment/ray/range sample.
- Temporal depth residuals, ego compensation, angular NMS, top-k selection,
  3-D association, and alpha-beta velocity filtering are tensor operations.
- No environment loop exists in the observation path.
- TF32 and cuDNN benchmarking are enabled for the policy network.

The V4 CPU U-depth/DBSCAN/Kalman nodes are not invoked in V5 because their
connected-component, KD-tree, and Python-list implementation restricted the
camera path to one environment. V5 retains their policy-side output contract,
but its CUDA temporal detector is an explicit high-throughput replacement, not
a numerically identical reimplementation. No YOLO or simulator semantic/mesh
ID is used for PPO input.

## Validation on RTX 4070 12 GiB

| Environments | ms/policy step | Environment steps/s | Voxel map | PyTorch peak |
|---:|---:|---:|---:|---:|
| 256 | 13.44 | 19,045 | 125 MiB | 247 MiB |
| 512 | 15.26 | 33,553 | 250 MiB | 485 MiB |
| 1024 | 18.94 | 54,065 | 500 MiB | 956 MiB |
| 2048 | 21.06 | 97,227 | 1000 MiB | 1902 MiB |
| 4096 | 36.48 | 112,276 | 2000 MiB | 3795 MiB |

The benchmark measures environment stepping without PPO. A separate 4096-env,
32-step, one-iteration PPO smoke test completed 131,072 transitions in 5.62 s
(23,307 steps/s including collection and optimization) without OOM. Therefore
4096 is the validated V5 default for this machine.

## Checks completed

- `compileall`: pass
- Ruff: pass
- Simulator-free unit/architecture tests: 9 passed
- 32-env Isaac sensor rollout: pass; structured shapes `[32,36,7]`, `[32,8]`,
  `[32,5,10]`, with dynamic tracks observed
- 32-env PPO iteration: pass
- 4096-env PPO iteration: pass
