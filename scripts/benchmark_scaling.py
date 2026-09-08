"""Measure V5 CUDA environment throughput at one parallel environment count."""

from __future__ import annotations

import argparse
import json
import time

from isaaclab.app import AppLauncher

TASK_ID = "Isaac-UAV-NavRL-V5-GPU-Direct-v0"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--warmup_steps", type=int, default=20)
parser.add_argument("--benchmark_steps", type=int, default=100)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main() -> None:
    import gymnasium as gym
    import navrl_uav_v5  # noqa: F401
    import torch
    from isaaclab_tasks.utils import parse_env_cfg

    if min(args_cli.num_envs, args_cli.warmup_steps, args_cli.benchmark_steps) <= 0:
        raise ValueError("benchmark counts must be positive")
    cfg = parse_env_cfg(TASK_ID, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(TASK_ID, cfg=cfg)
    actions = torch.full((args_cli.num_envs, 3), 0.5, device=env.unwrapped.device)
    try:
        env.reset()
        for _ in range(args_cli.warmup_steps):
            env.step(actions)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(args_cli.benchmark_steps):
            env.step(actions)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        result = {
            "num_envs": args_cli.num_envs,
            "steps": args_cli.benchmark_steps,
            "wall_time_s": elapsed,
            "ms_per_policy_step": 1000.0 * elapsed / args_cli.benchmark_steps,
            "environment_steps_per_second": args_cli.num_envs * args_cli.benchmark_steps / elapsed,
            "voxel_map_mib": env.unwrapped._perception.map.memory_bytes / 1024**2,
            "cuda_peak_allocated_mib": (
                torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
            ),
        }
        print(json.dumps(result, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
