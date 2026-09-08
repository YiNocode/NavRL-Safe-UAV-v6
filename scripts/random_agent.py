"""Run a finite V5 sensor/shape smoke test without PPO."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

TASK_ID = "Isaac-UAV-NavRL-V5-GPU-Direct-v0"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=50)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main() -> None:
    import gymnasium as gym
    import navrl_uav_v5  # noqa: F401
    import torch
    from isaaclab_tasks.utils import parse_env_cfg

    print("[V5 smoke] imports complete", flush=True)
    cfg = parse_env_cfg(TASK_ID, device=args_cli.device, num_envs=args_cli.num_envs)
    print("[V5 smoke] configuration parsed; creating environment", flush=True)
    env = gym.make(TASK_ID, cfg=cfg)
    print("[V5 smoke] environment created", flush=True)
    try:
        observations, _ = env.reset()
        print("[V5 smoke] reset complete", flush=True)
        expected = {
            "static_obstacles": (args_cli.num_envs, 36, 7),
            "internal_state": (args_cli.num_envs, 8),
            "dynamic_obstacles": (args_cli.num_envs, 5, 10),
        }
        assert {name: tuple(value.shape) for name, value in observations.items()} == expected
        for _ in range(args_cli.steps):
            actions = torch.rand((args_cli.num_envs, 3), device=env.unwrapped.device)
            observations, rewards, terminated, truncated, _ = env.step(actions)
            assert all(torch.isfinite(value).all() for value in observations.values())
            assert torch.isfinite(rewards).all()
        state = env.unwrapped.get_perception_state()
        print(
            "[PASS] V5 sensor-only rollout: "
            f"shapes={expected}, dynamic_tracks={int(state['dynamic_valid'].sum())}, "
            f"voxel_map={state['voxel_map_memory_bytes'] / 1024**2:.1f} MiB, "
            "truth_policy_input=false, shield=false"
        )
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
