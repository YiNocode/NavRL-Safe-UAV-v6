"""Evaluate a V5 checkpoint using the exact sensor-only training pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

TASK_ID = "Isaac-UAV-NavRL-V5-GPU-Direct-v0"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--episodes", type=int, default=100)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seed", type=int, default=1001)
parser.add_argument("--output", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main() -> None:
    import gymnasium as gym
    import navrl_uav_v5  # noqa: F401
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
    from rsl_rl.runners import OnPolicyRunner

    checkpoint = args_cli.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if args_cli.episodes <= 0 or args_cli.num_envs <= 0:
        raise ValueError("episodes and num_envs must be positive")
    env_cfg = parse_env_cfg(TASK_ID, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    agent_cfg = load_cfg_from_registry(TASK_ID, "rsl_rl_cfg_entry_point")
    agent_cfg.device = args_cli.device
    env = RslRlVecEnvWrapper(gym.make(TASK_ID, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device=agent_cfg.device)
    observations = env.get_observations()
    completed = successes = collisions = timeouts = out_of_bounds = 0
    returns: list[float] = []
    maximum_altitudes: list[float] = []
    episode_return = torch.zeros(args_cli.num_envs, device=env.device)
    try:
        while completed < args_cli.episodes:
            with torch.inference_mode():
                actions = policy(observations)
                observations, reward, dones, extras = env.step(actions)
            episode_return += reward
            finished = torch.where(dones)[0]
            for env_id in finished.tolist():
                if completed >= args_cli.episodes:
                    break
                completed += 1
                successes += int(extras["success"][env_id].item())
                collisions += int(extras["collision"][env_id].item())
                timeouts += int(extras["timeout"][env_id].item())
                out_of_bounds += int(extras["out_of_bounds"][env_id].item())
                returns.append(float(episode_return[env_id].item()))
                maximum_altitudes.append(float(extras["episode_max_altitude_m"][env_id].item()))
                episode_return[env_id] = 0.0
        result = {
            "checkpoint": str(checkpoint),
            "episodes": completed,
            "success_rate": successes / completed,
            "collision_rate": collisions / completed,
            "timeout_rate": timeouts / completed,
            "out_of_bounds_rate": out_of_bounds / completed,
            "mean_return": sum(returns) / len(returns),
            "takeoff_height_m": env_cfg.takeoff_start_height,
            "goal_height_range_m": list(env_cfg.goal_z_range),
            "soft_flight_ceiling_m": env_cfg.soft_flight_ceiling,
            "hard_flight_ceiling_m": env_cfg.termination_max_z,
            "maximum_altitude_observed_m": max(maximum_altitudes),
            "soft_ceiling_episode_violation_rate": sum(
                altitude > env_cfg.soft_flight_ceiling for altitude in maximum_altitudes
            ) / completed,
            "hard_ceiling_episode_termination_rate": sum(
                altitude > env_cfg.termination_max_z for altitude in maximum_altitudes
            ) / completed,
            "perception": "depth_sensor_to_gpu_voxel_raycast_motion_tracker",
            "simulator_truth_policy_input": False,
            "control": "direct_root_velocity",
            "ros2": False,
            "px4": False,
            "action_shield": False,
        }
        print(json.dumps(result, indent=2))
        if args_cli.output:
            args_cli.output.parent.mkdir(parents=True, exist_ok=True)
            args_cli.output.write_text(json.dumps(result, indent=2) + "\n")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
