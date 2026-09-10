"""Deterministic evaluation for the V6 finite-front-depth navigation policy."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from isaaclab.app import AppLauncher

TASK_ID = "Isaac-UAV-NavRL-V6-Front-Depth-M1-v0"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--episodes", type=int, default=500)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seed", type=int, default=1001)
parser.add_argument("--output", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values))


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = round(fraction * (len(ordered) - 1))
    return float(ordered[index])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    import gymnasium as gym
    import torch
    import navrl_uav_v5  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
    from rsl_rl.runners import OnPolicyRunner

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if args.episodes <= 0 or args.num_envs <= 0:
        raise ValueError("episodes and num_envs must be positive")

    env_cfg = parse_env_cfg(TASK_ID, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    agent_cfg = load_cfg_from_registry(TASK_ID, "rsl_rl_cfg_entry_point")
    agent_cfg.seed = args.seed
    agent_cfg.device = args.device
    env = RslRlVecEnvWrapper(
        gym.make(TASK_ID, cfg=env_cfg), clip_actions=agent_cfg.clip_actions
    )
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device=agent_cfg.device)
    observations = env.get_observations()

    completed = successes = collisions = timeouts = out_of_bounds = 0
    episode_return = torch.zeros(args.num_envs, device=env.device)
    episode_steps = torch.zeros(args.num_envs, dtype=torch.int64, device=env.device)
    returns: list[float] = []
    lengths: list[int] = []
    final_goal_distances: list[float] = []
    path_lengths: list[float] = []
    minimum_depths: list[float] = []
    maximum_altitudes: list[float] = []
    mean_dynamic_tracks: list[float] = []
    truth_leak_seen = False

    try:
        while completed < args.episodes:
            with torch.inference_mode():
                actions = policy(observations)
                observations, rewards, dones, extras = env.step(actions)
            episode_return += rewards
            episode_steps += 1
            truth_leak_seen |= bool(extras["simulator_truth_policy_input"].any().item())
            for env_id in torch.where(dones)[0].tolist():
                if completed >= args.episodes:
                    break
                completed += 1
                successes += int(extras["success"][env_id].item())
                collisions += int(extras["collision"][env_id].item())
                timeouts += int(extras["timeout"][env_id].item())
                out_of_bounds += int(extras["out_of_bounds"][env_id].item())
                returns.append(float(episode_return[env_id].item()))
                lengths.append(int(episode_steps[env_id].item()))
                final_goal_distances.append(float(extras["final_goal_distance"][env_id].item()))
                path_lengths.append(float(extras["episode_path_length"][env_id].item()))
                minimum_depths.append(float(extras["episode_min_static_distance"][env_id].item()))
                maximum_altitudes.append(float(extras["episode_max_altitude_m"][env_id].item()))
                mean_dynamic_tracks.append(
                    float(extras["episode_mean_valid_dynamic_tracks"][env_id].item())
                )
                episode_return[env_id] = 0.0
                episode_steps[env_id] = 0

        result = {
            "task": TASK_ID,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "deterministic_policy": True,
            "seed": args.seed,
            "episodes": completed,
            "num_envs": args.num_envs,
            "success_rate": successes / completed,
            "collision_rate": collisions / completed,
            "timeout_rate": timeouts / completed,
            "out_of_bounds_rate": out_of_bounds / completed,
            "mean_return": _mean(returns),
            "mean_episode_steps": _mean([float(value) for value in lengths]),
            "mean_final_goal_distance_m": _mean(final_goal_distances),
            "mean_path_length_m": _mean(path_lengths),
            "median_path_length_m": float(statistics.median(path_lengths)),
            "mean_minimum_front_depth_m": _mean(minimum_depths),
            "p05_minimum_front_depth_m": _percentile(minimum_depths, 0.05),
            "minimum_front_depth_m": min(minimum_depths),
            "mean_maximum_altitude_m": _mean(maximum_altitudes),
            "maximum_altitude_m": max(maximum_altitudes),
            "soft_ceiling_violation_rate": sum(
                value > env_cfg.soft_flight_ceiling for value in maximum_altitudes
            ) / completed,
            "mean_valid_dynamic_tracks": _mean(mean_dynamic_tracks),
            "simulator_truth_policy_input": truth_leak_seen,
            "perception": "rtx_front_depth_and_gpu_temporal_tracking",
            "control": "direct_root_velocity",
        }
        print(json.dumps(result, indent=2))
        if args.output is not None:
            output = args.output.expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2) + "\n")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
