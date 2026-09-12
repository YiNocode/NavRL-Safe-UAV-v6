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
parser.add_argument("--task", default=TASK_ID)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--episodes", type=int, default=500)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seed", type=int, default=1001)
parser.add_argument("--output", type=Path, default=None)
parser.add_argument("--diagnostic_window_s", type=float, default=1.0)
parser.add_argument("--collision_attribution_distance_m", type=float, default=0.60)
parser.add_argument("--fov_edge_fraction", type=float, default=0.15)
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


def _mean_or_none(values: list[float]) -> float | None:
    return _mean(values) if values else None


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
    if args.diagnostic_window_s <= 0.0:
        raise ValueError("--diagnostic_window_s must be positive")
    if args.collision_attribution_distance_m <= 0.0:
        raise ValueError("--collision_attribution_distance_m must be positive")
    if not 0.0 < args.fov_edge_fraction < 0.5:
        raise ValueError("--fov_edge_fraction must lie in (0, 0.5)")

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent_cfg.seed = args.seed
    agent_cfg.device = args.device
    env = RslRlVecEnvWrapper(
        gym.make(args.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions
    )
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device=agent_cfg.device)
    observations = env.get_observations()
    raw = env.unwrapped

    diagnostic_steps = max(1, round(args.diagnostic_window_s / raw.step_dt))
    history_commands = torch.zeros(
        (args.num_envs, diagnostic_steps, 3), device=env.device
    )
    history_goal_distance = torch.zeros(
        (args.num_envs, diagnostic_steps), device=env.device
    )
    history_front_depth = torch.full(
        (args.num_envs, diagnostic_steps), env_cfg.camera_far_m, device=env.device
    )
    history_dynamic_distance = torch.full(
        (args.num_envs, diagnostic_steps), env_cfg.dynamic_sensing_range, device=env.device
    )
    history_front_edge = torch.zeros(
        (args.num_envs, diagnostic_steps), dtype=torch.bool, device=env.device
    )
    history_valid = torch.zeros_like(history_front_edge)
    history_step = 0

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
    outcome_diagnostics: dict[str, dict[str, list[float]]] = {
        outcome: {
            "window_goal_progress_m": [],
            "mean_command_speed_mps": [],
            "mean_forward_command_mps": [],
            "mean_abs_lateral_command_mps": [],
            "mean_abs_vertical_command_mps": [],
            "last_front_depth_m": [],
            "last_tracked_dynamic_surface_distance_m": [],
        }
        for outcome in ("success", "collision", "timeout", "out_of_bounds")
    }
    collision_attribution = {
        "tracked_dynamic_front_confirmed": 0,
        "tracked_dynamic_no_front_depth": 0,
        "front_fov_center_only": 0,
        "front_fov_edge_only": 0,
        "unobserved_or_outside_front_fov": 0,
    }
    timeout_behavior = {
        "hover_stall": 0,
        "maneuver_without_goal_progress": 0,
        "slow_progress_timeout": 0,
    }
    truth_leak_seen = False

    try:
        while completed < args.episodes:
            with torch.inference_mode():
                actions = policy(observations)
                slot = history_step % diagnostic_steps
                centered_actions = 2.0 * actions - 1.0
                history_commands[:, slot, :2] = centered_actions[:, :2] * env_cfg.max_velocity_xy
                history_commands[:, slot, 2] = centered_actions[:, 2] * env_cfg.max_velocity_z
                history_goal_distance[:, slot] = raw._current_goal_distance
                flat_depth = raw._static_raw.flatten(1).clamp_max(env_cfg.camera_far_m)
                minimum_depth, minimum_index = flat_depth.min(dim=1)
                history_front_depth[:, slot] = minimum_depth
                pixel_u = minimum_index % env_cfg.camera_width
                pixel_v = minimum_index // env_cfg.camera_width
                edge_u = round(env_cfg.camera_width * args.fov_edge_fraction)
                edge_v = round(env_cfg.camera_height * args.fov_edge_fraction)
                history_front_edge[:, slot] = (
                    (pixel_u < edge_u)
                    | (pixel_u >= env_cfg.camera_width - edge_u)
                    | (pixel_v < edge_v)
                    | (pixel_v >= env_cfg.camera_height - edge_v)
                )
                dynamic_surface = torch.where(
                    raw._dynamic_observation.valid,
                    raw._dynamic_observation.surface_distances,
                    torch.full_like(
                        raw._dynamic_observation.surface_distances,
                        env_cfg.dynamic_sensing_range,
                    ),
                )
                history_dynamic_distance[:, slot] = dynamic_surface.amin(dim=1)
                history_valid[:, slot] = True
                observations, rewards, dones, extras = env.step(actions)
                history_step += 1
            episode_return += rewards
            episode_steps += 1
            truth_leak_seen |= bool(extras["simulator_truth_policy_input"].any().item())
            for env_id in torch.where(dones)[0].tolist():
                if completed >= args.episodes:
                    break
                completed += 1
                terminal_success = bool(extras["success"][env_id].item())
                terminal_collision = bool(extras["collision"][env_id].item())
                terminal_timeout = bool(extras["timeout"][env_id].item())
                terminal_out_of_bounds = bool(extras["out_of_bounds"][env_id].item())
                successes += int(terminal_success)
                collisions += int(terminal_collision)
                timeouts += int(terminal_timeout)
                out_of_bounds += int(terminal_out_of_bounds)
                if terminal_success:
                    outcome = "success"
                elif terminal_collision:
                    outcome = "collision"
                elif terminal_timeout:
                    outcome = "timeout"
                else:
                    outcome = "out_of_bounds"

                valid_count = int(history_valid[env_id].sum().item())
                oldest_slot = (slot - valid_count + 1) % diagnostic_steps
                valid_history = history_valid[env_id]
                commands = history_commands[env_id, valid_history]
                command_speed = torch.linalg.vector_norm(commands, dim=-1)
                window_progress = (
                    history_goal_distance[env_id, oldest_slot]
                    - history_goal_distance[env_id, slot]
                ).item()
                diagnostics = outcome_diagnostics[outcome]
                diagnostics["window_goal_progress_m"].append(float(window_progress))
                diagnostics["mean_command_speed_mps"].append(float(command_speed.mean().item()))
                diagnostics["mean_forward_command_mps"].append(float(commands[:, 0].mean().item()))
                diagnostics["mean_abs_lateral_command_mps"].append(
                    float(commands[:, 1].abs().mean().item())
                )
                diagnostics["mean_abs_vertical_command_mps"].append(
                    float(commands[:, 2].abs().mean().item())
                )
                last_front_depth = float(history_front_depth[env_id, slot].item())
                last_dynamic_distance = float(history_dynamic_distance[env_id, slot].item())
                diagnostics["last_front_depth_m"].append(last_front_depth)
                diagnostics["last_tracked_dynamic_surface_distance_m"].append(
                    last_dynamic_distance
                )

                if terminal_collision:
                    front_close = last_front_depth <= args.collision_attribution_distance_m
                    dynamic_close = last_dynamic_distance <= args.collision_attribution_distance_m
                    if dynamic_close and front_close:
                        category = "tracked_dynamic_front_confirmed"
                    elif dynamic_close:
                        category = "tracked_dynamic_no_front_depth"
                    elif front_close and bool(history_front_edge[env_id, slot].item()):
                        category = "front_fov_edge_only"
                    elif front_close:
                        category = "front_fov_center_only"
                    else:
                        category = "unobserved_or_outside_front_fov"
                    collision_attribution[category] += 1
                if terminal_timeout:
                    mean_speed = float(command_speed.mean().item())
                    if window_progress < 0.10 and mean_speed < 0.25:
                        timeout_behavior["hover_stall"] += 1
                    elif window_progress < 0.10:
                        timeout_behavior["maneuver_without_goal_progress"] += 1
                    else:
                        timeout_behavior["slow_progress_timeout"] += 1
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
                history_valid[env_id] = False

        result = {
            "task": args.task,
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
            "failure_diagnostics": {
                "method": "sensor_attributed_not_contact_object_ground_truth",
                "diagnostic_window_s": args.diagnostic_window_s,
                "diagnostic_window_steps": diagnostic_steps,
                "collision_attribution_distance_m": args.collision_attribution_distance_m,
                "fov_edge_fraction": args.fov_edge_fraction,
                "collision_attribution_counts": collision_attribution,
                "collision_attribution_rates_among_collisions": {
                    name: count / collisions if collisions else 0.0
                    for name, count in collision_attribution.items()
                },
                "timeout_behavior_counts": timeout_behavior,
                "timeout_behavior_rates_among_timeouts": {
                    name: count / timeouts if timeouts else 0.0
                    for name, count in timeout_behavior.items()
                },
                "leadup_means_by_outcome": {
                    outcome: {
                        name: _mean_or_none(values) for name, values in metrics.items()
                    }
                    for outcome, metrics in outcome_diagnostics.items()
                },
            },
            "simulator_truth_policy_input": truth_leak_seen,
            "perception": (
                "rtx_front_depth_to_gpu_local_voxel_and_temporal_tracking"
                if "Voxel" in args.task
                else "rtx_front_depth_and_gpu_temporal_tracking"
            ),
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
