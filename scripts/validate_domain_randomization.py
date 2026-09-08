"""Validate V5 per-environment scene randomization and sensor synchronization."""

from __future__ import annotations

import argparse
import json

from isaaclab.app import AppLauncher


TASK_ID = "Isaac-UAV-NavRL-V5-GPU-Direct-v0"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=32)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main() -> None:
    import gymnasium as gym
    import navrl_uav_v5  # noqa: F401
    import torch
    from isaaclab_tasks.utils import parse_env_cfg

    cfg = parse_env_cfg(TASK_ID, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(TASK_ID, cfg=cfg)
    try:
        first_obs, _ = env.reset()
        first_obs = {name: value.clone() for name, value in first_obs.items()}
        raw = env.unwrapped
        first = raw.get_randomized_scene_state()
        first_start = raw._start_pos_w.clone()
        first_goal = raw._goal_pos_w.clone()
        second_obs, _ = env.reset()
        second = raw.get_randomized_scene_state()

        static_count = second["static_active_mask"].sum(dim=1)
        dynamic_count = second["dynamic_active_mask"].sum(dim=1)
        if not torch.all(static_count == cfg.num_static_obstacles):
            raise RuntimeError(
                f"A reset environment does not contain exactly {cfg.num_static_obstacles} static obstacles"
            )
        if not torch.all(dynamic_count == cfg.num_dynamic_obstacles):
            raise RuntimeError(
                f"A reset environment does not contain exactly {cfg.num_dynamic_obstacles} dynamic obstacles"
            )
        static_heights = second["static_sizes"][:, 2]
        if static_heights.amin() <= cfg.termination_max_z:
            raise RuntimeError("A static obstacle can be overflown below the hard flight ceiling")
        if static_heights.amax() > cfg.voxel.map_size[2]:
            raise RuntimeError("A static obstacle exceeds the voxel-map height")

        static_changed = (
            (first["static_active_mask"] != second["static_active_mask"]).any(dim=1)
            | ((first["static_positions_local"] - second["static_positions_local"]).abs().amax(dim=(1, 2)) > 1.0e-4)
        )
        dynamic_changed = (
            (first["dynamic_active_mask"] != second["dynamic_active_mask"]).any(dim=1)
            | ((first["dynamic_positions_local"] - second["dynamic_positions_local"]).abs().amax(dim=(1, 2)) > 1.0e-4)
            | ((first["dynamic_goals_local"] - second["dynamic_goals_local"]).abs().amax(dim=(1, 2)) > 1.0e-4)
        )
        if not torch.all(static_changed):
            raise RuntimeError("Static layout failed to change for one or more environments")
        if not torch.all(dynamic_changed):
            raise RuntimeError("Dynamic domain failed to change for one or more environments")

        static_expected_w = second["static_positions_local"] + raw.scene.env_origins.unsqueeze(1)
        dynamic_expected_w = second["dynamic_positions_local"] + raw.scene.env_origins.unsqueeze(1)
        static_active = second["static_active_mask"]
        dynamic_active = second["dynamic_active_mask"]
        static_sync_error = torch.linalg.vector_norm(
            second["static_positions_w"][static_active] - static_expected_w[static_active], dim=-1
        ).amax()
        dynamic_sync_error = torch.linalg.vector_norm(
            second["dynamic_positions_w"][dynamic_active] - dynamic_expected_w[dynamic_active], dim=-1
        ).amax()
        if static_sync_error > 1.0e-4 or dynamic_sync_error > 1.0e-4:
            raise RuntimeError("Isaac geometry is not synchronized with reset-domain tensors")

        ids = torch.arange(raw.num_envs, device=raw.device)
        start_local = raw._start_pos_w - raw.scene.env_origins
        goal_local = raw._goal_pos_w - raw.scene.env_origins
        if not torch.allclose(
            start_local[:, 2],
            torch.full_like(start_local[:, 2], cfg.takeoff_start_height),
            atol=1.0e-5,
        ):
            raise RuntimeError("A UAV did not start at the configured ground take-off height")
        if not torch.all(
            (goal_local[:, 2] >= cfg.goal_z_range[0])
            & (goal_local[:, 2] <= cfg.goal_z_range[1])
        ):
            raise RuntimeError("A goal was sampled outside the low-altitude goal corridor")
        endpoint_clearance = torch.minimum(
            raw._minimum_static_clearance(start_local, ids),
            raw._minimum_static_clearance(goal_local, ids),
        )
        if endpoint_clearance.amin() <= cfg.endpoint_obstacle_clearance:
            raise RuntimeError("A randomized endpoint overlaps the static-obstacle clearance envelope")

        active_sizes = second["dynamic_sizes"].unsqueeze(0).expand(raw.num_envs, -1, -1)[dynamic_active]
        active_positions = second["dynamic_positions_local"][dynamic_active]
        # Initial dynamic centers are also required to remain outside the UAV
        # endpoint clearance envelope.  This test mirrors the conservative
        # bounding-radius criterion used by the GPU reset sampler.
        env_index = ids.unsqueeze(1).expand_as(dynamic_active)[dynamic_active]
        dynamic_radius = 0.5 * active_sizes[:, :2].amax(dim=1)
        endpoint_dynamic_distance = torch.minimum(
            torch.linalg.vector_norm(active_positions - start_local[env_index], dim=-1),
            torch.linalg.vector_norm(active_positions - goal_local[env_index], dim=-1),
        ) - dynamic_radius
        if endpoint_dynamic_distance.amin() <= cfg.endpoint_obstacle_clearance:
            raise RuntimeError("A randomized dynamic obstacle overlaps an endpoint")

        # Exercise the real Crazyflie/PhysX state for 0.4 s.  A moderate pure
        # vertical command must lift every vehicle clear of the ground without
        # consuming the initial contact grace as a collision termination.
        takeoff_actions = torch.full((raw.num_envs, 3), 0.5, device=raw.device)
        takeoff_actions[:, 2] = 0.75
        takeoff_collision = torch.zeros(raw.num_envs, dtype=torch.bool, device=raw.device)
        for _ in range(20):
            _, reward, terminated, truncated, extras = env.step(takeoff_actions)
            if not torch.isfinite(reward).all():
                raise RuntimeError("Non-finite reward during the take-off check")
            takeoff_collision |= extras["collision"]
            if torch.any(terminated | truncated):
                raise RuntimeError("An episode terminated during the 0.4 s take-off check")
        takeoff_final_z = (raw._drone.data.root_pos_w - raw.scene.env_origins)[:, 2]
        if torch.any(takeoff_final_z < cfg.takeoff_start_height + 0.15):
            raise RuntimeError("The Crazyflie failed to climb from its near-ground start")
        if torch.any(takeoff_collision):
            raise RuntimeError("PhysX reported a collision during nominal ground take-off")

        observation_change = {
            name: float((second_obs[name] - first_obs[name]).abs().mean().item())
            for name in first_obs
        }
        result = {
            "num_envs": raw.num_envs,
            "static_active_per_env": int(static_count[0].item()),
            "dynamic_active_per_env": int(dynamic_count[0].item()),
            "static_changed_fraction": float(static_changed.float().mean().item()),
            "dynamic_changed_fraction": float(dynamic_changed.float().mean().item()),
            "cross_env_static_layouts_differ": bool(
                (second["static_positions_local"][1:] - second["static_positions_local"][:1]).abs().sum(dim=(1, 2)).gt(0).all()
            ) if raw.num_envs > 1 else True,
            "minimum_endpoint_static_clearance_m": float(endpoint_clearance.amin().item()),
            "minimum_endpoint_dynamic_surface_distance_m": float(endpoint_dynamic_distance.amin().item()),
            "takeoff_height_min_m": float(start_local[:, 2].amin().item()),
            "takeoff_height_max_m": float(start_local[:, 2].amax().item()),
            "goal_height_min_m": float(goal_local[:, 2].amin().item()),
            "goal_height_max_m": float(goal_local[:, 2].amax().item()),
            "soft_flight_ceiling_m": cfg.soft_flight_ceiling,
            "hard_flight_ceiling_m": cfg.termination_max_z,
            "static_template_height_min_m": float(static_heights.amin().item()),
            "static_template_height_max_m": float(static_heights.amax().item()),
            "takeoff_height_after_0_4_s_min_m": float(takeoff_final_z.amin().item()),
            "takeoff_collision_count": int(takeoff_collision.sum().item()),
            "static_geometry_sync_max_error_m": float(static_sync_error.item()),
            "dynamic_geometry_sync_max_error_m": float(dynamic_sync_error.item()),
            "start_changed_fraction": float((first_start - raw._start_pos_w).abs().amax(dim=1).gt(1.0e-4).float().mean().item()),
            "goal_changed_fraction": float((first_goal - raw._goal_pos_w).abs().amax(dim=1).gt(1.0e-4).float().mean().item()),
            "observation_mean_abs_change": observation_change,
            "perception_reset_version": second["perception_reset_version"],
        }
        print("[PASS] V5 DOMAIN RANDOMIZATION + LOW-ALTITUDE TAKE-OFF", flush=True)
        print(json.dumps(result, indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
