"""Action-conditioned safety reward for the V6 front-depth voxel task."""

from __future__ import annotations

import torch
from isaaclab.utils import configclass

from .v6_voxel_env import V6FrontDepthVoxelEnv, V6FrontDepthVoxelEnvCfg


@configclass
class V6FrontDepthVoxelRewardV3EnvCfg(V6FrontDepthVoxelEnvCfg):
    """Keep environment semantics fixed while replacing only reward shaping."""

    static_risk_pixel_fraction = 0.01
    static_safety_margin_m = 1.00
    dynamic_ttc_horizon_s = 2.00
    dynamic_min_closing_speed_mps = 0.05
    dynamic_top_risks = 2
    stall_progress_threshold_m = 0.004
    stall_safe_static_clearance_m = 1.25
    stall_safe_dynamic_clearance_m = 1.50

    reward_progress_weight = 5.0
    reward_velocity_weight = 0.0
    reward_static_safety_weight = 0.40
    reward_dynamic_safety_weight = 0.35
    reward_stall_weight = 0.02
    reward_action_weight = 0.005
    reward_smoothness_weight = 0.03
    reward_altitude_weight = 4.0
    reward_time_penalty = 0.015
    reward_goal_bonus = 40.0
    reward_collision_penalty = 50.0
    reward_out_of_bounds_penalty = 50.0


class V6FrontDepthVoxelRewardV3Env(V6FrontDepthVoxelEnv):
    """Reward progress while penalizing only actions that close on sensed hazards."""

    cfg: V6FrontDepthVoxelRewardV3EnvCfg

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._reward_component_names = (*self._reward_component_names, "stall")
        self._episode_reward_sums["stall"] = torch.zeros(self.num_envs, device=self.device)
        self._last_reward_components["stall"] = torch.zeros(self.num_envs, device=self.device)

    def _static_action_risk(self, command_velocity_g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        clearance = self._static_raw.flatten(1).clamp_max(self.cfg.camera_far_m)
        count = max(1, int(clearance.shape[1] * self.cfg.static_risk_pixel_fraction))
        nearest, indices = torch.topk(clearance, count, dim=1, largest=False)
        pixel_u = (indices % self.cfg.camera_width).to(clearance.dtype)
        pixel_v = (indices // self.cfg.camera_width).to(clearance.dtype)
        intrinsics = self._camera.data.intrinsic_matrices
        optical_right = (pixel_u - intrinsics[:, 0, 2:3]) / intrinsics[:, 0, 0:1]
        optical_down = (pixel_v - intrinsics[:, 1, 2:3]) / intrinsics[:, 1, 1:2]
        rays_g = torch.stack(
            (torch.ones_like(optical_right), -optical_right, -optical_down), dim=-1
        )
        rays_g /= torch.linalg.vector_norm(rays_g, dim=-1, keepdim=True).clamp_min(1.0e-6)
        closing_speed = (command_velocity_g[:, None, :] * rays_g).sum(dim=-1).clamp_min(0.0)
        clearance_scale = self.cfg.static_safety_margin_m - self.cfg.collision_radius
        proximity = (
            (self.cfg.static_safety_margin_m - nearest) / clearance_scale
        ).clamp(0.0, 1.0).square()
        normalized_closing = (closing_speed / self.cfg.max_velocity_xy).clamp(0.0, 1.0)
        return -(proximity * normalized_closing).mean(dim=1), nearest.mean(dim=1)

    def _dynamic_ttc_risk(self) -> tuple[torch.Tensor, torch.Tensor]:
        tracked = self._camera_dynamic_observation
        relative = tracked.positions_w - self._drone.data.root_pos_w[:, None, :]
        direction = relative / torch.linalg.vector_norm(
            relative, dim=-1, keepdim=True
        ).clamp_min(1.0e-6)
        relative_velocity = tracked.velocities_w - self._drone.data.root_lin_vel_w[:, None, :]
        closing_speed = -(relative_velocity * direction).sum(dim=-1)
        surface = self._dynamic_observation.surface_distances
        closing = closing_speed > self.cfg.dynamic_min_closing_speed_mps
        ttc = surface / closing_speed.clamp_min(self.cfg.dynamic_min_closing_speed_mps)
        ttc_risk = (
            (self.cfg.dynamic_ttc_horizon_s - ttc) / self.cfg.dynamic_ttc_horizon_s
        ).clamp(0.0, 1.0).square()
        normalized_closing = (closing_speed / self.cfg.max_velocity_xy).clamp(0.0, 1.0)
        risk = ttc_risk * normalized_closing
        risk = torch.where(tracked.valid & closing, risk, torch.zeros_like(risk))
        top_count = min(self.cfg.dynamic_top_risks, risk.shape[1])
        top_risk = torch.topk(risk, top_count, dim=1).values
        contributors = (top_risk > 0.0).sum(dim=1).clamp_min(1)
        aggregated = top_risk.sum(dim=1) / contributors
        nearest_valid = torch.where(
            tracked.valid, surface, torch.full_like(surface, self.cfg.dynamic_sensing_range)
        ).amin(dim=1)
        return -aggregated, nearest_valid

    def _get_rewards(self) -> torch.Tensor:
        self._update_perception()
        relative_goal = self._goal_pos_w - self._drone.data.root_pos_w
        progress = self._previous_goal_distance - self._current_goal_distance
        centered_action = 2.0 * self._actions - 1.0
        command_velocity_g = centered_action.clone()
        command_velocity_g[:, :2] *= self.cfg.max_velocity_xy
        command_velocity_g[:, 2] *= self.cfg.max_velocity_z
        previous_centered_action = 2.0 * self._previous_actions - 1.0
        static_safety, nearest_static = self._static_action_risk(command_velocity_g)
        dynamic_safety, nearest_dynamic = self._dynamic_ttc_risk()
        safe_to_advance = (
            (nearest_static >= self.cfg.stall_safe_static_clearance_m)
            & (nearest_dynamic >= self.cfg.stall_safe_dynamic_clearance_m)
            & (self._current_goal_distance > self.cfg.goal_threshold)
        )
        progress_deficit = (
            (self.cfg.stall_progress_threshold_m - progress)
            / self.cfg.stall_progress_threshold_m
        ).clamp(0.0, 1.0)
        stall = -torch.where(safe_to_advance, progress_deficit, torch.zeros_like(progress_deficit))
        position_local = self._drone.data.root_pos_w - self.scene.env_origins
        start_height = self._start_pos_w[:, 2] - self.scene.env_origins[:, 2]
        goal_height = self._goal_pos_w[:, 2] - self.scene.env_origins[:, 2]
        corridor_ceiling = torch.minimum(
            torch.maximum(start_height, goal_height) + self.cfg.altitude_corridor_tolerance,
            torch.full_like(goal_height, self.cfg.soft_flight_ceiling),
        )
        altitude_excess = (position_local[:, 2] - corridor_ceiling).clamp_min(0.0)
        components = {
            "progress": self.cfg.reward_progress_weight * progress,
            "velocity": torch.zeros_like(progress),
            "static_safety": self.cfg.reward_static_safety_weight * static_safety,
            "dynamic_safety": self.cfg.reward_dynamic_safety_weight * dynamic_safety,
            "action": -self.cfg.reward_action_weight * centered_action.square().sum(dim=-1),
            "smoothness": -self.cfg.reward_smoothness_weight
            * (centered_action - previous_centered_action).square().sum(dim=-1),
            "altitude": -self.cfg.reward_altitude_weight * altitude_excess.square(),
            "time": torch.full_like(progress, -self.cfg.reward_time_penalty),
            "goal": self.cfg.reward_goal_bonus * self._success.float(),
            "collision": -self.cfg.reward_collision_penalty * self._collision.float(),
            "out_of_bounds": -self.cfg.reward_out_of_bounds_penalty * self._out_of_bounds.float(),
            "stall": self.cfg.reward_stall_weight * stall,
        }
        reward = torch.stack(tuple(components.values())).sum(dim=0)
        for name, value in components.items():
            self._last_reward_components[name].copy_(value)
            self._episode_reward_sums[name] += value
        self._episode_return += reward
        self._episode_success.copy_(torch.maximum(self._episode_success, self._success.float()))
        self._episode_collision.copy_(torch.maximum(self._episode_collision, self._collision.float()))
        self._episode_timeout.copy_(torch.maximum(self._episode_timeout, self._time_out.float()))
        self._episode_out_of_bounds.copy_(
            torch.maximum(self._episode_out_of_bounds, self._out_of_bounds.float())
        )
        self._previous_goal_distance.copy_(self._current_goal_distance)
        self._previous_actions.copy_(self._actions)
        self.extras["reward_components"] = {
            name: value.clone() for name, value in components.items()
        }
        if self.cfg.debug_checks:
            assert torch.isfinite(reward).all()
        return reward


__all__ = ["V6FrontDepthVoxelRewardV3Env", "V6FrontDepthVoxelRewardV3EnvCfg"]
