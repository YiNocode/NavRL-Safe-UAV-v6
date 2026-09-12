"""Risk-sensitive reward variant of the V6 finite-FOV voxel task."""

from __future__ import annotations

import torch
from isaaclab.utils import configclass

from .v6_voxel_env import V6FrontDepthVoxelEnv, V6FrontDepthVoxelEnvCfg


@configclass
class V6FrontDepthVoxelSafetyEnvCfg(V6FrontDepthVoxelEnvCfg):
    """Keep the voxel baseline configuration and change only reward shaping."""

    # Full-image averaging diluted small nearby obstacles.  Reward the lower
    # one-percent clearance tail and keep the response zero outside the margin.
    static_risk_pixel_fraction = 0.01
    static_safety_margin_m = 1.00
    dynamic_safety_margin_m = 1.25
    dynamic_ttc_horizon_s = 2.00
    dynamic_min_closing_speed_mps = 0.05

    reward_progress_weight = 3.0
    reward_velocity_weight = 0.10
    reward_static_safety_weight = 0.35
    reward_dynamic_safety_weight = 0.25
    reward_action_weight = 0.005
    reward_smoothness_weight = 0.03
    reward_altitude_weight = 4.0
    reward_time_penalty = 0.01
    reward_goal_bonus = 30.0
    reward_collision_penalty = 40.0
    reward_out_of_bounds_penalty = 40.0


class V6FrontDepthVoxelSafetyEnv(V6FrontDepthVoxelEnv):
    """Voxel task with sensor-only clearance-tail and tracked-TTC rewards."""

    cfg: V6FrontDepthVoxelSafetyEnvCfg

    def _static_risk_reward(self) -> torch.Tensor:
        clearance = self._static_raw.flatten(1).clamp_max(self.cfg.camera_far_m)
        count = max(1, int(clearance.shape[1] * self.cfg.static_risk_pixel_fraction))
        nearest = torch.topk(clearance, count, dim=1, largest=False).values
        denominator = self.cfg.static_safety_margin_m - self.cfg.collision_radius
        risk = (
            (self.cfg.static_safety_margin_m - nearest) / denominator
        ).clamp(0.0, 1.0).square().mean(dim=1)
        return -risk

    def _dynamic_risk_reward(self) -> torch.Tensor:
        tracked = self._camera_dynamic_observation
        relative = tracked.positions_w - self._drone.data.root_pos_w[:, None, :]
        direction = relative / torch.linalg.vector_norm(
            relative, dim=-1, keepdim=True
        ).clamp_min(1.0e-6)
        relative_velocity = (
            tracked.velocities_w - self._drone.data.root_lin_vel_w[:, None, :]
        )
        closing_speed = -(relative_velocity * direction).sum(dim=-1)
        surface = self._dynamic_observation.surface_distances
        clearance_denominator = (
            self.cfg.dynamic_safety_margin_m - self.cfg.collision_radius
        )
        clearance_risk = (
            (self.cfg.dynamic_safety_margin_m - surface) / clearance_denominator
        ).clamp(0.0, 1.0).square()
        closing = closing_speed > self.cfg.dynamic_min_closing_speed_mps
        ttc = surface / closing_speed.clamp_min(self.cfg.dynamic_min_closing_speed_mps)
        ttc_risk = (
            (self.cfg.dynamic_ttc_horizon_s - ttc) / self.cfg.dynamic_ttc_horizon_s
        ).clamp(0.0, 1.0).square()
        risk = torch.maximum(clearance_risk, torch.where(closing, ttc_risk, torch.zeros_like(ttc_risk)))
        risk = torch.where(tracked.valid, risk, torch.zeros_like(risk))
        return -risk.amax(dim=1)

    def _get_rewards(self) -> torch.Tensor:
        self._update_perception()
        relative_goal = self._goal_pos_w - self._drone.data.root_pos_w
        goal_direction = relative_goal / self._current_goal_distance.clamp_min(1.0e-6).unsqueeze(-1)
        progress = self._previous_goal_distance - self._current_goal_distance
        goal_velocity = torch.sum(self._drone.data.root_lin_vel_w * goal_direction, dim=-1)
        centered_action = 2.0 * self._actions - 1.0
        previous_centered_action = 2.0 * self._previous_actions - 1.0
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
            "velocity": self.cfg.reward_velocity_weight * goal_velocity,
            "static_safety": self.cfg.reward_static_safety_weight * self._static_risk_reward(),
            "dynamic_safety": self.cfg.reward_dynamic_safety_weight * self._dynamic_risk_reward(),
            "action": -self.cfg.reward_action_weight * centered_action.square().sum(dim=-1),
            "smoothness": -self.cfg.reward_smoothness_weight
            * (centered_action - previous_centered_action).square().sum(dim=-1),
            "altitude": -self.cfg.reward_altitude_weight * altitude_excess.square(),
            "time": torch.full_like(progress, -self.cfg.reward_time_penalty),
            "goal": self.cfg.reward_goal_bonus * self._success.float(),
            "collision": -self.cfg.reward_collision_penalty * self._collision.float(),
            "out_of_bounds": -self.cfg.reward_out_of_bounds_penalty * self._out_of_bounds.float(),
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


__all__ = ["V6FrontDepthVoxelSafetyEnv", "V6FrontDepthVoxelSafetyEnvCfg"]
