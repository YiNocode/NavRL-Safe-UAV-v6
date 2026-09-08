"""Vectorized reward calculation for static UAV navigation."""

from __future__ import annotations

import torch


def compute_navigation_reward(
    previous_goal_distance: torch.Tensor,
    current_goal_distance: torch.Tensor,
    velocity_w: torch.Tensor,
    goal_direction_w: torch.Tensor,
    minimum_lidar_distance: torch.Tensor,
    actions: torch.Tensor,
    previous_actions: torch.Tensor,
    success: torch.Tensor,
    collision: torch.Tensor,
    *,
    out_of_bounds: torch.Tensor | None = None,
    mean_log_lidar_distance: torch.Tensor | None = None,
    height_band_error: torch.Tensor | None = None,
    shield_correction_norm: torch.Tensor | None = None,
    progress_weight: float = 2.0,
    velocity_weight: float = 0.2,
    safety_weight: float = 0.5,
    static_log_safety_weight: float = 0.0,
    height_weight: float = 0.0,
    shield_intervention_weight: float = 0.0,
    action_weight: float = 0.01,
    smoothness_weight: float = 0.05,
    time_penalty: float = 0.01,
    safe_distance: float = 1.0,
    goal_bonus: float = 20.0,
    collision_penalty: float = 20.0,
    out_of_bounds_penalty: float = 100.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute total reward and each signed, weighted contribution.

    Every returned component already has its configured sign and weight, so
    their elementwise sum is exactly the returned total reward.
    """
    batch_size = previous_goal_distance.shape[0]
    if current_goal_distance.shape != (batch_size,):
        raise ValueError("Goal-distance tensors must both have shape (N,)")
    if velocity_w.shape != (batch_size, 3) or goal_direction_w.shape != (batch_size, 3):
        raise ValueError("velocity_w and goal_direction_w must have shape (N, 3)")
    if actions.shape != (batch_size, 3) or previous_actions.shape != (batch_size, 3):
        raise ValueError("actions and previous_actions must have shape (N, 3)")
    if minimum_lidar_distance.shape != (batch_size,):
        raise ValueError("minimum_lidar_distance must have shape (N,)")
    if safe_distance <= 0.0:
        raise ValueError("safe_distance must be positive")

    optional_inputs = {
        "mean_log_lidar_distance": mean_log_lidar_distance,
        "height_band_error": height_band_error,
        "shield_correction_norm": shield_correction_norm,
    }
    for name, value in optional_inputs.items():
        if value is not None and value.shape != (batch_size,):
            raise ValueError(f"{name} must have shape (N,)")

    progress_raw = previous_goal_distance - current_goal_distance
    velocity_raw = torch.sum(velocity_w * goal_direction_w, dim=-1)
    safety_raw = ((minimum_lidar_distance - safe_distance) / safe_distance).clamp(-1.0, 0.0)
    action_raw = torch.sum(actions.square(), dim=-1)
    smoothness_raw = torch.sum((actions - previous_actions).square(), dim=-1)
    if out_of_bounds is None:
        out_of_bounds = torch.zeros_like(collision)
    elif out_of_bounds.shape != (batch_size,):
        raise ValueError("out_of_bounds must have shape (N,)")

    zeros = torch.zeros_like(current_goal_distance)
    if mean_log_lidar_distance is None:
        mean_log_lidar_distance = zeros
    if height_band_error is None:
        height_band_error = zeros
    if shield_correction_norm is None:
        shield_correction_norm = zeros

    components = {
        "progress": progress_weight * progress_raw,
        "velocity": velocity_weight * velocity_raw,
        "safety": safety_weight * safety_raw,
        # NavRL Eq. 9: average logarithmic static-ray clearance.
        "static_log_safety": static_log_safety_weight * mean_log_lidar_distance,
        # NavRL Eq. 12: quadratic penalty only outside the start/goal band.
        "height": -height_weight * height_band_error.square(),
        # During shield-aware refinement, discourage relying on projection.
        "shield": -shield_intervention_weight * shield_correction_norm,
        "action": -action_weight * action_raw,
        "smoothness": -smoothness_weight * smoothness_raw,
        "time": torch.full_like(current_goal_distance, -time_penalty),
        "goal": goal_bonus * success.to(dtype=current_goal_distance.dtype),
        "collision": -collision_penalty * collision.to(dtype=current_goal_distance.dtype),
        "out_of_bounds": -out_of_bounds_penalty * out_of_bounds.to(dtype=current_goal_distance.dtype),
    }
    total = torch.stack(tuple(components.values()), dim=0).sum(dim=0)
    return total, components
