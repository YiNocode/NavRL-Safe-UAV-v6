"""Source-faithful NavRL dynamic-obstacle state and reward helpers.

The formulas and tensor layout in this module follow the public NavRL Isaac
training implementation at commit 3725bcc2e7c1be4ecf1455d922299ae85042603a.
They deliberately keep simulation ground truth separate from the RGB-D
detector/tracker used by the real-robot deployment stack.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .math import world_to_goal_frame


@dataclass(frozen=True)
class DynamicObservation:
    """Policy state and collision/reward auxiliaries for the closest obstacles."""

    state: torch.Tensor
    indices: torch.Tensor
    valid: torch.Tensor
    positions_w: torch.Tensor
    velocities_w: torch.Tensor
    radii: torch.Tensor
    surface_distances: torch.Tensor
    collision: torch.Tensor


def navrl_internal_state(
    robot_position_w: torch.Tensor,
    goal_position_w: torch.Tensor,
    robot_velocity_w: torch.Tensor,
    start_to_goal_w: torch.Tensor,
    eps: float = 1.0e-6,
    *,
    distance_scale_xy: float | None = None,
    distance_scale_z: float | None = None,
    velocity_scale: float | None = None,
) -> torch.Tensor:
    """Return the eight-element state used by the public Isaac trainer.

    Layout: goal unit vector (3), horizontal goal distance (1), signed vertical
    goal distance (1), and robot velocity in the fixed start-to-goal frame (3).
    Optional positive scales normalize distance to ``[0,1]`` and signed
    distance/velocity to ``[-1,1]`` for the policy while preserving the public
    eight-element layout.
    """
    relative_goal_w = goal_position_w - robot_position_w
    distance = torch.linalg.vector_norm(relative_goal_w, dim=-1, keepdim=True)
    direction_w = relative_goal_w / distance.clamp_min(eps)
    direction_g = world_to_goal_frame(direction_w, start_to_goal_w)
    distance_xy = torch.linalg.vector_norm(relative_goal_w[..., :2], dim=-1, keepdim=True)
    distance_z = relative_goal_w[..., 2:3]
    velocity_g = world_to_goal_frame(robot_velocity_w, start_to_goal_w)
    if distance_scale_xy is not None:
        if distance_scale_xy <= 0.0:
            raise ValueError("distance_scale_xy must be positive")
        distance_xy = (distance_xy / distance_scale_xy).clamp(0.0, 1.0)
    if distance_scale_z is not None:
        if distance_scale_z <= 0.0:
            raise ValueError("distance_scale_z must be positive")
        distance_z = (distance_z / distance_scale_z).clamp(-1.0, 1.0)
    if velocity_scale is not None:
        if velocity_scale <= 0.0:
            raise ValueError("velocity_scale must be positive")
        velocity_g = (velocity_g / velocity_scale).clamp(-1.0, 1.0)
    return torch.cat((direction_g, distance_xy, distance_z, velocity_g), dim=-1)


def apply_navigation_command_guard(
    policy_velocity_w: torch.Tensor,
    robot_position_w: torch.Tensor,
    goal_position_w: torch.Tensor,
    env_origins_w: torch.Tensor,
    *,
    action_limit: float,
    goal_capture_distance: float = 0.0,
    goal_capture_gain: float = 1.0,
    geofence_half_extent: float | None = None,
    eps: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply odometry-only terminal capture and horizontal geofence guards.

    All inputs use the world frame and have shape ``(N, 3)``.  The function
    never queries obstacle geometry: it uses only the same robot pose and task
    goal available to a real ROS2 flight controller.  A non-positive
    ``goal_capture_distance`` and ``geofence_half_extent=None`` disable the
    respective guards.

    Returns the guarded velocity plus ``(N,)`` capture/geofence masks.
    """
    expected = policy_velocity_w.shape
    if policy_velocity_w.ndim != 2 or expected[-1] != 3:
        raise ValueError("policy_velocity_w must have shape (N,3)")
    for name, value in (
        ("robot_position_w", robot_position_w),
        ("goal_position_w", goal_position_w),
        ("env_origins_w", env_origins_w),
    ):
        if value.shape != expected:
            raise ValueError(f"{name} must match policy_velocity_w")
    if action_limit <= 0.0 or goal_capture_gain <= 0.0:
        raise ValueError("action_limit and goal_capture_gain must be positive")
    if goal_capture_distance < 0.0:
        raise ValueError("goal_capture_distance must be non-negative")
    if geofence_half_extent is not None and geofence_half_extent <= 0.0:
        raise ValueError("geofence_half_extent must be positive when enabled")

    guarded = policy_velocity_w.clone()
    relative_goal = goal_position_w - robot_position_w
    goal_distance = torch.linalg.vector_norm(relative_goal, dim=-1)
    capture_mask = (goal_capture_distance > 0.0) & (goal_distance <= goal_capture_distance)
    if torch.any(capture_mask):
        goal_direction = relative_goal / goal_distance.clamp_min(eps).unsqueeze(-1)
        capture_speed = (goal_capture_gain * goal_distance).clamp(max=action_limit)
        capture_velocity = goal_direction * capture_speed.unsqueeze(-1)
        guarded[capture_mask] = capture_velocity[capture_mask]

    geofence_mask = torch.zeros(policy_velocity_w.shape[0], dtype=torch.bool, device=policy_velocity_w.device)
    if geofence_half_extent is not None:
        local_position = robot_position_w - env_origins_w
        outside_high = local_position[:, :2] >= geofence_half_extent
        outside_low = local_position[:, :2] <= -geofence_half_extent
        outward = (outside_high & (guarded[:, :2] > 0.0)) | (outside_low & (guarded[:, :2] < 0.0))
        guarded[:, :2] = torch.where(outward, torch.zeros_like(guarded[:, :2]), guarded[:, :2])
        geofence_mask = torch.any(outward, dim=-1)

    return guarded, capture_mask, geofence_mask


def navrl_dynamic_observation(
    robot_position_w: torch.Tensor,
    start_to_goal_w: torch.Tensor,
    obstacle_positions_w: torch.Tensor,
    obstacle_velocities_w: torch.Tensor,
    obstacle_sizes: torch.Tensor,
    obstacle_is_column: torch.Tensor,
    active_count: int | None,
    *,
    num_closest: int = 5,
    sensing_range: float = 4.0,
    width_resolution: float = 0.25,
    robot_radius: float = 0.30,
    active_mask: torch.Tensor | None = None,
    eps: float = 1.0e-6,
) -> DynamicObservation:
    """Construct NavRL's ``N_d x 10`` dynamic-obstacle matrix.

    The closest obstacles are selected by horizontal distance. Tall column
    obstacles ignore vertical separation, matching lines 486 and 521--527 of
    the upstream Isaac environment.
    """
    if robot_position_w.ndim != 2 or robot_position_w.shape[-1] != 3:
        raise ValueError("robot_position_w must have shape (N, 3)")
    num_envs = robot_position_w.shape[0]
    if obstacle_positions_w.ndim != 3 or obstacle_positions_w.shape[0] != num_envs:
        raise ValueError("obstacle_positions_w must have shape (N, M, 3)")
    obstacle_count = obstacle_positions_w.shape[1]
    if obstacle_velocities_w.shape != obstacle_positions_w.shape:
        raise ValueError("obstacle_velocities_w must match obstacle_positions_w")
    if obstacle_sizes.shape == (obstacle_count, 3):
        obstacle_sizes_batched = obstacle_sizes.unsqueeze(0).expand(num_envs, -1, -1)
    elif obstacle_sizes.shape == (num_envs, obstacle_count, 3):
        obstacle_sizes_batched = obstacle_sizes
    else:
        raise ValueError("obstacle_sizes must have shape (M,3) or (N,M,3)")
    if obstacle_is_column.shape == (obstacle_count,):
        obstacle_is_column_batched = obstacle_is_column.unsqueeze(0).expand(num_envs, -1)
    elif obstacle_is_column.shape == (num_envs, obstacle_count):
        obstacle_is_column_batched = obstacle_is_column
    else:
        raise ValueError("obstacle_is_column must have shape (M,) or (N,M)")
    if active_count is not None and not 0 <= active_count <= obstacle_count:
        raise ValueError("active_count is outside the obstacle pool")
    if active_mask is not None and active_mask.shape not in ((obstacle_count,), (num_envs, obstacle_count)):
        raise ValueError("active_mask must have shape (M,) or (N, M)")
    if num_closest <= 0 or num_closest > obstacle_count:
        raise ValueError("num_closest must be in [1, M]")
    if sensing_range <= 0.0 or width_resolution <= 0.0:
        raise ValueError("sensing_range and width_resolution must be positive")

    relative = obstacle_positions_w - robot_position_w.unsqueeze(1)
    # Long columns model 2-D moving obstacles and span the flight volume.
    relative_for_state = relative.clone()
    relative_for_state[..., 2] = torch.where(
        obstacle_is_column_batched,
        torch.zeros_like(relative_for_state[..., 2]),
        relative_for_state[..., 2],
    )
    horizontal_distance = torch.linalg.vector_norm(relative_for_state[..., :2], dim=-1)
    if active_mask is None:
        if active_count is None:
            raise ValueError("active_count or active_mask is required")
        active_mask = torch.arange(obstacle_count, device=robot_position_w.device) < active_count
    if active_mask.ndim == 1:
        active_mask = active_mask.unsqueeze(0).expand(num_envs, -1)
    horizontal_distance = torch.where(active_mask, horizontal_distance, torch.full_like(horizontal_distance, torch.inf))
    selected_distance_xy, indices = torch.topk(horizontal_distance, num_closest, dim=1, largest=False)
    valid = torch.isfinite(selected_distance_xy) & (selected_distance_xy <= sensing_range)

    gather3 = indices.unsqueeze(-1).expand(-1, -1, 3)
    selected_relative = torch.gather(relative_for_state, 1, gather3)
    selected_positions = torch.gather(obstacle_positions_w, 1, gather3)
    selected_velocities = torch.gather(obstacle_velocities_w, 1, gather3)
    selected_sizes = torch.gather(obstacle_sizes_batched, 1, gather3)
    selected_columns = torch.gather(obstacle_is_column_batched, 1, indices)

    selected_distance = torch.linalg.vector_norm(selected_relative, dim=-1, keepdim=True)
    relative_g = world_to_goal_frame(selected_relative, start_to_goal_w.unsqueeze(1).expand_as(selected_relative))
    unit_relative_g = relative_g / selected_distance.clamp_min(eps)
    distance_xy = torch.linalg.vector_norm(relative_g[..., :2], dim=-1, keepdim=True)
    distance_z = relative_g[..., 2:3]
    velocity_g = world_to_goal_frame(
        selected_velocities,
        start_to_goal_w.unsqueeze(1).expand_as(selected_velocities),
    )

    width = selected_sizes[..., 0:1]
    width_category = width / width_resolution - 1.0
    # Public trainer encodes flying cuboids as 1 and tall columns as 0.
    height_category = torch.where(selected_columns.unsqueeze(-1), torch.zeros_like(width), selected_sizes[..., 2:3])
    state = torch.cat((unit_relative_g, distance_xy, distance_z, velocity_g, width_category, height_category), dim=-1)
    state = torch.where(valid.unsqueeze(-1), state, torch.zeros_like(state))

    selected_velocities = torch.where(valid.unsqueeze(-1), selected_velocities, torch.zeros_like(selected_velocities))
    radii = 0.5 * selected_sizes[..., 0]
    radii = torch.where(valid, radii, torch.zeros_like(radii))
    # For the shield, a tall obstacle sphere is centered at robot height so it
    # acts as the intended vertical column in the 3-D velocity-obstacle solver.
    selected_positions = selected_positions.clone()
    selected_positions[..., 2] = torch.where(selected_columns, robot_position_w[:, None, 2], selected_positions[..., 2])

    collision_xy = selected_distance_xy <= (0.5 * selected_sizes[..., 0] + robot_radius)
    collision_z = selected_relative[..., 2].abs() <= (0.5 * selected_sizes[..., 2] + robot_radius)
    collision = torch.any(valid & collision_xy & collision_z, dim=1)
    surface_distances = selected_distance.squeeze(-1) - 0.5 * selected_sizes[..., 0]
    surface_distances = torch.where(valid, surface_distances, torch.full_like(surface_distances, sensing_range))

    return DynamicObservation(
        state=state,
        indices=indices,
        valid=valid,
        positions_w=selected_positions,
        velocities_w=selected_velocities,
        radii=radii,
        surface_distances=surface_distances,
        collision=collision,
    )


def navrl_reward(
    robot_velocity_w: torch.Tensor,
    goal_direction_w: torch.Tensor,
    static_ray_distances: torch.Tensor,
    dynamic_surface_distances: torch.Tensor,
    current_height: torch.Tensor,
    start_height: torch.Tensor,
    goal_height: torch.Tensor,
    previous_velocity_w: torch.Tensor,
    *,
    has_dynamic_obstacles: bool = True,
    static_weight: float = 1.0,
    static_nearfield_weight: float = 0.0,
    static_nearfield_distance: float = 1.0,
    dynamic_weight: float = 1.0,
    smoothness_weight: float = 0.1,
    height_weight: float = 8.0,
    constant_reward: float = 1.0,
    height_tolerance: float = 0.2,
    eps: float = 1.0e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the five-term reward from NavRL Eq. 7--12 and public code."""
    velocity = torch.sum(robot_velocity_w * goal_direction_w, dim=-1)
    static_safety = torch.log(static_ray_distances.clamp_min(eps)).mean(dim=-1)
    # The source reward averages all virtual-ray log distances.  With a dense
    # angular scan, one imminent collision ray can therefore be diluted by
    # hundreds of clear rays.  This optional perception-only term supplies a
    # bounded dense gradient for the nearest static ray without using simulator
    # geometry.  A zero weight exactly preserves the public NavRL reward.
    minimum_static_distance = static_ray_distances.amin(dim=-1)
    nearfield_ratio = (
        (static_nearfield_distance - minimum_static_distance).clamp_min(0.0)
        / max(static_nearfield_distance, eps)
    )
    static_nearfield = -static_nearfield_weight * nearfield_ratio.square()
    if has_dynamic_obstacles:
        dynamic_safety = torch.log(dynamic_surface_distances.clamp_min(eps)).mean(dim=-1)
    else:
        dynamic_safety = torch.zeros_like(velocity)
    smoothness = torch.linalg.vector_norm(robot_velocity_w - previous_velocity_w, dim=-1)
    lower = torch.minimum(start_height, goal_height) - height_tolerance
    upper = torch.maximum(start_height, goal_height) + height_tolerance
    height_error = torch.maximum(lower - current_height, current_height - upper).clamp_min(0.0)

    components = {
        "constant": torch.full_like(velocity, constant_reward),
        "velocity": velocity,
        "static_safety": static_weight * static_safety,
        "static_nearfield": static_nearfield,
        "dynamic_safety": dynamic_weight * dynamic_safety,
        "smoothness": -smoothness_weight * smoothness,
        "height": -height_weight * height_error.square(),
    }
    reward = torch.stack(tuple(components.values()), dim=0).sum(dim=0)
    return reward, components


__all__ = [
    "DynamicObservation",
    "apply_navigation_command_guard",
    "navrl_dynamic_observation",
    "navrl_internal_state",
    "navrl_reward",
]
