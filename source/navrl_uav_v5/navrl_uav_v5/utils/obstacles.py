"""Static-obstacle sampling and collision helpers."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True)
class ObstacleSpec:
    """Geometry and local pose of one axis-aligned static obstacle."""

    kind: Literal["cuboid", "cylinder"]
    center: tuple[float, float, float]
    width: float
    length: float
    height: float
    radius: float

    @property
    def footprint_radius(self) -> float:
        if self.kind == "cylinder":
            return self.radius
        return 0.5 * math.hypot(self.width, self.length)

    @property
    def x_extent(self) -> float:
        return self.radius if self.kind == "cylinder" else 0.5 * self.width


def sample_obstacle_specs(
    num_obstacles: int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    seed: int,
    min_surface_separation: float = 0.5,
    max_attempts: int = 1000,
    height_range: tuple[float, float] = (1.0, 4.0),
) -> tuple[ObstacleSpec, ...]:
    """Sample a reproducible mixed cuboid/cylinder topology."""
    if num_obstacles < 0:
        raise ValueError("num_obstacles must be non-negative")
    if min_surface_separation < 0.0:
        raise ValueError("min_surface_separation must be non-negative")
    if len(height_range) != 2 or height_range[0] <= 0.0 or height_range[1] < height_range[0]:
        raise ValueError("height_range must be a positive ordered pair")

    generator = random.Random(seed)
    specs: list[ObstacleSpec] = []
    for index in range(num_obstacles):
        kind: Literal["cuboid", "cylinder"] = "cuboid" if index % 2 == 0 else "cylinder"
        for _ in range(max_attempts):
            if kind == "cuboid":
                width = generator.uniform(0.5, 1.5)
                length = generator.uniform(0.5, 1.5)
                radius = 0.0
                footprint_radius = 0.5 * math.hypot(width, length)
            else:
                radius = generator.uniform(0.25, 0.75)
                width = 2.0 * radius
                length = 2.0 * radius
                footprint_radius = radius
            height = generator.uniform(*height_range)

            margin = footprint_radius + 0.5
            x = generator.uniform(x_range[0] + margin, x_range[1] - margin)
            y = generator.uniform(y_range[0] + margin, y_range[1] - margin)
            candidate = ObstacleSpec(kind, (x, y, 0.5 * height), width, length, height, radius)
            separated = all(
                math.hypot(x - other.center[0], y - other.center[1])
                - candidate.footprint_radius
                - other.footprint_radius
                > min_surface_separation
                for other in specs
            )
            if separated:
                specs.append(candidate)
                break
        else:
            raise RuntimeError(f"Unable to place obstacle {index} after {max_attempts} attempts")
    return tuple(specs)


def point_to_obstacle_distance(points: torch.Tensor, obstacle: ObstacleSpec) -> torch.Tensor:
    """Return exact outside distance from points to an axis-aligned collider."""
    if points.shape[-1:] != (3,):
        raise ValueError(f"points must have shape (..., 3); received {tuple(points.shape)}")
    center = points.new_tensor(obstacle.center)
    delta = points - center
    if obstacle.kind == "cuboid":
        half_extents = points.new_tensor((0.5 * obstacle.width, 0.5 * obstacle.length, 0.5 * obstacle.height))
        outside = (delta.abs() - half_extents).clamp_min(0.0)
        return torch.linalg.vector_norm(outside, dim=-1)

    radial_outside = (torch.linalg.vector_norm(delta[..., :2], dim=-1) - obstacle.radius).clamp_min(0.0)
    vertical_outside = (delta[..., 2].abs() - 0.5 * obstacle.height).clamp_min(0.0)
    return torch.sqrt(radial_outside.square() + vertical_outside.square())


def minimum_obstacle_clearance(points: torch.Tensor, obstacles: tuple[ObstacleSpec, ...]) -> torch.Tensor:
    """Return minimum point-to-collider distance for every point."""
    if len(obstacles) == 0:
        return torch.full(points.shape[:-1], torch.inf, dtype=points.dtype, device=points.device)
    distances = torch.stack([point_to_obstacle_distance(points, obstacle) for obstacle in obstacles], dim=-1)
    return distances.min(dim=-1).values


def sample_collision_free_start_goal(
    num_samples: int,
    lower: torch.Tensor,
    upper: torch.Tensor,
    min_start_goal_distance: float,
    obstacles: tuple[ObstacleSpec, ...],
    obstacle_clearance: float,
    max_attempts: int = 256,
    max_start_goal_distance: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample start/goal pairs that are mutually separated and clear of colliders.

    Candidate pairs are rejected in large device-side batches.  Dense scenes
    can have only a few percent valid pairs, so repeatedly resampling one
    partially valid environment batch would otherwise cause hundreds of GPU
    synchronization points at every reset.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    extent = upper - lower

    def sample(count: int) -> torch.Tensor:
        return lower + torch.rand((count, 3), dtype=lower.dtype, device=lower.device) * extent

    accepted_starts: list[torch.Tensor] = []
    accepted_goals: list[torch.Tensor] = []
    remaining = num_samples
    for _ in range(max_attempts):
        candidate_count = max(4096, remaining * 64)
        starts = sample(candidate_count)
        goals = sample(candidate_count)
        pair_distance = torch.linalg.vector_norm(goals - starts, dim=-1)
        valid = (
            (pair_distance >= min_start_goal_distance)
            & (minimum_obstacle_clearance(starts, obstacles) > obstacle_clearance)
            & (minimum_obstacle_clearance(goals, obstacles) > obstacle_clearance)
        )
        if max_start_goal_distance is not None:
            valid &= pair_distance <= max_start_goal_distance
        valid_indices = torch.nonzero(valid, as_tuple=False).flatten()[:remaining]
        if valid_indices.numel() == 0:
            continue
        accepted_starts.append(starts[valid_indices])
        accepted_goals.append(goals[valid_indices])
        remaining -= valid_indices.numel()
        if remaining == 0:
            return torch.cat(accepted_starts, dim=0), torch.cat(accepted_goals, dim=0)
    raise RuntimeError(f"Unable to sample {num_samples} collision-free start/goal pairs")


def contact_forces_to_collision(net_forces_w: torch.Tensor, threshold: float) -> torch.Tensor:
    """Reduce current/history contact-force tensors to one collision flag per environment."""
    if net_forces_w.ndim < 3 or net_forces_w.shape[-1] != 3:
        raise ValueError("net_forces_w must have shape (num_envs, ..., 3)")
    magnitudes = torch.linalg.vector_norm(net_forces_w, dim=-1)
    reduction_dims = tuple(range(1, magnitudes.ndim))
    return torch.amax(magnitudes, dim=reduction_dims) > threshold


def out_of_bounds_mask(
    position_local: torch.Tensor,
    x_limit: float = 10.0,
    y_limit: float = 10.0,
    min_z: float = 0.2,
    max_z: float = 5.0,
) -> torch.Tensor:
    """Return map/altitude termination flags."""
    return (
        (position_local[:, 0].abs() > x_limit)
        | (position_local[:, 1].abs() > y_limit)
        | (position_local[:, 2] < min_z)
        | (position_local[:, 2] > max_z)
    )
