"""Vectorized mathematics for the goal-aligned navigation frame."""

from __future__ import annotations

import torch


def _validate_vector_tensor(name: str, value: torch.Tensor) -> None:
    if value.shape[-1:] != (3,):
        raise ValueError(f"{name} must have shape (..., 3); received {tuple(value.shape)}")


def goal_frame_basis(relative_goal_w: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """Return the right-handed goal-frame basis expressed in world coordinates.

    The basis columns are ``x_g``, ``y_g``, and ``z_g``. ``x_g`` is the
    horizontal direction from the drone to the goal, ``y_g = z_w x x_g``, and
    ``z_g`` is world up. A purely vertical/zero relative goal uses world ``+x``
    as a deterministic horizontal fallback.
    """
    _validate_vector_tensor("relative_goal_w", relative_goal_w)

    horizontal_norm = torch.linalg.vector_norm(relative_goal_w[..., :2], dim=-1, keepdim=True)
    safe_norm = horizontal_norm.clamp_min(eps)
    forward_xy = relative_goal_w[..., :2] / safe_norm
    fallback_xy = torch.zeros_like(forward_xy)
    fallback_xy[..., 0] = 1.0
    forward_xy = torch.where(horizontal_norm > eps, forward_xy, fallback_xy)

    zeros = torch.zeros_like(forward_xy[..., :1])
    ones = torch.ones_like(zeros)
    forward = torch.cat((forward_xy, zeros), dim=-1)
    lateral = torch.cat((-forward_xy[..., 1:2], forward_xy[..., 0:1], zeros), dim=-1)
    up = torch.cat((zeros, zeros, ones), dim=-1)
    return torch.stack((forward, lateral, up), dim=-1)


def goal_frame_to_world(vector_goal: torch.Tensor, relative_goal_w: torch.Tensor) -> torch.Tensor:
    """Transform vectors from the goal frame to the world frame."""
    _validate_vector_tensor("vector_goal", vector_goal)
    _validate_vector_tensor("relative_goal_w", relative_goal_w)
    basis_w = goal_frame_basis(relative_goal_w)
    return torch.matmul(basis_w, vector_goal.unsqueeze(-1)).squeeze(-1)


def world_to_goal_frame(vector_w: torch.Tensor, relative_goal_w: torch.Tensor) -> torch.Tensor:
    """Transform world-frame vectors into the goal frame."""
    _validate_vector_tensor("vector_w", vector_w)
    _validate_vector_tensor("relative_goal_w", relative_goal_w)
    basis_w = goal_frame_basis(relative_goal_w)
    return torch.matmul(basis_w.transpose(-1, -2), vector_w.unsqueeze(-1)).squeeze(-1)


def sample_start_goal(
    num_samples: int,
    lower: torch.Tensor,
    upper: torch.Tensor,
    min_distance: float,
    max_attempts: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample start/goal pairs in a box using vectorized rejection sampling."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    _validate_vector_tensor("lower", lower)
    _validate_vector_tensor("upper", upper)
    if lower.ndim != 1 or upper.ndim != 1:
        raise ValueError("lower and upper must each have shape (3,)")
    if not torch.all(upper > lower):
        raise ValueError("Every upper bound must be greater than its lower bound")
    if min_distance < 0.0:
        raise ValueError("min_distance must be non-negative")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")

    extent = upper - lower

    def sample(count: int) -> torch.Tensor:
        return lower + torch.rand((count, 3), dtype=lower.dtype, device=lower.device) * extent

    starts = sample(num_samples)
    goals = sample(num_samples)
    for _ in range(max_attempts):
        invalid = torch.linalg.vector_norm(goals - starts, dim=-1) < min_distance
        invalid_count = int(invalid.sum().item())
        if invalid_count == 0:
            return starts, goals
        goals[invalid] = sample(invalid_count)

    minimum = torch.linalg.vector_norm(goals - starts, dim=-1).min().item()
    raise RuntimeError(
        f"Unable to sample start/goal pairs at least {min_distance:.3f} m apart "
        f"after {max_attempts} attempts (minimum={minimum:.3f} m)"
    )
