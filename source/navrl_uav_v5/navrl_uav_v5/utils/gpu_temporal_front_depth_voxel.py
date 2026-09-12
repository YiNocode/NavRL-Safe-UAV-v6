"""Ego-compensated short-term local voxel memory for front depth cameras."""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from .gpu_front_depth_voxel import GpuFrontDepthVoxelizer


def _rotate_wxyz(vectors: torch.Tensor, quaternion: torch.Tensor) -> torch.Tensor:
    quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    while quaternion.ndim < vectors.ndim:
        quaternion = quaternion.unsqueeze(1)
    scalar = quaternion[..., 0:1]
    vector = quaternion[..., 1:4].expand_as(vectors)
    cross = 2.0 * torch.cross(vector, vectors, dim=-1)
    return vectors + scalar * cross + torch.cross(vector, cross, dim=-1)


class GpuTemporalFrontDepthVoxelizer:
    """Warp previous camera-local voxels and fuse the current depth observation.

    State channels are occupied confidence, free confidence, observed flag and
    recency confidence.  The map is finite-FOV history, not a synthetic 360
    degree scan.  Dynamic pixels may be excluded from persistent occupancy.
    """

    def __init__(
        self,
        num_envs: int,
        image_shape: tuple[int, int],
        grid_shape: tuple[int, int, int] = (16, 32, 32),
        *,
        history_duration_s: float = 1.5,
        forward_range_m: tuple[float, float] = (0.0, 5.0),
        lateral_range_m: tuple[float, float] = (-2.5, 2.5),
        vertical_range_m: tuple[float, float] = (-2.0, 2.0),
        near_m: float = 0.10,
        far_m: float = 5.0,
        pixel_stride: int = 4,
        free_samples: int = 16,
        device: torch.device | str = "cpu",
    ) -> None:
        if history_duration_s <= 0.0:
            raise ValueError("history_duration_s must be positive")
        self.num_envs = int(num_envs)
        self.grid_shape = tuple(int(value) for value in grid_shape)
        self.history_duration_s = float(history_duration_s)
        self.device = torch.device(device)
        self.instantaneous = GpuFrontDepthVoxelizer(
            num_envs,
            image_shape,
            grid_shape,
            forward_range_m=forward_range_m,
            lateral_range_m=lateral_range_m,
            vertical_range_m=vertical_range_m,
            near_m=near_m,
            far_m=far_m,
            pixel_stride=pixel_stride,
            free_samples=free_samples,
            device=self.device,
        )
        self.minimum = self.instantaneous.minimum
        self.maximum = self.instantaneous.maximum
        self.state = torch.zeros((num_envs, 4, *self.grid_shape), device=self.device)
        self.previous_position_w = torch.zeros((num_envs, 3), device=self.device)
        self.previous_quaternion_w_ros = torch.zeros((num_envs, 4), device=self.device)
        self.previous_quaternion_w_ros[:, 0] = 1.0
        self.has_previous = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.cell_centers = self._make_cell_centers()

    def _make_cell_centers(self) -> torch.Tensor:
        z_cells, y_cells, x_cells = self.grid_shape
        sizes = self.maximum - self.minimum
        forward = self.minimum[0] + (torch.arange(x_cells, device=self.device) + 0.5) * sizes[0] / x_cells
        right = self.minimum[1] + (torch.arange(y_cells, device=self.device) + 0.5) * sizes[1] / y_cells
        up = self.minimum[2] + (torch.arange(z_cells, device=self.device) + 0.5) * sizes[2] / z_cells
        grid_up, grid_right, grid_forward = torch.meshgrid(up, right, forward, indexing="ij")
        return torch.stack((grid_forward, grid_right, grid_up), dim=-1)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        ids = (
            torch.arange(self.num_envs, device=self.device)
            if env_ids is None
            else torch.as_tensor(env_ids, dtype=torch.int64, device=self.device)
        )
        self.state[ids] = 0.0
        self.has_previous[ids] = False
        self.instantaneous.reset(ids)

    def _warp_previous(
        self, position_w: torch.Tensor, quaternion_w_ros: torch.Tensor
    ) -> torch.Tensor:
        shape = (self.num_envs, *self.grid_shape, 3)
        current_custom = self.cell_centers.unsqueeze(0).expand(shape)
        current_optical = torch.stack(
            (current_custom[..., 1], -current_custom[..., 2], current_custom[..., 0]), dim=-1
        )
        current_world = _rotate_wxyz(current_optical, quaternion_w_ros)
        current_world += position_w[:, None, None, None, :]
        relative_previous_world = current_world - self.previous_position_w[:, None, None, None, :]
        inverse_previous = self.previous_quaternion_w_ros.clone()
        inverse_previous[:, 1:] *= -1.0
        previous_optical = _rotate_wxyz(relative_previous_world, inverse_previous)
        previous_custom = torch.stack(
            (previous_optical[..., 2], previous_optical[..., 0], -previous_optical[..., 1]), dim=-1
        )
        normalized = 2.0 * (previous_custom - self.minimum) / (self.maximum - self.minimum) - 1.0
        sampling_grid = torch.stack(
            (normalized[..., 0], normalized[..., 1], normalized[..., 2]), dim=-1
        )
        warped = functional.grid_sample(
            self.state,
            sampling_grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=False,
        )
        return torch.where(
            self.has_previous[:, None, None, None, None], warped, torch.zeros_like(warped)
        )

    @torch.no_grad()
    def update(
        self,
        depth_m: torch.Tensor,
        intrinsics: torch.Tensor,
        position_w: torch.Tensor,
        quaternion_w_ros: torch.Tensor,
        dt: float,
        dynamic_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if position_w.shape != (self.num_envs, 3) or quaternion_w_ros.shape != (self.num_envs, 4):
            raise ValueError("camera pose has an unexpected shape")
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        current = self.instantaneous.update(depth_m, intrinsics, dynamic_mask)
        warped = self._warp_previous(position_w, quaternion_w_ros)
        decay = max(0.0, 1.0 - dt / self.history_duration_s)
        current_observed = current[:, 2] > 0.0
        occupied = torch.where(current_observed, current[:, 0], warped[:, 0] * decay)
        free = torch.where(current_observed, current[:, 1], warped[:, 1] * decay)
        recency = torch.where(current_observed, torch.ones_like(current[:, 2]), warped[:, 3] * decay)
        observed = (recency > 0.0).to(recency.dtype)
        self.state.copy_(torch.stack((occupied, free, observed, recency), dim=1))
        self.previous_position_w.copy_(position_w)
        self.previous_quaternion_w_ros.copy_(quaternion_w_ros)
        self.has_previous.fill_(True)
        return self.state

    @property
    def memory_bytes(self) -> int:
        tensors = (
            self.state,
            self.previous_position_w,
            self.previous_quaternion_w_ros,
            self.has_previous,
            self.cell_centers,
        )
        return self.instantaneous.memory_bytes + sum(
            tensor.numel() * tensor.element_size() for tensor in tensors
        )


__all__ = ["GpuTemporalFrontDepthVoxelizer"]
