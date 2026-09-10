"""GPU construction of a finite-FOV local voxel observation from axial depth."""

from __future__ import annotations

import torch


class GpuFrontDepthVoxelizer:
    """Backproject sampled depth rays into a camera-local occupancy grid.

    The returned channel-first tensor has channels ``occupied``, ``free`` and
    ``observed`` and shape ``[N, 3, Z, Y, X]``.  X points forward, Y right and
    Z up.  Unobserved cells remain zero in all channels; they are therefore not
    silently treated as free space.
    """

    def __init__(
        self,
        num_envs: int,
        image_shape: tuple[int, int],
        grid_shape: tuple[int, int, int] = (16, 32, 32),
        *,
        forward_range_m: tuple[float, float] = (0.0, 5.0),
        lateral_range_m: tuple[float, float] = (-2.5, 2.5),
        vertical_range_m: tuple[float, float] = (-2.0, 2.0),
        near_m: float = 0.10,
        far_m: float = 5.0,
        pixel_stride: int = 4,
        free_samples: int = 16,
        device: torch.device | str = "cpu",
    ) -> None:
        height, width = image_shape
        z_cells, y_cells, x_cells = grid_shape
        if min(num_envs, height, width, z_cells, y_cells, x_cells) <= 0:
            raise ValueError("batch, image, and grid dimensions must be positive")
        if pixel_stride <= 0 or free_samples <= 0:
            raise ValueError("pixel_stride and free_samples must be positive")
        if not (0.0 <= forward_range_m[0] < forward_range_m[1] <= far_m):
            raise ValueError("forward range must be ordered and lie within far_m")
        self.num_envs = int(num_envs)
        self.height = int(height)
        self.width = int(width)
        self.grid_shape = (int(z_cells), int(y_cells), int(x_cells))
        self.near_m = float(near_m)
        self.far_m = float(far_m)
        self.pixel_stride = int(pixel_stride)
        self.free_samples = int(free_samples)
        self.device = torch.device(device)
        self.minimum = torch.tensor(
            [forward_range_m[0], lateral_range_m[0], vertical_range_m[0]],
            device=self.device,
        )
        self.maximum = torch.tensor(
            [forward_range_m[1], lateral_range_m[1], vertical_range_m[1]],
            device=self.device,
        )
        rows = torch.arange(0, height, pixel_stride, device=self.device)
        columns = torch.arange(0, width, pixel_stride, device=self.device)
        v, u = torch.meshgrid(rows, columns, indexing="ij")
        self.pixel_u = u.flatten().long()
        self.pixel_v = v.flatten().long()
        self.free_fractions = torch.linspace(
            0.0, 1.0, free_samples + 2, device=self.device
        )[1:-1]
        self.state = torch.zeros(
            (num_envs, 3, z_cells, y_cells, x_cells), device=self.device
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        ids = (
            torch.arange(self.num_envs, device=self.device)
            if env_ids is None
            else torch.as_tensor(env_ids, dtype=torch.int64, device=self.device)
        )
        self.state[ids] = 0.0

    def _flat_indices(self, points_xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map ``[..., (forward,right,up)]`` points to flattened Z/Y/X cells."""
        scale = torch.tensor(
            [self.grid_shape[2], self.grid_shape[1], self.grid_shape[0]],
            dtype=points_xyz.dtype,
            device=self.device,
        )
        cells = torch.floor((points_xyz - self.minimum) / (self.maximum - self.minimum) * scale).long()
        inside = ((cells >= 0) & (cells < scale.long())).all(dim=-1)
        x, y, z = cells.unbind(dim=-1)
        flat = z.clamp(0, self.grid_shape[0] - 1) * (self.grid_shape[1] * self.grid_shape[2])
        flat += y.clamp(0, self.grid_shape[1] - 1) * self.grid_shape[2]
        flat += x.clamp(0, self.grid_shape[2] - 1)
        return flat, inside

    @torch.no_grad()
    def update(self, depth_m: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
        if depth_m.shape != (self.num_envs, self.height, self.width, 1):
            raise ValueError("depth_m has an unexpected shape")
        if intrinsics.shape != (self.num_envs, 3, 3):
            raise ValueError("intrinsics has an unexpected shape")
        depth = depth_m[:, self.pixel_v, self.pixel_u, 0]
        valid = torch.isfinite(depth) & (depth >= self.near_m) & (depth <= self.far_m)
        safe_depth = torch.where(valid, depth, torch.zeros_like(depth))
        fx = intrinsics[:, 0, 0:1]
        fy = intrinsics[:, 1, 1:2]
        cx = intrinsics[:, 0, 2:3]
        cy = intrinsics[:, 1, 2:3]
        right = (self.pixel_u.to(depth.dtype)[None] - cx) * safe_depth / fx
        down = (self.pixel_v.to(depth.dtype)[None] - cy) * safe_depth / fy
        endpoints = torch.stack((safe_depth, right, -down), dim=-1)

        voxel_count = self.grid_shape[0] * self.grid_shape[1] * self.grid_shape[2]
        occupied = torch.zeros((self.num_envs, voxel_count), device=self.device)
        free = torch.zeros_like(occupied)
        occupied_index, occupied_inside = self._flat_indices(endpoints)
        occupied.scatter_reduce_(
            1, occupied_index, (valid & occupied_inside).float(), reduce="amax", include_self=True
        )

        free_points = endpoints[:, :, None, :] * self.free_fractions[None, None, :, None]
        free_index, free_inside = self._flat_indices(free_points)
        free_valid = valid[:, :, None] & free_inside
        free.scatter_reduce_(
            1, free_index.flatten(1), free_valid.float().flatten(1), reduce="amax", include_self=True
        )
        free *= 1.0 - occupied
        observed = torch.maximum(occupied, free)
        self.state.copy_(
            torch.stack((occupied, free, observed), dim=1).view(
                self.num_envs, 3, *self.grid_shape
            )
        )
        return self.state

    @property
    def memory_bytes(self) -> int:
        tensors = (self.minimum, self.maximum, self.pixel_u, self.pixel_v, self.free_fractions, self.state)
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


__all__ = ["GpuFrontDepthVoxelizer"]
