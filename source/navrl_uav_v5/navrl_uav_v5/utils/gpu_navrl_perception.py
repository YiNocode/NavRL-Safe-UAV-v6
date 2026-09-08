"""GPU-only depth-to-voxel-to-virtual-ray perception for V5.

The policy boundary in this module is deliberately sensor-only.  Isaac/Warp
produces ray hit points, then PyTorch CUDA kernels build one occupancy layer per
parallel environment and extract NavRL's structured static and dynamic states.
No obstacle pose, velocity, size, mesh id, or semantic label is accepted by the
API.  Simulator ground truth may therefore be used for terminal/evaluation
labels without accidentally leaking into PPO observations.

Coordinate convention
---------------------
Isaac world coordinates and the map use a right-handed frame with +Z up.  Ray
directions are defined in the UAV body frame as +X forward, +Y left, +Z up and
are rotated into the world frame by the UAV yaw before voxel traversal.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as functional


@dataclass(frozen=True)
class GpuPerceptionOutput:
    """Structured perception tensors returned for every parallel environment."""

    static_raw: torch.Tensor
    static_normalized: torch.Tensor
    static_hit_mask: torch.Tensor
    dynamic_positions_w: torch.Tensor
    dynamic_velocities_w: torch.Tensor
    dynamic_sizes: torch.Tensor
    dynamic_valid: torch.Tensor
    motion_mask: torch.Tensor


def make_body_ray_directions(
    horizontal_fov: float,
    horizontal_step: float,
    vertical_fov: float,
    vertical_step: float,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return unit directions shaped ``[Nh,Nv,3]`` in the UAV body frame."""
    if not 0.0 < horizontal_fov <= 360.0:
        raise ValueError("horizontal_fov must lie in (0, 360]")
    if not 0.0 <= vertical_fov <= 180.0:
        raise ValueError("vertical_fov must lie in [0, 180]")
    if horizontal_step <= 0.0 or vertical_step <= 0.0:
        raise ValueError("ray angle steps must be positive")
    horizontal_count = max(1, int(round(horizontal_fov / horizontal_step)))
    vertical_count = max(1, int(round(vertical_fov / vertical_step)) + 1)
    azimuth = torch.arange(horizontal_count, dtype=torch.float32, device=device) * math.radians(horizontal_step)
    if horizontal_fov < 360.0:
        azimuth -= 0.5 * math.radians(horizontal_fov)
    elevation = torch.linspace(
        -0.5 * math.radians(vertical_fov),
        0.5 * math.radians(vertical_fov),
        vertical_count,
        dtype=torch.float32,
        device=device,
    )
    phi, theta = torch.meshgrid(azimuth, elevation, indexing="ij")
    directions = torch.stack(
        (
            torch.cos(theta) * torch.cos(phi),
            torch.cos(theta) * torch.sin(phi),
            torch.sin(theta),
        ),
        dim=-1,
    )
    return functional.normalize(directions, dim=-1)


class BatchedOccupancyVoxelMap:
    """Dense per-environment occupancy timestamps stored entirely on the GPU.

    ``grid`` has shape ``[N,X,Y,Z]``.  A cell is occupied only when its int32
    timestamp equals that environment's current sensor-frame token.  Old depth
    returns consequently disappear without clearing the whole tensor every
    policy step; selected environments are cleared only on episode reset.
    """

    _EMPTY = -2_000_000_000

    def __init__(
        self,
        num_envs: int,
        *,
        voxel_size: float,
        map_size: Sequence[float],
        origin_local: Sequence[float],
        device: torch.device | str,
    ) -> None:
        if num_envs <= 0 or voxel_size <= 0.0:
            raise ValueError("num_envs and voxel_size must be positive")
        if len(map_size) != 3 or len(origin_local) != 3 or any(value <= 0.0 for value in map_size):
            raise ValueError("map_size and origin_local must contain three values")
        self.num_envs = int(num_envs)
        self.voxel_size = float(voxel_size)
        self.device = torch.device(device)
        self.origin_local = torch.tensor(origin_local, dtype=torch.float32, device=self.device)
        self.grid_shape = tuple(int(math.ceil(float(value) / self.voxel_size)) for value in map_size)
        self.grid = torch.full(
            (self.num_envs, *self.grid_shape),
            self._EMPTY,
            dtype=torch.int32,
            device=self.device,
        )
        self.frame_token = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._flat_size = math.prod(self.grid_shape)

    @property
    def memory_bytes(self) -> int:
        """Allocated map storage in bytes (excluding small temporaries)."""
        return self.grid.numel() * self.grid.element_size()

    def reset(self, env_ids: torch.Tensor | Sequence[int] | None = None) -> None:
        """Clear all or selected per-environment maps and frame counters."""
        if env_ids is None:
            self.grid.fill_(self._EMPTY)
            self.frame_token.zero_()
            return
        ids = torch.as_tensor(env_ids, dtype=torch.int64, device=self.device)
        self.grid[ids] = self._EMPTY
        self.frame_token[ids] = 0

    def world_to_voxel(self, points_w: torch.Tensor, env_origins_w: torch.Tensor) -> torch.Tensor:
        """Convert ``[N,...,3]`` world points into integer local-map indices."""
        if points_w.shape[0] != self.num_envs or points_w.shape[-1] != 3:
            raise ValueError("points_w must have shape [N,...,3]")
        if env_origins_w.shape != (self.num_envs, 3):
            raise ValueError("env_origins_w must have shape [N,3]")
        view_shape = (self.num_envs,) + (1,) * (points_w.ndim - 2) + (3,)
        local = points_w - env_origins_w.view(view_shape)
        return torch.floor((local - self.origin_local) / self.voxel_size).to(torch.int64)

    def is_inside(self, indices: torch.Tensor) -> torch.Tensor:
        """Return an in-bounds mask for indices shaped ``[N,...,3]``."""
        shape = torch.tensor(self.grid_shape, dtype=torch.int64, device=self.device)
        return torch.all((indices >= 0) & (indices < shape), dim=-1)

    def _linear_indices(self, indices: torch.Tensor) -> torch.Tensor:
        size_y, size_z = self.grid_shape[1], self.grid_shape[2]
        return (indices[..., 0] * size_y + indices[..., 1]) * size_z + indices[..., 2]

    @torch.no_grad()
    def integrate_hits(
        self,
        hit_points_w: torch.Tensor,
        env_origins_w: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        """Build the current sensor-frame occupancy layer via one GPU scatter.

        Args:
            hit_points_w: Sensor hit points shaped ``[N,R,3]``.
            env_origins_w: Isaac environment origins shaped ``[N,3]``.
            valid: Static-hit mask shaped ``[N,R]``. Dynamic hypotheses must
                already be removed by the motion detector.
        """
        if hit_points_w.ndim != 3 or hit_points_w.shape[:2] != valid.shape:
            raise ValueError("hit_points_w/valid must have shapes [N,R,3]/[N,R]")
        self.frame_token.add_(1)
        indices = self.world_to_voxel(hit_points_w, env_origins_w)
        inside = valid & self.is_inside(indices)
        safe_indices = indices.clamp_min(0)
        maximum = torch.tensor(self.grid_shape, device=self.device) - 1
        safe_indices = torch.minimum(safe_indices, maximum)
        linear = self._linear_indices(safe_indices)
        env_index = torch.arange(self.num_envs, device=self.device).unsqueeze(1).expand_as(linear)
        flat = self.grid.view(self.num_envs, self._flat_size)
        flat[env_index[inside], linear[inside]] = self.frame_token.unsqueeze(1).expand_as(linear)[inside]

    @torch.no_grad()
    def ray_cast(
        self,
        uav_positions_w: torch.Tensor,
        env_origins_w: torch.Tensor,
        heading: torch.Tensor,
        body_directions: torch.Tensor,
        *,
        max_distance: float,
        ray_step_size: float,
        no_hit_offset: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Vectorized virtual ray traversal returning ``[N,Nh,Nv]`` tensors."""
        if uav_positions_w.shape != (self.num_envs, 3) or heading.shape != (self.num_envs,):
            raise ValueError("uav_positions_w/heading have unexpected shapes")
        if body_directions.ndim != 3 or body_directions.shape[-1] != 3:
            raise ValueError("body_directions must have shape [Nh,Nv,3]")
        if not 0.0 < ray_step_size <= self.voxel_size:
            raise ValueError("ray_step_size must lie in (0, voxel_size]")
        if max_distance <= 0.0 or no_hit_offset <= 0.0:
            raise ValueError("ray range and no-hit offset must be positive")

        cosine = torch.cos(heading)[:, None, None]
        sine = torch.sin(heading)[:, None, None]
        body = body_directions.to(device=self.device, dtype=uav_positions_w.dtype)
        x_body = body[..., 0].unsqueeze(0)
        y_body = body[..., 1].unsqueeze(0)
        z_body = body[..., 2].unsqueeze(0).expand(self.num_envs, -1, -1)
        directions_w = torch.stack(
            (
                cosine * x_body - sine * y_body,
                sine * x_body + cosine * y_body,
                z_body,
            ),
            dim=-1,
        )
        sample_distances = torch.arange(
            ray_step_size,
            max_distance + 0.5 * ray_step_size,
            ray_step_size,
            device=self.device,
            dtype=uav_positions_w.dtype,
        ).clamp_max(max_distance)
        points = (
            uav_positions_w[:, None, None, None, :]
            + directions_w.unsqueeze(-2) * sample_distances[None, None, None, :, None]
        )
        indices = self.world_to_voxel(points, env_origins_w)
        inside = self.is_inside(indices)
        safe_indices = indices.clamp_min(0)
        maximum = torch.tensor(self.grid_shape, device=self.device) - 1
        safe_indices = torch.minimum(safe_indices, maximum)
        linear = self._linear_indices(safe_indices)
        flat = self.grid.view(self.num_envs, self._flat_size)
        samples = torch.gather(flat, 1, linear.flatten(start_dim=1)).view_as(linear)
        occupied = inside & (samples == self.frame_token[:, None, None, None])
        has_hit = occupied.any(dim=-1)
        first_index = occupied.to(torch.int64).argmax(dim=-1)
        first_distance = sample_distances[first_index]
        raw = torch.where(
            has_hit,
            first_distance,
            torch.full_like(first_distance, max_distance + no_hit_offset),
        )
        normalized = raw.clamp(0.0, max_distance) / max_distance
        return raw, normalized, has_hit


class GpuDepthMotionTracker:
    """Batched temporal depth residual detector with an alpha-beta tracker.

    This is the high-throughput V5 counterpart of V4's CPU
    U-depth/DBSCAN/Kalman path.  It uses no semantic/mesh labels: after
    ego-motion compensation, depth discontinuities are non-max suppressed in
    angular space, lifted to 3-D, associated with prior tracks, and filtered on
    CUDA.  The output contract remains NavRL's tracked position/velocity/size.
    """

    def __init__(
        self,
        num_envs: int,
        ray_shape: tuple[int, int],
        max_tracks: int,
        *,
        max_distance: float,
        motion_threshold: float,
        association_distance: float,
        velocity_smoothing: float,
        horizontal_angle_step: float,
        device: torch.device | str,
    ) -> None:
        if num_envs <= 0 or max_tracks <= 0:
            raise ValueError("num_envs and max_tracks must be positive")
        if motion_threshold <= 0.0 or association_distance <= 0.0:
            raise ValueError("motion and association thresholds must be positive")
        if not 0.0 <= velocity_smoothing < 1.0:
            raise ValueError("velocity_smoothing must lie in [0,1)")
        self.num_envs = int(num_envs)
        self.ray_shape = ray_shape
        self.max_tracks = int(max_tracks)
        self.max_distance = float(max_distance)
        self.motion_threshold = float(motion_threshold)
        self.association_distance = float(association_distance)
        self.velocity_smoothing = float(velocity_smoothing)
        self.horizontal_angle_step = float(horizontal_angle_step)
        self.device = torch.device(device)
        self.previous_depth = torch.full(
            (self.num_envs, *self.ray_shape), self.max_distance, device=self.device
        )
        self.previous_origin_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.has_previous_frame = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.positions_w = torch.zeros((self.num_envs, self.max_tracks, 3), device=self.device)
        self.velocities_w = torch.zeros_like(self.positions_w)
        self.sizes = torch.zeros_like(self.positions_w)
        self.valid = torch.zeros((self.num_envs, self.max_tracks), dtype=torch.bool, device=self.device)

    def reset(self, env_ids: torch.Tensor | Sequence[int] | None = None) -> None:
        """Clear selected temporal frames and tracks."""
        ids = (
            torch.arange(self.num_envs, device=self.device)
            if env_ids is None
            else torch.as_tensor(env_ids, dtype=torch.int64, device=self.device)
        )
        self.previous_depth[ids] = self.max_distance
        self.previous_origin_w[ids] = 0.0
        self.has_previous_frame[ids] = False
        self.positions_w[ids] = 0.0
        self.velocities_w[ids] = 0.0
        self.sizes[ids] = 0.0
        self.valid[ids] = False

    @torch.no_grad()
    def update(
        self,
        depth_m: torch.Tensor,
        hit_points_w: torch.Tensor,
        directions_w: torch.Tensor,
        sensor_origins_w: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update all environments and return tracks plus ``[N,Nh,Nv]`` mask."""
        expected = (self.num_envs, *self.ray_shape)
        if depth_m.shape != expected:
            raise ValueError(f"depth_m must have shape {expected}")
        ray_count = math.prod(self.ray_shape)
        if hit_points_w.shape != (self.num_envs, ray_count, 3):
            raise ValueError("hit_points_w has an unexpected shape")
        if directions_w.shape != hit_points_w.shape or sensor_origins_w.shape != (self.num_envs, 3):
            raise ValueError("directions/origins have unexpected shapes")
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        direction_matrix = directions_w.reshape(self.num_envs, self.ray_shape[1], self.ray_shape[0], 3).transpose(1, 2)
        ego_delta = sensor_origins_w - self.previous_origin_w
        ego_radial = torch.sum(direction_matrix * ego_delta[:, None, None, :], dim=-1)
        predicted_static_depth = (self.previous_depth - ego_radial).clamp(0.0, self.max_distance)
        residual = (depth_m - predicted_static_depth).abs()
        current_valid = depth_m < self.max_distance - 1.0e-4
        previous_valid = self.previous_depth < self.max_distance - 1.0e-4
        motion = (
            current_valid
            & previous_valid
            & self.has_previous_frame[:, None, None]
            & (residual >= self.motion_threshold)
        )
        # Dilate each motion response to cover the local obstacle surface and
        # suppress adjacent duplicate track proposals on the GPU.
        motion = functional.max_pool2d(motion[:, None].float(), 3, stride=1, padding=1).squeeze(1).bool()
        score = torch.where(
            motion,
            residual * (1.0 - depth_m / self.max_distance).clamp_min(0.05),
            torch.zeros_like(residual),
        )
        local_max = functional.max_pool2d(score[:, None], 3, stride=1, padding=1).squeeze(1)
        score = torch.where(score >= local_max, score, torch.zeros_like(score))
        selected_score, selected_index = torch.topk(score.flatten(start_dim=1), self.max_tracks, dim=1)
        selected_valid = selected_score > 0.0

        # Isaac's LidarPattern flattening is [Nv,Nh]; transpose the hit image
        # to the policy's [Nh,Nv] convention before applying selected indices.
        flat_hits = hit_points_w.reshape(
            self.num_envs, self.ray_shape[1], self.ray_shape[0], 3
        ).transpose(1, 2).flatten(start_dim=1, end_dim=2)
        flat_directions = direction_matrix.flatten(start_dim=1, end_dim=2)
        gather3 = selected_index.unsqueeze(-1).expand(-1, -1, 3)
        surface_points = torch.gather(flat_hits, 1, gather3)
        selected_directions = torch.gather(flat_directions, 1, gather3)
        selected_depth = torch.gather(depth_m.flatten(start_dim=1), 1, selected_index)
        angular_width = 2.0 * selected_depth * math.tan(0.5 * math.radians(self.horizontal_angle_step))
        width = angular_width.clamp(0.20, 1.50)
        candidate_sizes = torch.stack((width, width, width.clamp_min(0.50)), dim=-1)
        candidates = surface_points + 0.5 * width.unsqueeze(-1) * selected_directions

        distance = torch.cdist(candidates, self.positions_w)
        distance = torch.where(self.valid[:, None, :], distance, torch.full_like(distance, torch.inf))
        association_distance, association_index = distance.min(dim=-1)
        matched = selected_valid & (association_distance <= self.association_distance)
        previous_position = torch.gather(self.positions_w, 1, association_index.unsqueeze(-1).expand(-1, -1, 3))
        previous_velocity = torch.gather(self.velocities_w, 1, association_index.unsqueeze(-1).expand(-1, -1, 3))
        measured_velocity = ((candidates - previous_position) / dt).clamp(-5.0, 5.0)
        filtered_velocity = (
            self.velocity_smoothing * previous_velocity
            + (1.0 - self.velocity_smoothing) * measured_velocity
        )
        filtered_velocity = torch.where(matched.unsqueeze(-1), filtered_velocity, torch.zeros_like(filtered_velocity))
        self.positions_w.copy_(torch.where(selected_valid.unsqueeze(-1), candidates, torch.zeros_like(candidates)))
        self.velocities_w.copy_(
            torch.where(
                selected_valid.unsqueeze(-1),
                filtered_velocity,
                torch.zeros_like(filtered_velocity),
            )
        )
        self.sizes.copy_(torch.where(selected_valid.unsqueeze(-1), candidate_sizes, torch.zeros_like(candidate_sizes)))
        self.valid.copy_(selected_valid)
        self.previous_depth.copy_(depth_m)
        self.previous_origin_w.copy_(sensor_origins_w)
        self.has_previous_frame.fill_(True)
        return self.positions_w, self.velocities_w, self.sizes, self.valid, motion


class GpuNavRLPerception:
    """Compose sensor-only motion tracking, occupancy mapping, and ray casting."""

    def __init__(
        self,
        num_envs: int,
        *,
        voxel_size: float,
        map_size: Sequence[float],
        map_origin_local: Sequence[float],
        horizontal_fov: float,
        horizontal_angle_step: float,
        vertical_fov: float,
        vertical_angle_step: float,
        max_distance: float,
        no_hit_offset: float,
        max_dynamic_tracks: int,
        motion_threshold: float,
        association_distance: float,
        velocity_smoothing: float,
        device: torch.device | str,
    ) -> None:
        self.device = torch.device(device)
        self.max_distance = float(max_distance)
        self.no_hit_offset = float(no_hit_offset)
        self.body_directions = make_body_ray_directions(
            horizontal_fov,
            horizontal_angle_step,
            vertical_fov,
            vertical_angle_step,
            device=self.device,
        )
        self.output_shape = tuple(self.body_directions.shape[:2])
        self.map = BatchedOccupancyVoxelMap(
            num_envs,
            voxel_size=voxel_size,
            map_size=map_size,
            origin_local=map_origin_local,
            device=self.device,
        )
        self.tracker = GpuDepthMotionTracker(
            num_envs,
            self.output_shape,
            max_dynamic_tracks,
            max_distance=max_distance,
            motion_threshold=motion_threshold,
            association_distance=association_distance,
            velocity_smoothing=velocity_smoothing,
            horizontal_angle_step=horizontal_angle_step,
            device=self.device,
        )

    def reset(self, env_ids: torch.Tensor | Sequence[int] | None = None) -> None:
        """Reset map and temporal tracker for selected environments."""
        self.map.reset(env_ids)
        self.tracker.reset(env_ids)

    @torch.no_grad()
    def update(
        self,
        ray_hits_w: torch.Tensor,
        ray_directions_w: torch.Tensor,
        sensor_origins_w: torch.Tensor,
        env_origins_w: torch.Tensor,
        uav_positions_w: torch.Tensor,
        heading: torch.Tensor,
        dt: float,
    ) -> GpuPerceptionOutput:
        """Run the complete batched perception chain for one policy step."""
        ray_count = math.prod(self.output_shape)
        if ray_hits_w.shape != (self.map.num_envs, ray_count, 3):
            raise ValueError(
                f"ray_hits_w must have shape {(self.map.num_envs, ray_count, 3)}; "
                f"received {tuple(ray_hits_w.shape)}"
            )
        metric = torch.linalg.vector_norm(ray_hits_w - sensor_origins_w.unsqueeze(1), dim=-1)
        metric = torch.nan_to_num(
            metric,
            nan=self.max_distance,
            posinf=self.max_distance,
            neginf=self.max_distance,
        ).clamp(0.0, self.max_distance)
        # Isaac LidarPattern is flattened [Nv,Nh].  The policy and CNN use the
        # documented NavRL matrix convention [Nh,Nv].
        depth = metric.reshape(self.map.num_envs, self.output_shape[1], self.output_shape[0]).transpose(1, 2)
        positions, velocities, sizes, dynamic_valid, motion_mask = self.tracker.update(
            depth,
            ray_hits_w,
            ray_directions_w,
            sensor_origins_w,
            dt,
        )
        static_valid = torch.isfinite(ray_hits_w).all(dim=-1)
        static_valid &= metric < self.max_distance - 1.0e-4
        static_valid &= ~motion_mask.transpose(1, 2).reshape(self.map.num_envs, ray_count)
        self.map.integrate_hits(ray_hits_w, env_origins_w, static_valid)
        static_raw, static_normalized, static_hit_mask = self.map.ray_cast(
            uav_positions_w,
            env_origins_w,
            heading,
            self.body_directions,
            max_distance=self.max_distance,
            ray_step_size=self.map.voxel_size,
            no_hit_offset=self.no_hit_offset,
        )
        return GpuPerceptionOutput(
            static_raw=static_raw,
            static_normalized=static_normalized,
            static_hit_mask=static_hit_mask,
            dynamic_positions_w=positions.clone(),
            dynamic_velocities_w=velocities.clone(),
            dynamic_sizes=sizes.clone(),
            dynamic_valid=dynamic_valid.clone(),
            motion_mask=motion_mask,
        )


__all__ = [
    "BatchedOccupancyVoxelMap",
    "GpuDepthMotionTracker",
    "GpuNavRLPerception",
    "GpuPerceptionOutput",
    "make_body_ray_directions",
]
