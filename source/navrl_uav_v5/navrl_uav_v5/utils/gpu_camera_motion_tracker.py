"""Sensor-only temporal motion tracking for V6 finite-FOV depth images."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from navrl_uav_v5.utils.gpu_camera_geometry import backproject_axial_depth, optical_points_to_world
from navrl_uav_v5.utils.math import world_to_goal_frame


@dataclass(frozen=True)
class CameraDynamicObservation:
    """Tracked camera observations and their NavRL-compatible policy encoding."""

    state: torch.Tensor
    positions_w: torch.Tensor
    velocities_w: torch.Tensor
    sizes: torch.Tensor
    valid: torch.Tensor
    missed_frames: torch.Tensor
    motion_mask: torch.Tensor


def _rotate_wxyz(vectors: torch.Tensor, quaternion: torch.Tensor) -> torch.Tensor:
    quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    while quaternion.ndim < vectors.ndim:
        quaternion = quaternion.unsqueeze(1)
    qw = quaternion[..., 0:1]
    qv = quaternion[..., 1:4].expand_as(vectors)
    cross = 2.0 * torch.cross(qv, vectors, dim=-1)
    return vectors + qw * cross + torch.cross(qv, cross, dim=-1)


class GpuCameraMotionTracker:
    """Batched depth-residual detector with ego compensation and track aging."""

    def __init__(
        self,
        num_envs: int,
        image_shape: tuple[int, int],
        max_tracks: int = 5,
        *,
        near_m: float = 0.1,
        far_m: float = 5.0,
        motion_threshold_m: float = 0.08,
        association_distance_m: float = 0.75,
        velocity_smoothing: float = 0.7,
        max_missed_frames: int = 3,
        nms_kernel: int = 9,
        width_resolution_m: float = 0.25,
        device: torch.device | str = "cpu",
    ) -> None:
        height, width = image_shape
        if num_envs <= 0 or height <= 0 or width <= 0 or max_tracks <= 0:
            raise ValueError("batch, image dimensions, and max_tracks must be positive")
        if not 0.0 <= velocity_smoothing < 1.0:
            raise ValueError("velocity_smoothing must lie in [0,1)")
        if nms_kernel <= 0 or nms_kernel % 2 == 0:
            raise ValueError("nms_kernel must be a positive odd integer")
        if max_missed_frames < 0:
            raise ValueError("max_missed_frames must be non-negative")
        self.num_envs = int(num_envs)
        self.height = int(height)
        self.width = int(width)
        self.max_tracks = int(max_tracks)
        self.near_m = float(near_m)
        self.far_m = float(far_m)
        self.motion_threshold_m = float(motion_threshold_m)
        self.association_distance_m = float(association_distance_m)
        self.velocity_smoothing = float(velocity_smoothing)
        self.max_missed_frames = int(max_missed_frames)
        self.nms_kernel = int(nms_kernel)
        self.width_resolution_m = float(width_resolution_m)
        self.device = torch.device(device)
        self.previous_depth = torch.full(
            (num_envs, height, width, 1), torch.inf, device=self.device
        )
        self.previous_position_w = torch.zeros((num_envs, 3), device=self.device)
        self.previous_quaternion_w_ros = torch.zeros((num_envs, 4), device=self.device)
        self.previous_quaternion_w_ros[:, 0] = 1.0
        self.has_previous_frame = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.positions_w = torch.zeros((num_envs, max_tracks, 3), device=self.device)
        self.velocities_w = torch.zeros_like(self.positions_w)
        self.sizes = torch.zeros_like(self.positions_w)
        self.valid = torch.zeros((num_envs, max_tracks), dtype=torch.bool, device=self.device)
        self.missed_frames = torch.zeros((num_envs, max_tracks), dtype=torch.int64, device=self.device)
        self.last_matched_count = 0
        self.last_detection_count = 0
        self.last_max_measured_speed = 0.0
        self.last_min_association_distance = torch.inf

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        ids = (
            torch.arange(self.num_envs, device=self.device)
            if env_ids is None
            else torch.as_tensor(env_ids, dtype=torch.int64, device=self.device)
        )
        self.previous_depth[ids] = torch.inf
        self.has_previous_frame[ids] = False
        self.positions_w[ids] = 0.0
        self.velocities_w[ids] = 0.0
        self.sizes[ids] = 0.0
        self.valid[ids] = False
        self.missed_frames[ids] = 0

    def _previous_depth_in_current_camera(
        self,
        intrinsics: torch.Tensor,
        position_w: torch.Tensor,
        quaternion_w_ros: torch.Tensor,
    ) -> torch.Tensor:
        """Reproject the previous depth image into the current optical frame."""
        previous_points_c = backproject_axial_depth(self.previous_depth, intrinsics)
        previous_points_w = optical_points_to_world(
            previous_points_c, self.previous_position_w, self.previous_quaternion_w_ros
        )
        relative_w = previous_points_w - position_w[:, None, None, :]
        inverse_quaternion = quaternion_w_ros.clone()
        inverse_quaternion[:, 1:] *= -1.0
        current_points_c = _rotate_wxyz(relative_w, inverse_quaternion)
        x, y, z = current_points_c.unbind(dim=-1)
        u = torch.round(intrinsics[:, None, None, 0, 0] * x / z + intrinsics[:, None, None, 0, 2]).long()
        v = torch.round(intrinsics[:, None, None, 1, 1] * y / z + intrinsics[:, None, None, 1, 2]).long()
        projectable = (
            torch.isfinite(current_points_c).all(dim=-1)
            & (z >= self.near_m)
            & (z <= self.far_m)
            & (u >= 0)
            & (u < self.width)
            & (v >= 0)
            & (v < self.height)
        )
        flat_index = (v.clamp(0, self.height - 1) * self.width + u.clamp(0, self.width - 1)).flatten(1)
        source_depth = torch.where(projectable, z, torch.full_like(z, torch.inf)).flatten(1)
        predicted = torch.full(
            (self.num_envs, self.height * self.width), torch.inf,
            dtype=self.previous_depth.dtype, device=self.device,
        )
        predicted.scatter_reduce_(1, flat_index, source_depth, reduce="amin", include_self=True)
        return predicted.view(self.num_envs, self.height, self.width)

    def _select_detections(self, score: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Select spatially separated detections with GPU suppression."""
        work = score.clone()
        indices = []
        scores = []
        pixel_ids = torch.arange(self.height * self.width, device=self.device)
        pixel_v = (pixel_ids // self.width).view(1, -1)
        pixel_u = (pixel_ids % self.width).view(1, -1)
        radius = self.nms_kernel // 2
        for _ in range(self.max_tracks):
            selected_score, selected_index = work.max(dim=1)
            indices.append(selected_index)
            scores.append(selected_score)
            selected_v = (selected_index // self.width).unsqueeze(1)
            selected_u = (selected_index % self.width).unsqueeze(1)
            suppress = (pixel_v - selected_v).abs() <= radius
            suppress &= (pixel_u - selected_u).abs() <= radius
            work.masked_fill_(suppress, 0.0)
        return torch.stack(indices, dim=1), torch.stack(scores, dim=1)

    def _encode(
        self,
        robot_position_w: torch.Tensor,
        start_to_goal_w: torch.Tensor,
    ) -> torch.Tensor:
        relative = self.positions_w - robot_position_w[:, None, :]
        distance = torch.linalg.vector_norm(relative, dim=-1, keepdim=True)
        goal_frame = start_to_goal_w[:, None, :].expand_as(relative)
        relative_g = world_to_goal_frame(relative, goal_frame)
        velocity_g = world_to_goal_frame(self.velocities_w, goal_frame)
        direction_g = relative_g / distance.clamp_min(1.0e-6)
        distance_xy = torch.linalg.vector_norm(relative_g[..., :2], dim=-1, keepdim=True)
        distance_z = relative_g[..., 2:3]
        width_category = self.sizes[..., 0:1] / self.width_resolution_m - 1.0
        state = torch.cat(
            (direction_g, distance_xy, distance_z, velocity_g, width_category, self.sizes[..., 2:3]),
            dim=-1,
        )
        return torch.where(self.valid.unsqueeze(-1), state, torch.zeros_like(state))

    @torch.no_grad()
    def update(
        self,
        depth_m: torch.Tensor,
        intrinsics: torch.Tensor,
        position_w: torch.Tensor,
        quaternion_w_ros: torch.Tensor,
        robot_position_w: torch.Tensor,
        start_to_goal_w: torch.Tensor,
        dt: float,
    ) -> CameraDynamicObservation:
        expected_depth = (self.num_envs, self.height, self.width, 1)
        if depth_m.shape != expected_depth:
            raise ValueError(f"depth_m must have shape {expected_depth}")
        if intrinsics.shape != (self.num_envs, 3, 3):
            raise ValueError("intrinsics has an unexpected shape")
        if position_w.shape != (self.num_envs, 3) or quaternion_w_ros.shape != (self.num_envs, 4):
            raise ValueError("camera pose has an unexpected shape")
        if robot_position_w.shape != (self.num_envs, 3) or start_to_goal_w.shape != (self.num_envs, 3):
            raise ValueError("navigation frame inputs have unexpected shapes")
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        predicted_depth = self._previous_depth_in_current_camera(
            intrinsics, position_w, quaternion_w_ros
        )
        current_depth = depth_m[..., 0]
        current_valid = torch.isfinite(current_depth) & (current_depth >= self.near_m) & (current_depth <= self.far_m)
        predicted_valid = torch.isfinite(predicted_depth)
        residual = (current_depth - predicted_depth).abs()
        motion_mask = (
            current_valid
            & predicted_valid
            & self.has_previous_frame[:, None, None]
            & (residual >= self.motion_threshold_m)
        )
        motion_mask = functional.max_pool2d(
            motion_mask[:, None].float(), 3, stride=1, padding=1
        ).squeeze(1).bool()
        candidate_mask = motion_mask & current_valid & torch.isfinite(residual)
        foreground_weight = (1.0 - current_depth / self.far_m).clamp_min(0.05)
        score_image = torch.where(
            candidate_mask, residual * foreground_weight, torch.zeros_like(residual)
        ).flatten(1)
        selected_index, selected_score = self._select_detections(score_image)
        detected = selected_score > 0.0

        points_c = backproject_axial_depth(depth_m, intrinsics)
        points_w = optical_points_to_world(points_c, position_w, quaternion_w_ros).flatten(1, 2)
        gather3 = selected_index.unsqueeze(-1).expand(-1, -1, 3)
        candidates = torch.gather(points_w, 1, gather3)
        selected_depth = torch.gather(current_depth.flatten(1), 1, selected_index)
        fx = intrinsics[:, 0, 0].unsqueeze(1)
        fy = intrinsics[:, 1, 1].unsqueeze(1)
        estimated_width = (selected_depth * self.nms_kernel / fx).clamp(0.10, 1.50)
        estimated_height = (selected_depth * self.nms_kernel / fy).clamp(0.10, 1.50)
        sizes = torch.stack((estimated_width, estimated_width, estimated_height), dim=-1)
        ray_w = candidates - position_w[:, None, :]
        ray_w = ray_w / torch.linalg.vector_norm(ray_w, dim=-1, keepdim=True).clamp_min(1.0e-6)
        candidates = candidates + 0.5 * estimated_width.unsqueeze(-1) * ray_w

        old_positions = self.positions_w.clone()
        predicted_positions = old_positions + self.velocities_w * dt
        old_valid = self.valid.clone()
        new_positions = predicted_positions.clone()
        new_velocities = self.velocities_w.clone()
        new_sizes = self.sizes.clone()
        new_missed = self.missed_frames + old_valid.to(torch.int64)
        observed = torch.zeros_like(old_valid)
        claimed = torch.zeros_like(old_valid)
        env_ids = torch.arange(self.num_envs, device=self.device)
        matched_count = 0
        max_measured_speed = torch.zeros((), device=self.device)
        min_association_distance = torch.full((), torch.inf, device=self.device)

        for detection_index in range(self.max_tracks):
            candidate = candidates[:, detection_index]
            available = old_valid & ~claimed
            distances = torch.linalg.vector_norm(predicted_positions - candidate[:, None, :], dim=-1)
            distances = torch.where(available, distances, torch.full_like(distances, torch.inf))
            nearest_distance, nearest_slot = distances.min(dim=1)
            matched = detected[:, detection_index] & (nearest_distance <= self.association_distance_m)
            if detected[:, detection_index].any():
                min_association_distance = torch.minimum(
                    min_association_distance, nearest_distance[detected[:, detection_index]].min()
                )
            matched_count += int(matched.sum().item())
            free = ~old_valid & ~claimed
            free_slot = free.to(torch.int64).argmax(dim=1)
            has_free = free.any(dim=1)
            replaceable = ~claimed
            replacement_score = torch.where(
                replaceable, new_missed, torch.full_like(new_missed, -1)
            )
            replacement_slot = replacement_score.argmax(dim=1)
            has_replacement = replaceable.any(dim=1)
            new_slot = torch.where(has_free, free_slot, replacement_slot)
            slot = torch.where(matched, nearest_slot, new_slot)
            active = detected[:, detection_index] & (matched | has_free | has_replacement)
            active_env = env_ids[active]
            active_slot = slot[active]
            if active_env.numel() == 0:
                continue
            previous = old_positions[active_env, active_slot]
            elapsed_frames = self.missed_frames[active_env, active_slot] + 1
            elapsed = elapsed_frames.to(candidate.dtype).unsqueeze(-1) * dt
            measured_velocity = ((candidate[active] - previous) / elapsed).clamp(-5.0, 5.0)
            was_matched = matched[active].unsqueeze(-1)
            if was_matched.any():
                measured_speed = torch.linalg.vector_norm(measured_velocity[was_matched[:, 0]], dim=-1)
                max_measured_speed = torch.maximum(max_measured_speed, measured_speed.max())
            filtered_velocity = self.velocity_smoothing * new_velocities[active_env, active_slot]
            filtered_velocity += (1.0 - self.velocity_smoothing) * measured_velocity
            new_positions[active_env, active_slot] = candidate[active]
            new_velocities[active_env, active_slot] = torch.where(
                was_matched, filtered_velocity, torch.zeros_like(filtered_velocity)
            )
            new_sizes[active_env, active_slot] = sizes[:, detection_index][active]
            new_missed[active_env, active_slot] = 0
            observed[active_env, active_slot] = True
            claimed[active_env, active_slot] = True
            old_valid[active_env, active_slot] = True

        new_valid = old_valid & (new_missed <= self.max_missed_frames)
        self.positions_w.copy_(torch.where(new_valid.unsqueeze(-1), new_positions, torch.zeros_like(new_positions)))
        self.velocities_w.copy_(torch.where(new_valid.unsqueeze(-1), new_velocities, torch.zeros_like(new_velocities)))
        self.sizes.copy_(torch.where(new_valid.unsqueeze(-1), new_sizes, torch.zeros_like(new_sizes)))
        self.valid.copy_(new_valid)
        self.missed_frames.copy_(torch.where(new_valid, new_missed, torch.zeros_like(new_missed)))
        self.previous_depth.copy_(depth_m)
        self.previous_position_w.copy_(position_w)
        self.previous_quaternion_w_ros.copy_(quaternion_w_ros)
        self.has_previous_frame.fill_(True)
        self.last_matched_count = matched_count
        self.last_detection_count = int(detected.sum().item())
        self.last_max_measured_speed = float(max_measured_speed.item())
        self.last_min_association_distance = float(min_association_distance.item())
        return CameraDynamicObservation(
            state=self._encode(robot_position_w, start_to_goal_w),
            positions_w=self.positions_w.clone(),
            velocities_w=self.velocities_w.clone(),
            sizes=self.sizes.clone(),
            valid=self.valid.clone(),
            missed_frames=self.missed_frames.clone(),
            motion_mask=motion_mask,
        )


__all__ = ["CameraDynamicObservation", "GpuCameraMotionTracker"]
