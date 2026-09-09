"""Batched GPU geometry for V6 axial camera depth."""
from __future__ import annotations

import torch


def backproject_axial_depth(depth_m: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """Backproject [N,H,W,1] axial depth to ROS optical points [N,H,W,3]."""
    if depth_m.ndim != 4 or depth_m.shape[-1] != 1:
        raise ValueError("depth_m must have shape [N,H,W,1]")
    n, height, width, _ = depth_m.shape
    if intrinsics.shape != (n, 3, 3):
        raise ValueError("intrinsics must have shape [N,3,3]")
    if depth_m.device != intrinsics.device:
        raise ValueError("depth and intrinsics must share a device")
    u = torch.arange(width, device=depth_m.device, dtype=depth_m.dtype).view(1, 1, width)
    v = torch.arange(height, device=depth_m.device, dtype=depth_m.dtype).view(1, height, 1)
    z = depth_m[..., 0]
    x = (u - intrinsics[:, None, None, 0, 2]) / intrinsics[:, None, None, 0, 0] * z
    y = (v - intrinsics[:, None, None, 1, 2]) / intrinsics[:, None, None, 1, 1] * z
    return torch.stack((x.expand_as(z), y.expand_as(z), z), dim=-1)


def optical_points_to_world(
    points: torch.Tensor, position_w: torch.Tensor, quaternion_w_ros: torch.Tensor
) -> torch.Tensor:
    """Transform ROS optical-frame points to world using wxyz camera poses."""
    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError("points must have shape [N,H,W,3]")
    n = points.shape[0]
    if position_w.shape != (n, 3) or quaternion_w_ros.shape != (n, 4):
        raise ValueError("camera pose has an unexpected shape")
    q = quaternion_w_ros / quaternion_w_ros.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    qw = q[:, None, None, 0:1]
    qv = q[:, None, None, 1:4]
    t = 2.0 * torch.cross(qv.expand_as(points), points, dim=-1)
    rotated = points + qw * t + torch.cross(qv.expand_as(points), t, dim=-1)
    return rotated + position_w[:, None, None, :]


__all__ = ["backproject_axial_depth", "optical_points_to_world"]
