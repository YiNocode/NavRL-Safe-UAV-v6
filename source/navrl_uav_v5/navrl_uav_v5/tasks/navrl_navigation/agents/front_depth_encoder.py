"""Finite-FOV front depth encoder for the V6 policy."""

from __future__ import annotations

import torch
import torch.nn as nn


class FrontDepthEncoder(nn.Module):
    """Encode axial depth ``[B,H,W,1]`` in metres into ``[B,128]``.

    The policy input remains a finite-FOV camera image. Internally, depth is
    converted into a near-obstacle proximity channel and an explicit validity
    channel before convolution. Invalid/no-return pixels therefore never
    introduce infinities into the network and remain distinct from valid pixels
    at the far clipping plane.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        near_m: float = 0.10,
        far_m: float = 5.0,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if near_m < 0.0 or far_m <= near_m:
            raise ValueError("expected 0 <= near_m < far_m")
        self.embedding_dim = int(embedding_dim)
        self.near_m = float(near_m)
        self.far_m = float(far_m)
        self.convolutions = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 5)),
            nn.Flatten(),
        )
        self.projection = nn.Sequential(
            nn.Linear(64 * 3 * 5, self.embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(self.embedding_dim),
        )

    def prepare_input(self, front_depth_m: torch.Tensor) -> torch.Tensor:
        """Convert channel-last metric depth to finite channel-first features."""
        if front_depth_m.ndim != 4 or front_depth_m.shape[-1] != 1:
            raise ValueError(
                "front_depth_m must have shape (B,H,W,1); "
                f"received {tuple(front_depth_m.shape)}"
            )
        if not front_depth_m.is_floating_point():
            raise ValueError("front_depth_m must be a floating-point tensor")
        depth = front_depth_m[..., 0]
        valid = torch.isfinite(depth) & (depth >= self.near_m) & (depth <= self.far_m)
        clipped = torch.nan_to_num(
            depth, nan=self.far_m, posinf=self.far_m, neginf=self.near_m
        ).clamp(self.near_m, self.far_m)
        proximity = (self.far_m - clipped) / (self.far_m - self.near_m)
        proximity = proximity * valid
        return torch.stack((proximity, valid.to(depth.dtype)), dim=1)

    def forward(self, front_depth_m: torch.Tensor) -> torch.Tensor:
        """Return the fixed-size V6 static embedding."""
        encoder_input = self.prepare_input(front_depth_m)
        features = self.projection(self.convolutions(encoder_input))
        expected = (front_depth_m.shape[0], self.embedding_dim)
        if features.shape != expected:
            raise RuntimeError(
                f"FrontDepthEncoder produced {tuple(features.shape)}, expected {expected}"
            )
        return features


__all__ = ["FrontDepthEncoder"]
