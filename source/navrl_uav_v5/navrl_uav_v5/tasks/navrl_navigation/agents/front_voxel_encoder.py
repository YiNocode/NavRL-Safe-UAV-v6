"""Finite-FOV local voxel encoder for the V6 voxel-policy variant."""

from __future__ import annotations

import torch
import torch.nn as nn


class FrontVoxelEncoder(nn.Module):
    """Encode ``[B,3,Z,Y,X]`` occupied/free/observed voxels to ``[B,128]``."""

    def __init__(self, embedding_dim: int = 128) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.embedding_dim = int(embedding_dim)
        self.convolutions = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((2, 2, 2)),
            nn.Flatten(),
        )
        self.projection = nn.Sequential(
            nn.Linear(64 * 2 * 2 * 2, self.embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(self.embedding_dim),
        )

    def forward(self, front_voxel: torch.Tensor) -> torch.Tensor:
        if front_voxel.ndim != 5 or front_voxel.shape[1] != 3:
            raise ValueError("front_voxel must have shape (B,3,Z,Y,X)")
        if not front_voxel.is_floating_point():
            raise ValueError("front_voxel must be a floating-point tensor")
        features = self.projection(self.convolutions(front_voxel))
        if features.shape != (front_voxel.shape[0], self.embedding_dim):
            raise RuntimeError("FrontVoxelEncoder produced an unexpected shape")
        return features


__all__ = ["FrontVoxelEncoder"]
