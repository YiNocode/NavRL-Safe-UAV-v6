"""Compact CNN encoder for NavRL static virtual range matrices."""

from __future__ import annotations

import torch
import torch.nn as nn


class StaticObstacleEncoder(nn.Module):
    """Encode normalized scans ``[B,1,Nh,Nv]`` into ``[B,E]`` features."""

    def __init__(self, embedding_dim: int = 128) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.embedding_dim = int(embedding_dim)
        self.convolutions = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=(2, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 2)),
            nn.Flatten(),
        )
        self.projection = nn.Sequential(
            nn.Linear(32 * 4 * 2, self.embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(self.embedding_dim),
        )

    def forward(self, static_obstacles: torch.Tensor) -> torch.Tensor:
        """Encode a channel-first normalized obstacle-distance matrix."""
        if static_obstacles.ndim != 4 or static_obstacles.shape[1] != 1:
            raise ValueError(
                "static_obstacles must have shape (B,1,Nh,Nv); "
                f"received {tuple(static_obstacles.shape)}"
            )
        features = self.projection(self.convolutions(static_obstacles))
        if features.shape != (static_obstacles.shape[0], self.embedding_dim):
            raise RuntimeError("StaticObstacleEncoder produced an unexpected shape")
        return features


__all__ = ["StaticObstacleEncoder"]
