"""V6 finite-FOV voxel Actor/Critic with the established NavRL PPO heads."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Beta

from .front_voxel_encoder import FrontVoxelEncoder
from .navrl_actor_critic import NavRLActorCritic, _mlp


class V6VoxelNavRLActorCritic(NavRLActorCritic):
    """Fuse front voxel 128, dynamic tracks 64, and internal state 8."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        *,
        static_embedding_dim: int = 128,
        internal_state_dim: int = 8,
        dynamic_observation_count: int = 5,
        dynamic_state_dim: int = 10,
        voxel_channels: int = 3,
        beta_concentration_scale: float = 1.0,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        **kwargs: dict[str, Any],
    ) -> None:
        nn.Module.__init__(self)
        if actor_obs_normalization or critic_obs_normalization:
            raise ValueError("V6 voxel observations are already explicitly encoded")
        for field in (
            "actor_hidden_dims", "critic_hidden_dims", "activation", "init_noise_std",
            "noise_std_type", "state_dependent_std", "camera_near_m", "camera_far_m",
        ):
            kwargs.pop(field, None)
        if kwargs:
            raise ValueError(f"Unexpected V6VoxelNavRLActorCritic arguments: {sorted(kwargs)}")
        self.obs_groups = obs_groups
        self.static_embedding_dim = int(static_embedding_dim)
        self.internal_state_dim = int(internal_state_dim)
        self.dynamic_observation_count = int(dynamic_observation_count)
        self.dynamic_state_dim = int(dynamic_state_dim)
        self.voxel_channels = int(voxel_channels)
        self.dynamic_dim = self.dynamic_observation_count * self.dynamic_state_dim
        self.beta_concentration_scale = float(beta_concentration_scale)
        self.num_actions = int(num_actions)
        if self.static_embedding_dim != 128:
            raise ValueError("V6 voxel static_embedding_dim must be 128")
        if self.beta_concentration_scale <= 0.0:
            raise ValueError("beta_concentration_scale must be positive")
        required = {"front_voxel", "internal_state", "dynamic_obstacles"}
        for set_name in ("policy", "critic"):
            missing = required.difference(self.obs_groups[set_name])
            if missing:
                raise ValueError(f"{set_name} observation set is missing {sorted(missing)}")
        batch_size = obs.batch_size[0]
        if obs["front_voxel"].ndim != 5 or obs["front_voxel"].shape[1] != self.voxel_channels:
            raise ValueError(
                f"front_voxel must have shape (B,{self.voxel_channels},Z,Y,X)"
            )
        if obs["internal_state"].shape != (batch_size, self.internal_state_dim):
            raise ValueError("internal_state has an unexpected shape")
        if obs["dynamic_obstacles"].shape != (
            batch_size, self.dynamic_observation_count, self.dynamic_state_dim
        ):
            raise ValueError("dynamic_obstacles has an unexpected shape")

        self.static_encoder = FrontVoxelEncoder(
            self.static_embedding_dim, input_channels=self.voxel_channels
        )
        self.dynamic_extractor = _mlp((self.dynamic_dim, 128, 64))
        self.fused_feature_dim = self.static_embedding_dim + 64 + self.internal_state_dim
        if self.fused_feature_dim != 200:
            raise ValueError("V6 voxel fused feature dimension must be 200")
        self.shared_trunk = _mlp((self.fused_feature_dim, 256, 256))
        self.alpha_layer = nn.Linear(256, self.num_actions)
        self.beta_layer = nn.Linear(256, self.num_actions)
        self.value_head = nn.Linear(256, 1)
        self.softplus = nn.Softplus()
        self.distribution = None
        for module in (self.alpha_layer, self.beta_layer, self.value_head):
            nn.init.orthogonal_(module.weight, 0.01)
            nn.init.constant_(module.bias, 0.0)
        Beta.set_default_validate_args(False)
        print(
            f"V6VoxelNavRLActorCritic(front_voxel={tuple(obs['front_voxel'].shape[1:])}->128, "
            "dynamic=5x10->64, internal=8, fused=200, trunk=256x256, Beta)"
        )

    def _features(self, obs: TensorDict, observation_set: str) -> torch.Tensor:
        required = ("front_voxel", "internal_state", "dynamic_obstacles")
        if any(name not in self.obs_groups[observation_set] for name in required):
            raise ValueError(f"{observation_set} observation set is incomplete")
        static_feature = self.static_encoder(obs["front_voxel"])
        dynamic_feature = self.dynamic_extractor(obs["dynamic_obstacles"].flatten(start_dim=-2))
        fused = torch.cat((static_feature, obs["internal_state"], dynamic_feature), dim=-1)
        return self.shared_trunk(fused)


def register_v6_voxel_navrl_rsl_rl_class() -> None:
    import rsl_rl.runners.on_policy_runner as runner_module

    runner_module.V6VoxelNavRLActorCritic = V6VoxelNavRLActorCritic


__all__ = ["V6VoxelNavRLActorCritic", "register_v6_voxel_navrl_rsl_rl_class"]
