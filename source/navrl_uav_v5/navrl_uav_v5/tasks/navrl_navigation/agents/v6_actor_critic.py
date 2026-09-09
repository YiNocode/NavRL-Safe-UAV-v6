"""V6 finite-FOV depth Actor/Critic with the established NavRL PPO heads."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Beta

from .front_depth_encoder import FrontDepthEncoder
from .navrl_actor_critic import NavRLActorCritic, _mlp


class V6NavRLActorCritic(NavRLActorCritic):
    """Fuse front depth 128, dynamic tracks 64, and internal state 8."""

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
        camera_near_m: float = 0.10,
        camera_far_m: float = 5.0,
        beta_concentration_scale: float = 1.0,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        **kwargs: dict[str, Any],
    ) -> None:
        nn.Module.__init__(self)
        if actor_obs_normalization or critic_obs_normalization:
            raise ValueError("V6 performs explicit sensor normalization in its depth encoder")
        for field in (
            "actor_hidden_dims", "critic_hidden_dims", "activation", "init_noise_std",
            "noise_std_type", "state_dependent_std",
        ):
            kwargs.pop(field, None)
        if kwargs:
            raise ValueError(f"Unexpected V6NavRLActorCritic arguments: {sorted(kwargs)}")

        self.obs_groups = obs_groups
        self.static_embedding_dim = int(static_embedding_dim)
        self.internal_state_dim = int(internal_state_dim)
        self.dynamic_observation_count = int(dynamic_observation_count)
        self.dynamic_state_dim = int(dynamic_state_dim)
        self.dynamic_dim = self.dynamic_observation_count * self.dynamic_state_dim
        self.beta_concentration_scale = float(beta_concentration_scale)
        self.num_actions = int(num_actions)
        if self.static_embedding_dim != 128:
            raise ValueError("V6 M6 requires static_embedding_dim=128")
        if self.beta_concentration_scale <= 0.0:
            raise ValueError("beta_concentration_scale must be positive")

        required = {"front_depth", "internal_state", "dynamic_obstacles"}
        for set_name in ("policy", "critic"):
            missing = required.difference(self.obs_groups[set_name])
            if missing:
                raise ValueError(f"{set_name} observation set is missing {sorted(missing)}")
        batch_size = obs.batch_size[0]
        front_depth = obs["front_depth"]
        if front_depth.ndim != 4 or front_depth.shape[-1] != 1:
            raise ValueError("front_depth must have shape (B,H,W,1)")
        if obs["internal_state"].shape != (batch_size, self.internal_state_dim):
            raise ValueError("internal_state has an unexpected shape")
        if obs["dynamic_obstacles"].shape != (
            batch_size, self.dynamic_observation_count, self.dynamic_state_dim
        ):
            raise ValueError("dynamic_obstacles has an unexpected shape")

        self.static_encoder = FrontDepthEncoder(
            embedding_dim=self.static_embedding_dim,
            near_m=camera_near_m,
            far_m=camera_far_m,
        )
        self.dynamic_extractor = _mlp((self.dynamic_dim, 128, 64))
        self.fused_feature_dim = self.static_embedding_dim + 64 + self.internal_state_dim
        if self.fused_feature_dim != 200:
            raise ValueError(f"V6 fused feature dimension must be 200, got {self.fused_feature_dim}")
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
            f"V6NavRLActorCritic(front_depth={tuple(front_depth.shape[1:])}->128, "
            "dynamic=5x10->64, internal=8, fused=200, trunk=256x256, Beta)"
        )

    def _features(self, obs: TensorDict, observation_set: str) -> torch.Tensor:
        required = ("front_depth", "internal_state", "dynamic_obstacles")
        if any(name not in self.obs_groups[observation_set] for name in required):
            raise ValueError(f"{observation_set} observation set is incomplete")
        static_feature = self.static_encoder(obs["front_depth"])
        dynamic_feature = self.dynamic_extractor(obs["dynamic_obstacles"].flatten(start_dim=-2))
        fused = torch.cat((static_feature, obs["internal_state"], dynamic_feature), dim=-1)
        if fused.shape[-1] != self.fused_feature_dim:
            raise RuntimeError("V6 policy fusion produced an unexpected dimension")
        return self.shared_trunk(fused)


def register_v6_navrl_rsl_rl_class() -> None:
    """Expose the V6 policy to RSL-RL's class-name based factory."""
    import rsl_rl.runners.on_policy_runner as runner_module

    runner_module.V6NavRLActorCritic = V6NavRLActorCritic


__all__ = ["V6NavRLActorCritic", "register_v6_navrl_rsl_rl_class"]
