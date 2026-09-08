"""NavRL structured-observation CNN/MLP actor-critic with a Beta policy."""

from __future__ import annotations

from typing import Any, NoReturn

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Beta, Independent

from .static_obstacle_encoder import StaticObstacleEncoder


def _mlp(units: tuple[int, ...] | list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for input_size, output_size in zip(units[:-1], units[1:], strict=True):
        layers.extend((nn.Linear(input_size, output_size), nn.LeakyReLU(), nn.LayerNorm(output_size)))
    return nn.Sequential(*layers)


class NavRLActorCritic(nn.Module):
    """RSL-RL compatible port of NavRL's public PPO network.

    The static ``[B,Nh,Nv]`` matrix remains structured until it enters the CNN.
    Internal and dynamic states are fused only after their respective encoders.
    """

    is_recurrent = False

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
        beta_concentration_scale: float = 1.0,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        **kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        if actor_obs_normalization or critic_obs_normalization:
            raise ValueError("NavRL's published trainer does not normalize policy observations")
        # Standard ActorCritic-only fields are accepted so this class can use
        # Isaac Lab's normal RSL-RL config schema.
        kwargs.pop("actor_hidden_dims", None)
        kwargs.pop("critic_hidden_dims", None)
        kwargs.pop("activation", None)
        kwargs.pop("init_noise_std", None)
        kwargs.pop("noise_std_type", None)
        kwargs.pop("state_dependent_std", None)
        if kwargs:
            raise ValueError(f"Unexpected NavRLActorCritic arguments: {sorted(kwargs)}")

        self.obs_groups = obs_groups
        self.static_embedding_dim = int(static_embedding_dim)
        self.internal_state_dim = int(internal_state_dim)
        self.dynamic_observation_count = int(dynamic_observation_count)
        self.dynamic_state_dim = int(dynamic_state_dim)
        self.beta_concentration_scale = float(beta_concentration_scale)
        if self.beta_concentration_scale <= 0.0:
            raise ValueError("beta_concentration_scale must be positive")
        self.dynamic_dim = self.dynamic_observation_count * self.dynamic_state_dim
        self.num_actions = int(num_actions)

        required = {"static_obstacles", "internal_state", "dynamic_obstacles"}
        for set_name in ("policy", "critic"):
            missing = required.difference(self.obs_groups[set_name])
            if missing:
                raise ValueError(f"{set_name} observation set is missing {sorted(missing)}")
        static_obstacles = obs["static_obstacles"]
        internal_state = obs["internal_state"]
        dynamic_obstacles = obs["dynamic_obstacles"]
        if static_obstacles.ndim != 3:
            raise ValueError("static_obstacles must have shape (B,Nh,Nv)")
        if internal_state.shape != (obs.batch_size[0], self.internal_state_dim):
            raise ValueError("internal_state has an unexpected shape")
        if dynamic_obstacles.shape != (
            obs.batch_size[0],
            self.dynamic_observation_count,
            self.dynamic_state_dim,
        ):
            raise ValueError("dynamic_obstacles has an unexpected shape")

        self.static_encoder = StaticObstacleEncoder(self.static_embedding_dim)
        self.dynamic_extractor = _mlp((self.dynamic_dim, 128, 64))
        self.shared_trunk = _mlp((self.static_embedding_dim + self.internal_state_dim + 64, 256, 256))
        self.alpha_layer = nn.Linear(256, self.num_actions)
        self.beta_layer = nn.Linear(256, self.num_actions)
        self.value_head = nn.Linear(256, 1)
        self.softplus = nn.Softplus()
        self.distribution: Independent | None = None

        for module in (self.alpha_layer, self.beta_layer, self.value_head):
            nn.init.orthogonal_(module.weight, 0.01)
            nn.init.constant_(module.bias, 0.0)
        Beta.set_default_validate_args(False)
        print(
            "NavRLActorCritic(static=1x"
            f"{static_obstacles.shape[-2]}x{static_obstacles.shape[-1]}->{self.static_embedding_dim}, "
            f"dynamic=1x{self.dynamic_observation_count}x{self.dynamic_state_dim}->64, trunk=256x256, Beta)"
        )

    def forward(self) -> NoReturn:
        raise NotImplementedError

    def reset(self, dones: torch.Tensor | None = None) -> None:
        return None

    def _features(self, obs: TensorDict, observation_set: str) -> torch.Tensor:
        required = ("static_obstacles", "internal_state", "dynamic_obstacles")
        if any(name not in self.obs_groups[observation_set] for name in required):
            raise ValueError(f"{observation_set} observation set is incomplete")
        static = obs["static_obstacles"]
        internal = obs["internal_state"]
        dynamic = obs["dynamic_obstacles"].flatten(start_dim=-2)
        static_feature = self.static_encoder(static.unsqueeze(-3))
        dynamic_feature = self.dynamic_extractor(dynamic)
        return self.shared_trunk(torch.cat((static_feature, internal, dynamic_feature), dim=-1))

    def _concentrations(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        features = self._features(obs, "policy")
        alpha = 1.0 + self.softplus(self.alpha_layer(features)) + 1.0e-6
        beta = 1.0 + self.softplus(self.beta_layer(features)) + 1.0e-6
        if self.beta_concentration_scale != 1.0:
            # Temper the Beta policy without shifting the deterministic actor
            # mean. This is the Beta analogue of reducing Gaussian action std.
            total = (alpha + beta) * self.beta_concentration_scale
            mean = alpha / (alpha + beta)
            alpha = mean * total
            beta = (1.0 - mean) * total
        return alpha, beta

    def _update_distribution(self, obs: TensorDict) -> None:
        alpha, beta = self._concentrations(obs)
        self.distribution = Independent(Beta(alpha, beta), 1)

    @property
    def action_mean(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Action distribution has not been initialized")
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Action distribution has not been initialized")
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Action distribution has not been initialized")
        return self.distribution.entropy()

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        self._update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        alpha, beta = self._concentrations(obs)
        return alpha / (alpha + beta)

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        return self.value_head(self._features(obs, "critic"))

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Action distribution has not been initialized")
        return self.distribution.log_prob(actions.clamp(1.0e-6, 1.0 - 1.0e-6))

    def update_normalization(self, obs: TensorDict) -> None:
        return None

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        super().load_state_dict(state_dict, strict=strict)
        return True


def register_navrl_rsl_rl_class() -> None:
    """Expose the custom policy to RSL-RL's class-name based runner factory."""
    import rsl_rl.runners.on_policy_runner as runner_module

    runner_module.NavRLActorCritic = NavRLActorCritic


__all__ = ["NavRLActorCritic", "register_navrl_rsl_rl_class"]
