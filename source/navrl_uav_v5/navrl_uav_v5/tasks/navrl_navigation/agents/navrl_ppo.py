"""RSL-RL adapter for NavRL's published PPO update.

The rollout runner and storage remain RSL-RL, while return normalization,
Huber critic loss, parameter groups, and the update equations follow NavRL's
public ``isaac-training/training/scripts/ppo.py`` at commit
3725bcc2e7c1be4ecf1455d922299ae85042603a.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO
from torch.distributions import Beta, Independent, kl_divergence

from .navrl_actor_critic import NavRLActorCritic


class ValueNorm(nn.Module):
    """NavRL's exponentially weighted, debiased return normalizer."""

    def __init__(self, input_shape: int = 1, beta: float = 0.995, epsilon: float = 1.0e-5) -> None:
        super().__init__()
        self.beta = float(beta)
        self.epsilon = float(epsilon)
        self.register_buffer("running_mean", torch.zeros(input_shape))
        self.register_buffer("running_mean_sq", torch.zeros(input_shape))
        self.register_buffer("debiasing_term", torch.tensor(0.0))

    def running_mean_var(self) -> tuple[torch.Tensor, torch.Tensor]:
        denominator = self.debiasing_term.clamp_min(self.epsilon)
        mean = self.running_mean / denominator
        mean_sq = self.running_mean_sq / denominator
        return mean, (mean_sq - mean.square()).clamp_min(1.0e-2)

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        dimensions = tuple(range(values.ndim - 1))
        batch_mean = values.mean(dim=dimensions)
        batch_sq_mean = values.square().mean(dim=dimensions)
        self.running_mean.mul_(self.beta).add_(batch_mean * (1.0 - self.beta))
        self.running_mean_sq.mul_(self.beta).add_(batch_sq_mean * (1.0 - self.beta))
        self.debiasing_term.mul_(self.beta).add_(1.0 - self.beta)

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        mean, variance = self.running_mean_var()
        return (values - mean) / torch.sqrt(variance)

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        mean, variance = self.running_mean_var()
        return values * torch.sqrt(variance) + mean


class NavRLPPO(PPO):
    """NavRL PPO math on top of RSL-RL's runner/storage interface."""

    def __init__(self, policy: NavRLActorCritic, **kwargs) -> None:
        if not isinstance(policy, NavRLActorCritic):
            raise TypeError("NavRLPPO requires NavRLActorCritic")
        self.hard_kl_multiplier = float(kwargs.pop("hard_kl_multiplier", 1.5))
        self.min_policy_updates_before_kl_stop = int(kwargs.pop("min_policy_updates_before_kl_stop", 2))
        if self.hard_kl_multiplier <= 0.0:
            raise ValueError("hard_kl_multiplier must be positive")
        if self.min_policy_updates_before_kl_stop < 1:
            raise ValueError("min_policy_updates_before_kl_stop must be at least one")
        super().__init__(policy, **kwargs)
        if self.rnd is not None or self.symmetry is not None:
            raise ValueError("NavRL's published PPO does not use RND or symmetry augmentation")

        self.value_norm = ValueNorm(1).to(self.device)
        # Register ValueNorm below the policy so normalizer buffers are saved by
        # the unmodified RSL-RL runner checkpoint format.
        self.policy.value_norm = self.value_norm
        feature_parameters = list(self.policy.static_encoder.parameters())
        feature_parameters += list(self.policy.dynamic_extractor.parameters())
        feature_parameters += list(self.policy.shared_trunk.parameters())
        actor_parameters = list(self.policy.alpha_layer.parameters()) + list(self.policy.beta_layer.parameters())
        critic_parameters = list(self.policy.value_head.parameters())
        grouped_parameters = feature_parameters + actor_parameters + critic_parameters
        if {id(parameter) for parameter in grouped_parameters} != {
            id(parameter) for parameter in self.policy.parameters()
        }:
            raise RuntimeError("NavRL optimizer parameter partition is incomplete")
        # Three parameter groups in one Adam are mathematically equivalent to
        # the upstream three Adam instances because their learning rates and
        # step schedules are identical. This also preserves RSL-RL checkpointing.
        self.optimizer = torch.optim.Adam(
            (
                {"params": feature_parameters, "name": "feature_extractor"},
                {"params": actor_parameters, "name": "actor"},
                {"params": critic_parameters, "name": "critic"},
            ),
            lr=self.learning_rate,
        )
        self._actor_parameters = actor_parameters
        self._critic_parameters = critic_parameters
        self.critic_loss_fn = nn.HuberLoss(delta=10.0)

    def act(self, obs):
        """Sample an action while freezing the matching observation snapshot.

        V5 intentionally reuses CUDA observation buffers to avoid allocations.
        RSL-RL normally stores ``transition.observations`` only after
        ``env.step()``, by which time those buffers have been updated in place.
        Cloning here preserves the required (observation, action, old-policy)
        correspondence in rollout storage.
        """
        actions = super().act(obs)
        self.transition.observations = obs.clone()
        return actions

    def process_env_step(self, obs, rewards: torch.Tensor, dones: torch.Tensor, extras: dict) -> None:
        """Store transitions while bootstrapping timeouts in unnormalized value units."""
        self.policy.update_normalization(obs)
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if "time_outs" in extras:
            timeout_value = self.value_norm.denormalize(self.transition.values)
            self.transition.rewards += self.gamma * torch.squeeze(
                timeout_value * extras["time_outs"].unsqueeze(1).to(self.device), 1
            )
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, obs) -> None:
        """Compute GAE using denormalized values, then normalize returns."""
        last_value_normalized = self.policy.evaluate(obs).detach()
        last_value = self.value_norm.denormalize(last_value_normalized)
        values = self.value_norm.denormalize(self.storage.values)
        advantage = torch.zeros_like(last_value)
        returns = torch.zeros_like(self.storage.returns)
        for step in reversed(range(self.storage.num_transitions_per_env)):
            next_value = last_value if step == self.storage.num_transitions_per_env - 1 else values[step + 1]
            not_terminated = 1.0 - self.storage.dones[step].float()
            delta = self.storage.rewards[step] + self.gamma * next_value * not_terminated - values[step]
            advantage = delta + self.gamma * self.lam * not_terminated * advantage
            returns[step] = advantage + values[step]
        advantages = returns - values
        self.storage.advantages = (advantages - advantages.mean()) / (advantages.std() + 1.0e-7)
        self.value_norm.update(returns)
        self.storage.returns.copy_(self.value_norm.normalize(returns))

    def update(self) -> dict[str, float]:
        """Run NavRL's clipped Beta-policy PPO and Huber critic update."""
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_actor_grad_norm = 0.0
        mean_critic_grad_norm = 0.0
        mean_total_grad_norm = 0.0
        mean_approx_kl = 0.0
        max_approx_kl = 0.0
        applied_updates = 0
        kl_early_stop = False
        initial_kl = 0.0
        initial_mean_delta = 0.0
        initial_std_delta = 0.0
        observations = self.storage.observations.flatten(0, 1)
        actions = self.storage.actions.flatten(0, 1)
        target_values = self.storage.values.flatten(0, 1)
        advantages = self.storage.advantages.flatten(0, 1)
        returns = self.storage.returns.flatten(0, 1)
        old_action_mean = self.storage.mu.flatten(0, 1)
        old_action_std = self.storage.sigma.flatten(0, 1)
        usable_batch_size = (actions.shape[0] // self.num_mini_batches) * self.num_mini_batches
        mini_batch_size = usable_batch_size // self.num_mini_batches

        # Upstream calls make_batch() once per epoch, so each epoch receives a
        # fresh permutation and drops only a possible indivisible remainder.
        for _epoch in range(self.num_learning_epochs):
            permutation = torch.randperm(usable_batch_size, device=self.device).reshape(
                self.num_mini_batches, mini_batch_size
            )
            for indices in permutation:
                obs_batch = observations[indices]
                actions_batch = actions[indices]
                target_values_batch = target_values[indices]
                advantages_batch = advantages[indices]
                returns_batch = returns[indices]
                old_distribution = self._beta_from_moments(
                    old_action_mean[indices], old_action_std[indices]
                )

                self.policy.act(obs_batch)
                log_prob = self.policy.get_actions_log_prob(actions_batch)
                old_log_prob = old_distribution.log_prob(actions_batch.clamp(1.0e-6, 1.0 - 1.0e-6))
                value = self.policy.evaluate(obs_batch)
                entropy = self.policy.entropy.mean()

                log_ratio = log_prob - old_log_prob
                ratio_flat = torch.exp(log_ratio)
                if not isinstance(self.policy.distribution, Independent):
                    raise TypeError("NavRLPPO requires an Independent Beta action distribution")
                exact_kl = kl_divergence(old_distribution, self.policy.distribution).mean()
                approx_kl_value = float(exact_kl)
                if applied_updates == 0:
                    initial_kl = approx_kl_value
                    initial_mean_delta = float(
                        (self.policy.action_mean - old_action_mean[indices]).abs().mean()
                    )
                    initial_std_delta = float(
                        (self.policy.action_std - old_action_std[indices]).abs().mean()
                    )
                max_approx_kl = max(max_approx_kl, approx_kl_value)
                # PPO clipping bounds the objective, not the actual parameter update.
                # Stop the remaining mini-batches when the sampled joint-action KL
                # exceeds the configured hard trust-region limit.
                if (
                    self.desired_kl is not None
                    and applied_updates >= self.min_policy_updates_before_kl_stop
                    and approx_kl_value > self.desired_kl * self.hard_kl_multiplier
                ):
                    kl_early_stop = True
                    break

                ratio = ratio_flat.unsqueeze(-1)
                surrogate = advantages_batch * ratio
                surrogate_clipped = advantages_batch * ratio.clamp(
                    1.0 - self.clip_param, 1.0 + self.clip_param
                )
                # The public NavRL trainer multiplies the mean clipped objective by
                # the action dimension after using an Independent Beta distribution.
                surrogate_loss = -torch.minimum(surrogate, surrogate_clipped).mean() * self.policy.num_actions

                value_clipped = target_values_batch + (value - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_loss_original = self.critic_loss_fn(returns_batch, value)
                value_loss_clipped = self.critic_loss_fn(returns_batch, value_clipped)
                value_loss = torch.maximum(value_loss_original, value_loss_clipped)
                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                actor_grad_norm = self._gradient_norm(self._actor_parameters)
                critic_grad_norm = self._gradient_norm(self._critic_parameters)
                # Clip once across every trainable parameter.  The previous code
                # omitted the shared CNN/trunk parameters, allowing large feature
                # updates even when actor/critic head gradients appeared bounded.
                total_grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_entropy += entropy.item()
                mean_actor_grad_norm += float(actor_grad_norm)
                mean_critic_grad_norm += float(critic_grad_norm)
                mean_total_grad_norm += float(total_grad_norm)
                mean_approx_kl += approx_kl_value
                applied_updates += 1

            if kl_early_stop:
                break

        if applied_updates == 0:
            raise RuntimeError("KL guard prevented every PPO optimizer update")
        self.storage.clear()
        return {
            "value_function": mean_value_loss / applied_updates,
            "surrogate": mean_surrogate_loss / applied_updates,
            "entropy": mean_entropy / applied_updates,
            "actor_grad_norm": mean_actor_grad_norm / applied_updates,
            "critic_grad_norm": mean_critic_grad_norm / applied_updates,
            "total_grad_norm": mean_total_grad_norm / applied_updates,
            "approx_kl": mean_approx_kl / applied_updates,
            "max_approx_kl": max_approx_kl,
            "policy_updates": float(applied_updates),
            "kl_early_stop": float(kl_early_stop),
            "initial_kl": initial_kl,
            "initial_mean_delta": initial_mean_delta,
            "initial_std_delta": initial_std_delta,
        }

    @staticmethod
    def _gradient_norm(parameters: list[nn.Parameter]) -> torch.Tensor:
        """Return the pre-clipping L2 gradient norm for one parameter group."""
        gradients = [parameter.grad.detach().norm(2) for parameter in parameters if parameter.grad is not None]
        if not gradients:
            return torch.tensor(0.0)
        return torch.stack(gradients).norm(2)

    @staticmethod
    def _beta_from_moments(mean: torch.Tensor, std: torch.Tensor) -> Independent:
        """Reconstruct the rollout Beta distribution from stored mean/std."""
        mean = mean.clamp(1.0e-6, 1.0 - 1.0e-6)
        variance = std.square().clamp_min(1.0e-10)
        total_concentration = (mean * (1.0 - mean) / variance - 1.0).clamp_min(1.0e-4)
        alpha = (mean * total_concentration).clamp_min(1.0e-4)
        beta = ((1.0 - mean) * total_concentration).clamp_min(1.0e-4)
        return Independent(Beta(alpha, beta), 1)


def register_navrl_ppo_class() -> None:
    """Expose NavRLPPO to RSL-RL's class-name based runner factory."""
    import rsl_rl.runners.on_policy_runner as runner_module

    runner_module.NavRLPPO = NavRLPPO


__all__ = ["NavRLPPO", "ValueNorm", "register_navrl_ppo_class"]
