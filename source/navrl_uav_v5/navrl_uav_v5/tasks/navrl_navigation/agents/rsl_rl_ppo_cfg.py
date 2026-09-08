"""High-throughput PPO configuration for V5's structured observation."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class NavRLActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name = "NavRLActorCritic"
    init_noise_std = 1.0
    actor_obs_normalization = False
    critic_obs_normalization = False
    actor_hidden_dims = []
    critic_hidden_dims = []
    activation = "elu"
    static_embedding_dim = 128
    internal_state_dim = 8
    dynamic_observation_count = 5
    dynamic_state_dim = 10
    beta_concentration_scale = 1.0


@configclass
class NavRLPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """NavRL PPO controls including a hard sampled-KL update guard."""

    hard_kl_multiplier = 1.5
    min_policy_updates_before_kl_stop = 1


@configclass
class NavRLGpuPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Large-batch NavRL PPO; rollout, CNN, and optimizer stay on CUDA."""

    seed = 42
    device = "cuda:0"
    num_steps_per_env = 32
    max_iterations = 10_000
    save_interval = 250
    experiment_name = "uav_v5_navrl_gpu"
    run_name = "static40_dynamic15_sensor_only"
    logger = "tensorboard"
    obs_groups = {
        "policy": ["static_obstacles", "internal_state", "dynamic_obstacles"],
        "critic": ["static_obstacles", "internal_state", "dynamic_obstacles"],
    }
    clip_actions = 1.0
    best_metric_key = "Episode/success"
    best_metric_window = 25
    best_metric_min_delta = 0.002
    early_stopping_patience = 500
    early_stopping_warmup = 50
    early_stopping_max_drop = 0.08
    early_stopping_degradation_patience = 50

    policy = NavRLActorCriticCfg()
    algorithm = NavRLPpoAlgorithmCfg(
        class_name="NavRLPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.1,
        entropy_coef=1.0e-3,
        num_learning_epochs=2,
        num_mini_batches=8,
        learning_rate=2.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.005,
        max_grad_norm=1.0,
    )


__all__ = ["NavRLActorCriticCfg", "NavRLGpuPPORunnerCfg", "NavRLPpoAlgorithmCfg"]
