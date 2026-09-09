"""M6 one-iteration PPO configuration for V6 camera observations."""

from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import NavRLActorCriticCfg, NavRLGpuPPORunnerCfg


@configclass
class V6NavRLActorCriticCfg(NavRLActorCriticCfg):
    class_name = "V6NavRLActorCritic"
    camera_near_m = 0.10
    camera_far_m = 5.0


@configclass
class V6NavRLGpuPPORunnerCfg(NavRLGpuPPORunnerCfg):
    """Small M6 smoke config; formal training parameters are selected at M8."""

    num_steps_per_env = 8
    max_iterations = 1
    save_interval = 1
    experiment_name = "uav_v6_front_depth"
    run_name = "m6_ppo_smoke"
    obs_groups = {
        "policy": ["front_depth", "internal_state", "dynamic_obstacles"],
        "critic": ["front_depth", "internal_state", "dynamic_obstacles"],
    }
    policy = V6NavRLActorCriticCfg()


__all__ = ["V6NavRLActorCriticCfg", "V6NavRLGpuPPORunnerCfg"]
