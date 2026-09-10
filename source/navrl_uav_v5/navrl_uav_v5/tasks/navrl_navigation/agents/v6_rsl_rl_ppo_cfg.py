"""Formal PPO configuration for V6 finite-FOV camera observations."""

from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import NavRLActorCriticCfg, NavRLGpuPPORunnerCfg


@configclass
class V6NavRLActorCriticCfg(NavRLActorCriticCfg):
    class_name = "V6NavRLActorCritic"
    camera_near_m = 0.10
    camera_far_m = 5.0


@configclass
class V6NavRLGpuPPORunnerCfg(NavRLGpuPPORunnerCfg):
    """V6 formal runner; CLI overrides remain available for smoke tests."""

    num_steps_per_env = 32
    max_iterations = 10_000
    save_interval = 250
    experiment_name = "uav_v6_front_depth"
    run_name = "formal_camera_navigation"
    obs_groups = {
        "policy": ["front_depth", "internal_state", "dynamic_obstacles"],
        "critic": ["front_depth", "internal_state", "dynamic_obstacles"],
    }
    policy = V6NavRLActorCriticCfg()


__all__ = ["V6NavRLActorCriticCfg", "V6NavRLGpuPPORunnerCfg"]
