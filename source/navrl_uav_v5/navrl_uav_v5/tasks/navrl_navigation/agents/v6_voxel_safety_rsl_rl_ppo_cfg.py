"""PPO configuration for the V6 voxel safety-reward experiment."""

from isaaclab.utils import configclass

from .v6_voxel_rsl_rl_ppo_cfg import (
    V6VoxelNavRLActorCriticCfg,
    V6VoxelNavRLGpuPPORunnerCfg,
)


@configclass
class V6VoxelSafetyNavRLGpuPPORunnerCfg(V6VoxelNavRLGpuPPORunnerCfg):
    experiment_name = "uav_v6_front_depth_voxel_safety_reward"
    run_name = "formal_voxel_safety_reward_v2"
    policy = V6VoxelNavRLActorCriticCfg()


__all__ = ["V6VoxelSafetyNavRLGpuPPORunnerCfg"]
