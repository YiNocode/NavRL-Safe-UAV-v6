"""PPO configuration for the V6 voxel reward-v3 experiment."""

from isaaclab.utils import configclass

from .v6_voxel_rsl_rl_ppo_cfg import V6VoxelNavRLActorCriticCfg, V6VoxelNavRLGpuPPORunnerCfg


@configclass
class V6VoxelRewardV3NavRLGpuPPORunnerCfg(V6VoxelNavRLGpuPPORunnerCfg):
    experiment_name = "uav_v6_front_depth_voxel_reward_v3"
    run_name = "formal_voxel_reward_v3"
    policy = V6VoxelNavRLActorCriticCfg()


__all__ = ["V6VoxelRewardV3NavRLGpuPPORunnerCfg"]
