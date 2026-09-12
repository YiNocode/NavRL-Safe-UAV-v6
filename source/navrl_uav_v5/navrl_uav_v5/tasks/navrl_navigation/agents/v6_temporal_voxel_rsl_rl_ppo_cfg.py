"""PPO configuration for the V4 temporal-voxel experiment."""

from isaaclab.utils import configclass

from .v6_voxel_rsl_rl_ppo_cfg import V6VoxelNavRLActorCriticCfg, V6VoxelNavRLGpuPPORunnerCfg


@configclass
class V6TemporalVoxelActorCriticCfg(V6VoxelNavRLActorCriticCfg):
    voxel_channels = 4


@configclass
class V6TemporalVoxelPPORunnerCfg(V6VoxelNavRLGpuPPORunnerCfg):
    experiment_name = "uav_v6_temporal_voxel_v4"
    run_name = "formal_temporal_voxel_v4"
    policy = V6TemporalVoxelActorCriticCfg()


__all__ = ["V6TemporalVoxelActorCriticCfg", "V6TemporalVoxelPPORunnerCfg"]
