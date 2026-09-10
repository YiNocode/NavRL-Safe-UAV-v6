"""PPO configuration for the finite-FOV front-depth voxel variant."""

from isaaclab.utils import configclass

from .v6_rsl_rl_ppo_cfg import V6NavRLActorCriticCfg, V6NavRLGpuPPORunnerCfg


@configclass
class V6VoxelNavRLActorCriticCfg(V6NavRLActorCriticCfg):
    class_name = "V6VoxelNavRLActorCritic"


@configclass
class V6VoxelNavRLGpuPPORunnerCfg(V6NavRLGpuPPORunnerCfg):
    experiment_name = "uav_v6_front_depth_voxel"
    run_name = "formal_camera_voxel_navigation"
    obs_groups = {
        "policy": ["front_voxel", "internal_state", "dynamic_obstacles"],
        "critic": ["front_voxel", "internal_state", "dynamic_obstacles"],
    }
    policy = V6VoxelNavRLActorCriticCfg()


__all__ = ["V6VoxelNavRLActorCriticCfg", "V6VoxelNavRLGpuPPORunnerCfg"]
