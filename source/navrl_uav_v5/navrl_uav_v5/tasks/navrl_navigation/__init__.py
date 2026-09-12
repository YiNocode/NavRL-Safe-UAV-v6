"""Register the V5 GPU sensor-perception task."""

import gymnasium as gym

TASK_ID = "Isaac-UAV-NavRL-V5-GPU-Direct-v0"
V6_M1_TASK_ID = "Isaac-UAV-NavRL-V6-Front-Depth-M1-v0"
V6_VOXEL_TASK_ID = "Isaac-UAV-NavRL-V6-Front-Depth-Voxel-v0"
V6_VOXEL_SAFETY_TASK_ID = "Isaac-UAV-NavRL-V6-Front-Depth-Voxel-SafetyReward-v0"

if TASK_ID not in gym.registry:
    gym.register(
        id=TASK_ID,
        entry_point=f"{__name__}.navrl_env:NavRLGpuEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.navrl_env_cfg:NavRLGpuEnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:NavRLGpuPPORunnerCfg",
        },
    )

if V6_M1_TASK_ID not in gym.registry:
    gym.register(
        id=V6_M1_TASK_ID,
        entry_point=f"{__name__}.v6_camera_env:V6FrontDepthCameraEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point":f"{__name__}.v6_camera_env:V6FrontDepthCameraEnvCfg",
            "rsl_rl_cfg_entry_point":f"{__name__}.agents.v6_rsl_rl_ppo_cfg:V6NavRLGpuPPORunnerCfg",
        },
    )

if V6_VOXEL_TASK_ID not in gym.registry:
    gym.register(
        id=V6_VOXEL_TASK_ID,
        entry_point=f"{__name__}.v6_voxel_env:V6FrontDepthVoxelEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.v6_voxel_env:V6FrontDepthVoxelEnvCfg",
            "rsl_rl_cfg_entry_point": (
                f"{__name__}.agents.v6_voxel_rsl_rl_ppo_cfg:V6VoxelNavRLGpuPPORunnerCfg"
            ),
        },
    )

if V6_VOXEL_SAFETY_TASK_ID not in gym.registry:
    gym.register(
        id=V6_VOXEL_SAFETY_TASK_ID,
        entry_point=f"{__name__}.v6_voxel_safety_env:V6FrontDepthVoxelSafetyEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                f"{__name__}.v6_voxel_safety_env:V6FrontDepthVoxelSafetyEnvCfg"
            ),
            "rsl_rl_cfg_entry_point": (
                f"{__name__}.agents.v6_voxel_safety_rsl_rl_ppo_cfg:"
                "V6VoxelSafetyNavRLGpuPPORunnerCfg"
            ),
        },
    )

__all__ = ["TASK_ID", "V6_M1_TASK_ID", "V6_VOXEL_TASK_ID", "V6_VOXEL_SAFETY_TASK_ID"]
