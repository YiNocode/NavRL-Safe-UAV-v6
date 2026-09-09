"""Register the V5 GPU sensor-perception task."""

import gymnasium as gym

TASK_ID = "Isaac-UAV-NavRL-V5-GPU-Direct-v0"
V6_M1_TASK_ID = "Isaac-UAV-NavRL-V6-Front-Depth-M1-v0"

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
        kwargs={"env_cfg_entry_point":f"{__name__}.v6_camera_env:V6FrontDepthCameraEnvCfg"},
    )

__all__ = ["TASK_ID", "V6_M1_TASK_ID"]
