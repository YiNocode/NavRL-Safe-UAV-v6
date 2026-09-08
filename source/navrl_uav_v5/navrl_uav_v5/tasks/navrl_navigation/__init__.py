"""Register the V5 GPU sensor-perception task."""

import gymnasium as gym

TASK_ID = "Isaac-UAV-NavRL-V5-GPU-Direct-v0"

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

__all__ = ["TASK_ID"]
