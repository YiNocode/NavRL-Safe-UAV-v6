"""Train V5 with CUDA perception, structured NavRL PPO, and no control override."""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "Isaac-UAV-NavRL-V5-GPU-Direct-v0"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default=TASK_ID)
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument(
    "--learning_rate",
    type=float,
    default=None,
    help="Optional PPO learning-rate override for distribution-shift fine-tuning.",
)
parser.add_argument("--run_name", default=None)
parser.add_argument("--resume", type=Path, default=None)
parser.add_argument(
    "--reset_optimizer_on_resume",
    action="store_true",
    help="Load policy/value-normalizer weights but start with a fresh optimizer.",
)
parser.add_argument(
    "--initial_best_metric",
    type=float,
    default=None,
    help="Independent evaluation score for the resumed checkpoint.",
)
parser.add_argument(
    "--initial_best_checkpoint",
    type=Path,
    default=None,
    help="Checkpoint to preserve as model_best.pt when it differs from --resume.",
)
parser.add_argument("--logger", choices=("tensorboard", "wandb"), default="tensorboard")
parser.add_argument("--wandb_project", default="navrl-safe-uav-v5")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main() -> None:
    import gymnasium as gym
    import navrl_uav_v5  # noqa: F401
    import torch
    from isaaclab.utils.io import dump_yaml
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
    from navrl_uav_v5.tasks.navrl_navigation.agents.stability_runner import (
        StabilityOnPolicyRunner,
        TrainingEarlyStopped,
    )

    if args_cli.num_envs <= 0:
        raise ValueError("--num_envs must be positive")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg.seed = args_cli.seed
    agent_cfg.device = args_cli.device
    agent_cfg.logger = args_cli.logger
    agent_cfg.wandb_project = args_cli.wandb_project
    if args_cli.max_iterations is not None:
        if args_cli.max_iterations <= 0:
            raise ValueError("--max_iterations must be positive")
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.learning_rate is not None:
        if args_cli.learning_rate <= 0.0:
            raise ValueError("--learning_rate must be positive")
        agent_cfg.algorithm.learning_rate = args_cli.learning_rate
    if args_cli.run_name:
        agent_cfg.run_name = args_cli.run_name

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = f"_{agent_cfg.run_name}" if agent_cfg.run_name else ""
    log_dir = PROJECT_ROOT / "logs" / "rsl_rl" / agent_cfg.experiment_name / f"{timestamp}{suffix}"
    log_dir.mkdir(parents=True, exist_ok=True)
    dump_yaml(str(log_dir / "env.yaml"), env_cfg)
    dump_yaml(str(log_dir / "agent.yaml"), agent_cfg)
    if agent_cfg.logger == "wandb":
        os.environ.setdefault("WANDB_PROJECT", args_cli.wandb_project)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = StabilityOnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=str(log_dir),
        device=agent_cfg.device,
    )
    if args_cli.resume is not None:
        checkpoint = args_cli.resume.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        runner.load(str(checkpoint), load_optimizer=not args_cli.reset_optimizer_on_resume)
        if args_cli.reset_optimizer_on_resume:
            learning_rate = float(agent_cfg.algorithm.learning_rate)
            runner.alg.learning_rate = learning_rate
            for parameter_group in runner.alg.optimizer.param_groups:
                parameter_group["lr"] = learning_rate
        if args_cli.initial_best_metric is not None:
            best_checkpoint = (
                args_cli.initial_best_checkpoint.expanduser().resolve()
                if args_cli.initial_best_checkpoint is not None
                else checkpoint
            )
            runner.initialize_best_checkpoint(best_checkpoint, args_cli.initial_best_metric)
        runner.current_learning_iteration += 1

    raw = env.unwrapped
    if hasattr(raw,"_perception"):
        print(
            f"[V5] envs={raw.num_envs}, device={raw.device}, "
            f"voxel_map={raw._perception.map.memory_bytes / 1024**2:.1f} MiB, "
            "input=depth->GPU voxel/raycast/motion tracks, control=direct velocity, "
            "ROS2=false, PX4=false, action shield=false",flush=True)
    else:
        print(
            f"[V6 M6] envs={raw.num_envs}, device={raw.device}, "
            "front_depth->128 + dynamic_5x10->64 + internal_8 = fused_200, "
            "simulator_truth_policy_input=false",flush=True)
    training_finished = False
    try:
        try:
            runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
        except TrainingEarlyStopped as error:
            print(f"[STABILITY] Training stopped safely: {error}", flush=True)
            training_finished = True
        else:
            runner.save(str(log_dir / "model_final.pt"))
            training_finished = True
    finally:
        # Isaac Sim can translate SIGINT/SIGTERM into a clean SystemExit.  Save
        # the current optimizer and policy before closing so a service restart
        # continues from the latest update instead of falling back to the best
        # evaluation checkpoint.
        if not training_finished:
            runner.save(str(log_dir / "model_interrupted.pt"))
            print(f"[V5] Saved interrupted training state: {log_dir / 'model_interrupted.pt'}", flush=True)
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
