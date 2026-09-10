"""Benchmark one V5 or V6 environment scale without running PPO training."""

from __future__ import annotations

import argparse
import json
import time

from isaaclab.app import AppLauncher

V5_TASK_ID = "Isaac-UAV-NavRL-V5-GPU-Direct-v0"
V6_TASK_ID = "Isaac-UAV-NavRL-V6-Front-Depth-M1-v0"
V6_VOXEL_TASK_ID = "Isaac-UAV-NavRL-V6-Front-Depth-Voxel-v0"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default=V6_TASK_ID)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--warmup_steps", type=int, default=20)
parser.add_argument("--benchmark_steps", type=int, default=100)
parser.add_argument("--profile_steps", type=int, default=10)
parser.add_argument("--camera_read_repetitions", type=int, default=50)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main() -> None:
    import gymnasium as gym
    import navrl_uav_v5  # noqa: F401
    import torch
    from isaaclab_tasks.utils import parse_env_cfg

    counts = (
        args_cli.num_envs, args_cli.warmup_steps, args_cli.benchmark_steps,
        args_cli.profile_steps, args_cli.camera_read_repetitions,
    )
    if min(counts) <= 0:
        raise ValueError("benchmark counts must be positive")
    if args_cli.benchmark_steps < 2:
        raise ValueError("benchmark_steps must be at least two for memory stability checks")
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=cfg)
    raw = env.unwrapped
    device = raw.device
    actions = torch.zeros((args_cli.num_envs, 3), device=device)

    def synchronize() -> None:
        if torch.cuda.is_available() and torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)

    def memory_mib() -> dict[str, float]:
        if not torch.cuda.is_available() or torch.device(device).type != "cuda":
            return {
                "allocated": 0.0, "reserved": 0.0, "device_used": 0.0,
                "device_free": 0.0, "device_total": 0.0,
            }
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        return {
            "allocated": torch.cuda.memory_allocated(device) / 1024**2,
            "reserved": torch.cuda.memory_reserved(device) / 1024**2,
            "device_used": (total_bytes - free_bytes) / 1024**2,
            "device_free": free_bytes / 1024**2,
            "device_total": total_bytes / 1024**2,
        }

    try:
        observations, _ = env.reset()
        for _ in range(args_cli.warmup_steps):
            observations, *_ = env.step(actions)
        synchronize()
        warmup_memory = memory_mib()
        if torch.cuda.is_available() and torch.device(device).type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        first_half = args_cli.benchmark_steps // 2
        second_half = args_cli.benchmark_steps - first_half
        start = time.perf_counter()
        for _ in range(first_half):
            observations, *_ = env.step(actions)
        synchronize()
        midpoint = time.perf_counter()
        midpoint_memory = memory_mib()
        for _ in range(second_half):
            observations, *_ = env.step(actions)
        synchronize()
        end = time.perf_counter()
        end_memory = memory_mib()
        elapsed = end - start

        component_seconds = {"pre_physics": 0.0, "apply_action": 0.0, "observation": 0.0}
        originals = {
            "pre_physics": raw._pre_physics_step,
            "apply_action": raw._apply_action,
            "observation": raw._get_observations,
        }

        def timed(name, function):
            def wrapper(*args, **kwargs):
                synchronize()
                component_start = time.perf_counter()
                value = function(*args, **kwargs)
                synchronize()
                component_seconds[name] += time.perf_counter() - component_start
                return value
            return wrapper

        raw._pre_physics_step = timed("pre_physics", originals["pre_physics"])
        raw._apply_action = timed("apply_action", originals["apply_action"])
        raw._get_observations = timed("observation", originals["observation"])
        synchronize()
        profile_start = time.perf_counter()
        for _ in range(args_cli.profile_steps):
            observations, *_ = env.step(actions)
        synchronize()
        profile_elapsed = time.perf_counter() - profile_start
        raw._pre_physics_step = originals["pre_physics"]
        raw._apply_action = originals["apply_action"]
        raw._get_observations = originals["observation"]

        camera_read_ms = None
        policy_inference_ms = None
        render_products = 0
        camera_output_mib = 0.0
        if hasattr(raw, "_camera"):
            render_products = 1
            synchronize()
            read_start = time.perf_counter()
            for _ in range(args_cli.camera_read_repetitions):
                depth = raw._camera.data.output["distance_to_image_plane"]
            synchronize()
            camera_read_ms = 1000.0 * (
                time.perf_counter() - read_start
            ) / args_cli.camera_read_repetitions
            camera_output_mib = depth.numel() * depth.element_size() / 1024**2

        if args_cli.task in (V6_TASK_ID, V6_VOXEL_TASK_ID):
            from tensordict import TensorDict
            if args_cli.task == V6_VOXEL_TASK_ID:
                from navrl_uav_v5.tasks.navrl_navigation.agents.v6_voxel_actor_critic import (
                    V6VoxelNavRLActorCritic as PolicyClass,
                )
                static_key = "front_voxel"
            else:
                from navrl_uav_v5.tasks.navrl_navigation.agents.v6_actor_critic import (
                    V6NavRLActorCritic as PolicyClass,
                )
                static_key = "front_depth"
            obs_groups = {
                "policy": [static_key, "internal_state", "dynamic_obstacles"],
                "critic": [static_key, "internal_state", "dynamic_obstacles"],
            }
            policy_observations = TensorDict(observations, batch_size=[args_cli.num_envs])
            policy = PolicyClass(
                policy_observations, obs_groups, 3,
                camera_near_m=cfg.camera_near_m, camera_far_m=cfg.camera_far_m,
            ).to(device).eval()
            with torch.inference_mode():
                for _ in range(5):
                    policy.act_inference(policy_observations)
                synchronize()
                policy_start = time.perf_counter()
                for _ in range(args_cli.profile_steps):
                    policy_actions = policy.act_inference(policy_observations)
                synchronize()
                policy_inference_ms = 1000.0 * (
                    time.perf_counter() - policy_start
                ) / args_cli.profile_steps
            if policy_actions.shape != (args_cli.num_envs, 3):
                raise RuntimeError("V6 policy inference produced an unexpected action shape")

        profile_ms = {
            name: 1000.0 * seconds / args_cli.profile_steps
            for name, seconds in component_seconds.items()
        }
        profile_total_ms = 1000.0 * profile_elapsed / args_cli.profile_steps
        profile_ms["physics_render_wrapper_residual"] = max(
            0.0, profile_total_ms - sum(profile_ms.values())
        )
        profile_ms["total"] = profile_total_ms

        reserved_growth = end_memory["reserved"] - midpoint_memory["reserved"]
        allocated_growth = end_memory["allocated"] - midpoint_memory["allocated"]
        device_growth = end_memory["device_used"] - midpoint_memory["device_used"]
        growth_allowance = max(16.0, 0.05 * max(midpoint_memory["reserved"], 1.0))
        device_growth_allowance = max(64.0, 0.05 * max(midpoint_memory["device_used"], 1.0))
        memory_stable = (
            reserved_growth <= growth_allowance
            and allocated_growth <= growth_allowance
            and device_growth <= device_growth_allowance
        )
        environment_fps = args_cli.num_envs * args_cli.benchmark_steps / elapsed
        result = {
            "task": args_cli.task,
            "num_envs": args_cli.num_envs,
            "warmup_steps": args_cli.warmup_steps,
            "benchmark_steps": args_cli.benchmark_steps,
            "profile_steps": args_cli.profile_steps,
            "wall_time_s": elapsed,
            "policy_steps_per_second": args_cli.benchmark_steps / elapsed,
            "environment_fps": environment_fps,
            "camera_fps": environment_fps * render_products,
            "render_products": render_products,
            "camera_resolution": [
                getattr(cfg, "camera_height", None), getattr(cfg, "camera_width", None)
            ],
            "camera_update_period_s": getattr(
                getattr(cfg, "front_depth_camera", None), "update_period", None
            ),
            "camera_output_mib": camera_output_mib,
            "camera_tensor_read_ms": camera_read_ms,
            "policy_inference_ms": policy_inference_ms,
            "step_latency_ms": profile_ms,
            "gpu_memory_mib": {
                "after_warmup": warmup_memory,
                "midpoint": midpoint_memory,
                "end": end_memory,
                "peak_allocated": (
                    torch.cuda.max_memory_allocated(device) / 1024**2
                    if torch.cuda.is_available() else 0.0
                ),
                "peak_reserved": (
                    torch.cuda.max_memory_reserved(device) / 1024**2
                    if torch.cuda.is_available() else 0.0
                ),
                "second_half_allocated_growth": allocated_growth,
                "second_half_reserved_growth": reserved_growth,
                "second_half_device_used_growth": device_growth,
            },
            "memory_stable": memory_stable,
            "oom": False,
        }
        if hasattr(raw, "_perception"):
            result["voxel_map_mib"] = raw._perception.map.memory_bytes / 1024**2
        if not memory_stable:
            raise RuntimeError(f"persistent CUDA memory growth: {json.dumps(result)}")
        print(f"[PASS] M7_SCALING {json.dumps(result, sort_keys=True)}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        print(f"[FAIL] M7_SCALING {type(error).__name__}: {error}", flush=True)
        raise
    finally:
        simulation_app.close()
