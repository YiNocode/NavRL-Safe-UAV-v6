"""Runtime gates for the formal V6 camera-navigation environment."""
import argparse
import traceback
from isaaclab.app import AppLauncher

TASK_ID = "Isaac-UAV-NavRL-V6-Front-Depth-M1-v0"
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--motion_steps", type=int, default=8)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app


def main():
    import gymnasium as gym
    import torch
    import navrl_uav_v5  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    if args.num_envs < 1 or args.motion_steps < 2:
        raise ValueError("num_envs must be positive and motion_steps must be at least two")
    cfg = parse_env_cfg(TASK_ID, device=args.device, num_envs=args.num_envs)
    env = gym.make(TASK_ID, cfg=cfg)
    raw = env.unwrapped
    try:
        observations, _ = env.reset()
        assert observations["front_depth"].shape == (
            args.num_envs, cfg.camera_height, cfg.camera_width, 1
        )
        assert observations["internal_state"].shape == (args.num_envs, 8)
        assert observations["dynamic_obstacles"].shape == (args.num_envs, 5, 10)
        assert torch.isfinite(observations["internal_state"]).all()
        assert torch.isfinite(observations["dynamic_obstacles"]).all()

        scene = raw.get_randomized_scene_state()
        assert torch.all(scene["static_active_mask"].sum(1) == cfg.num_static_obstacles)
        assert torch.all(scene["dynamic_active_mask"].sum(1) == cfg.num_dynamic_obstacles)
        assert torch.allclose(
            raw._start_pos_w[:, 2] - raw.scene.env_origins[:, 2],
            torch.full((args.num_envs,), 1.5, device=raw.device),
        )

        initial_position = raw._drone.data.root_pos_w.clone()
        forward = torch.tensor([1.0, 0.5, 0.5], device=raw.device).expand(args.num_envs, -1)
        rewards_seen = []
        for _ in range(args.motion_steps):
            _, reward, terminated, truncated, _ = env.step(forward)
            rewards_seen.append(reward)
            assert not truncated.any()
        control = raw.get_control_state()
        assert control["backend"] == "direct_root_velocity"
        assert torch.linalg.vector_norm(control["executed_velocity_world"], dim=-1).min() > 0.1
        displacement = torch.linalg.vector_norm(raw._drone.data.root_pos_w - initial_position, dim=-1)
        assert displacement.min() > 0.01
        rewards = torch.stack(rewards_seen)
        assert torch.isfinite(rewards).all() and torch.any(rewards != 0.0)

        # Success gate: place the goal at the current pose and let the normal
        # DirectRLEnv step order evaluate termination before automatic reset.
        raw._goal_pos_w.copy_(raw._drone.data.root_pos_w)
        neutral = torch.full((args.num_envs, 3), 0.5, device=raw.device)
        _, _, success_terminated, _, extras = env.step(neutral)
        assert success_terminated.all()
        assert extras["success"].all()
        assert "log" in extras and "Episode/success" in extras["log"]

        # Out-of-bounds gate on the freshly reset episodes.
        state = raw._drone.data.root_state_w.clone()
        state[:, 0] = raw.scene.env_origins[:, 0] + max(abs(cfg.x_range[0]), abs(cfg.x_range[1])) + 1.0
        raw._drone.write_root_pose_to_sim(state[:, :7])
        raw._drone.write_root_velocity_to_sim(torch.zeros_like(state[:, 7:]))
        _, _, bounds_terminated, _, extras = env.step(neutral)
        assert bounds_terminated.all()
        assert extras["out_of_bounds"].all()

        # Collision gate: overlap each drone with one active static collider
        # after the configured ground/contact grace window.
        scene = raw.get_randomized_scene_state()
        first_active = scene["static_active_mask"].to(torch.int64).argmax(dim=1)
        rows = torch.arange(args.num_envs, device=raw.device)
        obstacle_position = scene["static_positions_w"][rows, first_active]
        state = raw._drone.data.root_state_w.clone()
        state[:, :3] = obstacle_position
        raw._drone.write_root_pose_to_sim(state[:, :7])
        raw._drone.write_root_velocity_to_sim(torch.zeros_like(state[:, 7:]))
        raw.episode_length_buf.fill_(cfg.contact_reset_grace_steps + 1)
        _, _, collision_terminated, _, extras = env.step(neutral)
        assert collision_terminated.all()
        assert extras["collision"].all()

        # Timeout gate, isolated from task terminations after collision reset.
        raw.episode_length_buf.fill_(raw.max_episode_length)
        _, _, timeout_terminated, timeout_truncated, extras = env.step(neutral)
        assert not timeout_terminated.any() and timeout_truncated.all()
        assert extras["timeout"].all()

        assert raw._contact_sensor.data.net_forces_w_history.shape[0] == args.num_envs
        assert extras["simulator_truth_policy_input"].logical_not().all()
        print(
            f"[PASS] V6 formal gates envs={args.num_envs} "
            f"depth={tuple(observations['front_depth'].shape)} "
            f"min_displacement_m={displacement.min().item():.4f} "
            "action=true reward=true success=true collision=true out_of_bounds=true timeout=true "
            "contact_sensor=true random_scene=true metrics=true truth_leak=false",
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        print(f"[FAIL] V6 formal gates {type(error).__name__}: {error}", flush=True)
        traceback.print_exc()
        raise
    finally:
        app.close()
