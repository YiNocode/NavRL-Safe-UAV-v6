"""Visualize a trained V5 policy in Isaac Sim."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from isaaclab.app import AppLauncher

TASK_ID = "Isaac-UAV-NavRL-V5-GPU-Direct-v0"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=20_000)
parser.add_argument("--seed", type=int, default=2027)
parser.add_argument(
    "--third-person-viewport",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Follow env-0 with a third-person chase camera.",
)
parser.add_argument(
    "--real-time",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Throttle playback to the environment control frequency.",
)
parser.add_argument(
    "--playback-speed",
    type=float,
    default=1.0,
    help="Wall-clock playback rate relative to simulation real time (for example, 0.25 is quarter speed).",
)
parser.add_argument(
    "--single-episode",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Run exactly one episode, suppress the terminal auto-reset, and freeze the final frame.",
)
parser.add_argument(
    "--start-delay",
    type=float,
    default=0.0,
    help="Seconds to hold the initialized scene before the episode starts.",
)
parser.add_argument(
    "--hold-final-frame",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Keep the Isaac Sim window open on the terminal frame in single-episode mode.",
)
parser.add_argument(
    "--loop-episodes",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Replay complete episodes continuously; requires --single-episode.",
)
parser.add_argument(
    "--repeat-same-scene",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Reseed before every loop reset so obstacle geometry and trajectories repeat.",
)
parser.add_argument(
    "--inter-episode-delay",
    type=float,
    default=2.0,
    help="Seconds to freeze the terminal frame before starting the next loop.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main() -> None:
    import gymnasium as gym
    import isaaclab.sim as sim_utils
    import navrl_uav_v5  # noqa: F401
    import torch
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
    from rsl_rl.runners import OnPolicyRunner

    checkpoint = args_cli.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if args_cli.playback_speed <= 0.0:
        raise ValueError("--playback-speed must be positive")
    if args_cli.start_delay < 0.0:
        raise ValueError("--start-delay must be non-negative")
    if args_cli.inter_episode_delay < 0.0:
        raise ValueError("--inter-episode-delay must be non-negative")
    if args_cli.loop_episodes and not args_cli.single_episode:
        raise ValueError("--loop-episodes requires --single-episode")
    if args_cli.repeat_same_scene and not args_cli.loop_episodes:
        raise ValueError("--repeat-same-scene requires --loop-episodes")
    env_cfg = parse_env_cfg(TASK_ID, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    # Keep sensor/contact debug drawing disabled: it is not part of the policy
    # and can create heavy point-instancer updates in the GUI.  The goal and
    # chase camera below are display-only overlays.
    env_cfg.debug_vis = False
    agent_cfg = load_cfg_from_registry(TASK_ID, "rsl_rl_cfg_entry_point")
    agent_cfg.device = args_cli.device
    env = RslRlVecEnvWrapper(gym.make(TASK_ID, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device=agent_cfg.device)
    observations = env.get_observations()
    raw_env = env.unwrapped
    body_indices, _ = raw_env._drone.find_bodies("body")
    if len(body_indices) != 1:
        raise RuntimeError(f"Expected one Crazyflie body link, found {body_indices}")
    drone_body_index = int(body_indices[0])

    # DirectRLEnv normally resets a completed environment inside step(), before
    # returning control to the caller.  In single-episode visualization mode we
    # suppress only that terminal reset.  The environment was already reset
    # normally during construction, so one fixed randomized scene is shown from
    # its initial state through its terminal state without displaying the next
    # episode's obstacle layout.
    original_reset_idx = raw_env._reset_idx
    terminal_reset_suppressed = False

    if args_cli.single_episode:

        if args_cli.repeat_same_scene:
            # Establish a reset whose random state can be reproduced exactly at
            # every loop boundary, including static and dynamic obstacle state.
            env.seed(args_cli.seed)
            observations, _ = env.reset()

        def suppress_terminal_reset(env_ids: torch.Tensor) -> None:
            del env_ids
            nonlocal terminal_reset_suppressed
            terminal_reset_suppressed = True

        raw_env._reset_idx = suppress_terminal_reset

    # The red sphere is display-only: the policy continues to receive the same
    # camera/perception observation used during training, never simulator truth.
    goal_visualizer = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/NavRLV5/goal",
            markers={
                "goal": sim_utils.SphereCfg(
                    radius=0.25,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.02, 0.02)),
                )
            },
        )
    )
    viewport = None
    drone_body_prim = None
    render_link_attrs = []
    static_render_attrs = []
    dynamic_render_attrs = []
    rt_gf = None
    render_sync_update_index = 0
    render_sync_check_interval = 50
    render_position_tolerance = 1.0e-4
    render_orientation_tolerance = 1.0e-4
    last_render_sync_errors: dict[str, tuple[float, float]] = {}
    trajectory_draw = None
    trajectory_points: list[tuple[float, float, float]] = []
    camera_prim_path = "/World/NavRLThirdPersonCamera"
    if args_cli.third_person_viewport:
        if args_cli.headless:
            raise ValueError("Third-person viewport requires the Isaac Sim GUI; remove --headless")
        import omni.usd
        import carb
        from isaacsim.util.debug_draw import _debug_draw
        from omni.kit.viewport.utility import get_viewport_from_window_name
        from pxr import Gf, Usd, UsdGeom, UsdPhysics
        from usdrt import Gf as RtGf, Rt, Sdf as RtSdf, Usd as RtUsd

        viewport = get_viewport_from_window_name("Viewport")
        if viewport is None:
            raise RuntimeError("Isaac Sim did not create the main Viewport window")
        stage = omni.usd.get_context().get_stage()
        camera_prim = stage.DefinePrim(camera_prim_path, "Camera")
        camera_prim.GetAttribute("focalLength").Set(25.0)
        camera_prim.GetAttribute("clippingRange").Set(Gf.Vec2f(0.01, 1000.0))
        viewport.set_active_camera(camera_prim_path)
        drone_body_prim = stage.GetPrimAtPath("/World/envs/env_0/Drone/body")
        if not drone_body_prim.IsValid():
            raise RuntimeError("The real Crazyflie body prim is missing from the USD stage")
        collider_path = "/World/envs/env_0/Drone/body/body_collision/geometry"
        collider_prim = stage.GetPrimAtPath(collider_path)
        if not collider_prim.IsValid() or not collider_prim.HasAPI(UsdPhysics.CollisionAPI):
            raise RuntimeError(f"The official Crazyflie PhysX collider is missing: {collider_path}")
        # This GPU scene is intentionally Fabric-backed.  PhysX link poses are
        # therefore not written into ordinary USD xformOps.  Explicitly mirror
        # the five real cf2x rigid-link poses into OmniHydra's Fabric transform
        # attributes so that the rendered official asset and its collider stay
        # on the exact same PhysX poses.  These are render attributes on the
        # existing asset prims--no proxy or duplicate model is created.
        rt_stage = RtUsd.Stage.Attach(omni.usd.get_context().get_stage_id())
        rt_gf = RtGf
        for link_index, link_name in enumerate(raw_env._drone.body_names):
            link_path = f"/World/envs/env_0/Drone/{link_name}"
            rt_prim = rt_stage.GetPrimAtPath(RtSdf.Path(link_path))
            if not rt_prim.IsValid():
                raise RuntimeError(f"The real Crazyflie link prim is missing: {link_path}")
            rt_xformable = Rt.Xformable(rt_prim)
            position_attr = rt_xformable.CreateWorldPositionAttr()
            orientation_attr = rt_xformable.CreateWorldOrientationAttr()
            render_link_attrs.append((link_index, position_attr, orientation_attr))
        print(f"[V5 CF2X] Fabric render sync links: {raw_env._drone.body_names}", flush=True)

        # Obstacle poses are also updated through PhysX tensor APIs.  Fabric
        # does not mirror those values into ordinary USD xformOps, so the
        # viewport otherwise shows stale obstacle meshes at their authored USD
        # locations while collision and sensing use the new PhysX locations.
        def create_obstacle_render_attrs(prefix: str, count: int) -> list[tuple[int, object, object]]:
            attrs: list[tuple[int, object, object]] = []
            for obstacle_index in range(count):
                obstacle_path = f"/World/envs/env_0/{prefix}_{obstacle_index:03d}"
                rt_prim = rt_stage.GetPrimAtPath(RtSdf.Path(obstacle_path))
                if not rt_prim.IsValid():
                    raise RuntimeError(f"Obstacle render prim is missing: {obstacle_path}")
                rt_xformable = Rt.Xformable(rt_prim)
                attrs.append(
                    (
                        obstacle_index,
                        rt_xformable.CreateWorldPositionAttr(),
                        rt_xformable.CreateWorldOrientationAttr(),
                    )
                )
            return attrs

        static_render_attrs = create_obstacle_render_attrs(
            "StaticObstacle", raw_env.cfg.static_pool_size
        )
        dynamic_render_attrs = create_obstacle_render_attrs(
            "DynamicObstacle", raw_env.cfg.dynamic_pool_size
        )
        print(
            "[V5 OBSTACLES] Fabric render sync: "
            f"static={len(static_render_attrs)}, dynamic={len(dynamic_render_attrs)}",
            flush=True,
        )

        # Print the real asset hierarchy once.  This proves which render meshes
        # and native PhysX collision prims belong to cf2x.usd without creating
        # any display-only UAV geometry.
        print("[V5 CF2X] Real USD subtree:", flush=True)
        for prim in Usd.PrimRange(drone_body_prim):
            imageable = UsdGeom.Imageable(prim)
            visibility = "n/a"
            purpose = "n/a"
            if imageable:
                visibility = str(imageable.ComputeVisibility())
                purpose = str(imageable.ComputePurpose())
            has_collision = prim.HasAPI(UsdPhysics.CollisionAPI)
            print(
                f"[V5 CF2X] {prim.GetPath()} type={prim.GetTypeName()} "
                f"visible={visibility} purpose={purpose} collision={has_collision}",
                flush=True,
            )
        # Keep the real CollisionAPI shape active in PhysX, but hide every
        # collider overlay.  The user-requested global view should show the
        # actual Crazyflie, obstacles, goal and trajectory--not the cyan guide.
        omni.usd.get_context().get_selection().set_selected_prim_paths([], True)
        carb.settings.get_settings().set_int(
            "/persistent/physics/visualizationDisplayColliders", 0
        )
        carb.settings.get_settings().set_bool(
            "/persistent/physics/visualizationSimulationOutput", False
        )
        collider_imageable = UsdGeom.Imageable(collider_prim)
        collider_imageable.CreatePurposeAttr().Set(UsdGeom.Tokens.guide)
        trajectory_draw = _debug_draw.acquire_debug_draw_interface()
        trajectory_draw.clear_lines()
        print(
            f"[V5 CF2X] PhysX collider remains active but its visual guide is hidden: {collider_path}",
            flush=True,
        )
        # At Crazyflie scale a 200 rad/s propeller becomes a screen-wide blur
        # in a close chase view.  Turning off post-process motion blur changes
        # only rendering and keeps the real propeller joint motion intact.
        carb.settings.get_settings().set_bool("/rtx/post/motionblur/enabled", False)

    def clear_trajectory() -> None:
        """Clear the env-0 display-only trajectory at an episode boundary."""
        trajectory_points.clear()
        if trajectory_draw is not None:
            trajectory_draw.clear_lines()

    def sync_fabric_render_poses(
        label: str,
        attrs: list[tuple[int, object, object]],
        positions_w: torch.Tensor,
        orientations_w: torch.Tensor,
        *,
        validate: bool,
    ) -> tuple[float, float]:
        """Mirror PhysX poses into env-0 Fabric render attributes and optionally verify them."""
        if not attrs:
            return 0.0, 0.0
        if positions_w.shape != (len(attrs), 3) or orientations_w.shape != (len(attrs), 4):
            raise RuntimeError(
                f"{label} render-sync shape mismatch: "
                f"positions={tuple(positions_w.shape)}, orientations={tuple(orientations_w.shape)}"
            )

        # One batched device-to-host copy per tensor avoids a CUDA sync for
        # every obstacle.  Isaac quaternions and Fabric Quatf both use wxyz.
        positions = positions_w.detach().cpu().tolist()
        orientations = orientations_w.detach().cpu().tolist()
        for (object_index, position_attr, orientation_attr), position, orientation in zip(
            attrs, positions, orientations, strict=True
        ):
            if object_index < 0 or object_index >= len(attrs):
                raise RuntimeError(f"Invalid {label} render object index: {object_index}")
            px, py, pz = position
            qw, qx, qy, qz = orientation
            position_attr.Set(rt_gf.Vec3d(px, py, pz))
            orientation_attr.Set(rt_gf.Quatf(qw, qx, qy, qz))

        if not validate:
            return last_render_sync_errors.get(label, (0.0, 0.0))

        max_position_error = 0.0
        max_orientation_error = 0.0
        for (_, position_attr, orientation_attr), expected_position, expected_orientation in zip(
            attrs, positions, orientations, strict=True
        ):
            rendered_position = position_attr.Get()
            rendered_orientation = orientation_attr.Get()
            if rendered_position is None or rendered_orientation is None:
                raise RuntimeError(f"{label} Fabric render pose could not be read back")
            position_error = sum(
                (float(rendered_position[axis]) - expected_position[axis]) ** 2
                for axis in range(3)
            ) ** 0.5
            rendered_imaginary = rendered_orientation.GetImaginary()
            rendered_quaternion = (
                float(rendered_orientation.GetReal()),
                float(rendered_imaginary[0]),
                float(rendered_imaginary[1]),
                float(rendered_imaginary[2]),
            )
            # q and -q encode the same rotation, so use the smaller error.
            quaternion_error = min(
                sum((actual - expected) ** 2 for actual, expected in zip(
                    rendered_quaternion, expected_orientation, strict=True
                )) ** 0.5,
                sum((actual + expected) ** 2 for actual, expected in zip(
                    rendered_quaternion, expected_orientation, strict=True
                )) ** 0.5,
            )
            max_position_error = max(max_position_error, position_error)
            max_orientation_error = max(max_orientation_error, quaternion_error)

        if max_position_error > render_position_tolerance:
            raise RuntimeError(
                f"{label} render/PhysX position error {max_position_error:.6g} m exceeds "
                f"{render_position_tolerance:.6g} m"
            )
        if max_orientation_error > render_orientation_tolerance:
            raise RuntimeError(
                f"{label} render/PhysX quaternion error {max_orientation_error:.6g} exceeds "
                f"{render_orientation_tolerance:.6g}"
            )
        last_render_sync_errors[label] = (max_position_error, max_orientation_error)
        return max_position_error, max_orientation_error

    def update_visualization() -> None:
        """Synchronize env-0 render poses, goal marker, camera, and trajectory."""
        nonlocal render_sync_update_index
        render_sync_update_index += 1
        validate_render_sync = (
            render_sync_update_index == 1
            or render_sync_update_index % render_sync_check_interval == 0
        )
        goal_marker_position_w = raw_env._goal_pos_w[:1].clone()
        goal_marker_position_w[:, 2] += 0.25
        goal_visualizer.visualize(goal_marker_position_w)
        if render_link_attrs:
            sync_fabric_render_poses(
                "drone",
                render_link_attrs,
                raw_env._drone.data.body_pos_w[0],
                raw_env._drone.data.body_quat_w[0],
                validate=validate_render_sync,
            )
            sync_fabric_render_poses(
                "static_obstacles",
                static_render_attrs,
                raw_env._static_collection.data.object_pos_w[0],
                raw_env._static_collection.data.object_quat_w[0],
                validate=validate_render_sync,
            )
            sync_fabric_render_poses(
                "dynamic_obstacles",
                dynamic_render_attrs,
                raw_env._dynamic_collection.data.object_pos_w[0],
                raw_env._dynamic_collection.data.object_quat_w[0],
                validate=validate_render_sync,
            )
            if validate_render_sync and (
                render_sync_update_index == 1 or render_sync_update_index % 250 == 0
            ):
                summary = ", ".join(
                    f"{name}:pos={errors[0]:.2e}m,quat={errors[1]:.2e}"
                    for name, errors in last_render_sync_errors.items()
                )
                print(f"[V5 RENDER SYNC] {summary}", flush=True)
        uav_position_batch_w = raw_env._drone.data.body_pos_w[:1, drone_body_index]
        if viewport is None:
            return
        uav_position_w = uav_position_batch_w[0]
        current_point = tuple(float(value) for value in uav_position_w.detach().cpu().tolist())
        if not trajectory_points:
            trajectory_points.append(current_point)
        else:
            previous_point = trajectory_points[-1]
            displacement = sum((a - b) ** 2 for a, b in zip(current_point, previous_point)) ** 0.5
            if displacement >= 0.025:
                trajectory_draw.draw_lines(
                    [previous_point],
                    [current_point],
                    [(1.0, 0.05, 0.80, 1.0)],
                    [5.0],
                )
                trajectory_points.append(current_point)

        # Fixed global third-party view.  The 20 m x 20 m arena, obstacle field,
        # enlarged red goal marker and complete magenta flight path stay in one
        # frame.  The camera never follows or rotates with the UAV.
        env_origin_w = raw_env.scene.env_origins[0]
        eye_w = env_origin_w + torch.tensor((16.0, -19.0, 22.0), device=raw_env.device)
        target_w = env_origin_w + torch.tensor((0.0, 0.0, 2.0), device=raw_env.device)
        raw_env.sim.set_camera_view(
            eye=eye_w.detach().cpu().tolist(),
            target=target_w.detach().cpu().tolist(),
            camera_prim_path=camera_prim_path,
        )

    def print_pose_diagnostic(label: str) -> None:
        """Print the latest Fabric render/PhysX pose validation errors."""
        if not last_render_sync_errors:
            return
        summary = ", ".join(
            f"{name}:pos={errors[0]:.2e}m,quat={errors[1]:.2e}"
            for name, errors in last_render_sync_errors.items()
        )
        print(
            f"[V5 RENDER SYNC] {label} {summary}",
            flush=True,
        )

    if viewport is not None:
        viewport.set_active_camera(camera_prim_path)
    # App updates advance the active Isaac timeline.  Pause it while preparing
    # and previewing the initialized scene so physics cannot run before the PPO
    # policy supplies its first action.
    raw_env.sim.pause()
    update_visualization()
    simulation_app.update()
    print_pose_diagnostic("initialized")
    print(f"[V5] Checkpoint: {checkpoint}")
    print("[V5] Isaac visualization uses sensor-only observations and raw direct-velocity actions.")
    print("[V5] Display: fixed global view, real cf2x.usd, hidden collider guide, magenta trajectory.")
    print(f"[V5] Playback speed: {args_cli.playback_speed:.2f}x real time.")
    if args_cli.single_episode:
        print("[V5] Single-episode mode: one fixed scene; terminal auto-reset is disabled.")
        print("[V5] Dynamic obstacles move during the episode; static obstacles remain fixed.")
    if args_cli.loop_episodes:
        scene_mode = "the same deterministic scene" if args_cli.repeat_same_scene else "a new randomized scene"
        print(
            f"[V5] Loop mode: replaying complete episodes in {scene_mode}; "
            f"terminal hold={args_cli.inter_episode_delay:.1f}s."
        )
    if args_cli.start_delay > 0.0:
        print(f"[V5] Episode starts after a {args_cli.start_delay:.1f} s scene preview.")
        preview_deadline = time.monotonic() + args_cli.start_delay
        while simulation_app.is_running() and time.monotonic() < preview_deadline:
            simulation_app.update()
            time.sleep(1.0 / 30.0)
    raw_env.sim.play()
    episode_number = 1
    episode_steps = 0
    try:
        for _ in range(args_cli.steps):
            if not simulation_app.is_running():
                break
            step_start = time.perf_counter()
            with torch.inference_mode():
                observations, _, dones, _ = env.step(policy(observations))
            episode_steps += 1
            if not args_cli.single_episode and bool(dones[0].item()):
                clear_trajectory()
            update_visualization()
            # env.step() renders before the chase-camera/Fabric-link update.
            # Refresh once more so the viewport shows the current physics pose.
            if viewport is not None:
                raw_env.sim.render()
            if episode_steps == 1 or episode_steps % 250 == 0:
                print_pose_diagnostic(f"episode={episode_number} step={episode_steps}")
            if not simulation_app.is_running():
                break
            if args_cli.single_episode and bool(dones[0].item()):
                if not terminal_reset_suppressed:
                    raise RuntimeError("Terminal reset was not suppressed in single-episode mode")
                if bool(raw_env._success[0].item()):
                    result = "success"
                elif bool(raw_env._collision[0].item()):
                    result = "collision"
                elif bool(raw_env._out_of_bounds[0].item()):
                    result = "out_of_bounds"
                elif bool(raw_env._time_out[0].item()):
                    result = "timeout"
                else:
                    result = "terminated"
                print(
                    "[V5 EPISODE] "
                    f"episode={episode_number} result={result} steps={episode_steps} "
                    f"duration={float(episode_steps * raw_env.step_dt):.2f}s "
                    f"return={float(raw_env._episode_return[0].item()):.3f} "
                    f"path_length={float(raw_env._episode_path_length[0].item()):.3f}m "
                    f"final_goal_distance={float(raw_env._current_goal_distance[0].item()):.3f}m"
                )
                if args_cli.loop_episodes:
                    print(
                        f"[V5] Episode complete. Holding the terminal frame for "
                        f"{args_cli.inter_episode_delay:.1f}s before replay."
                    )
                    # Stop residual vehicle motion, then keep the Kit event
                    # loop responsive without SimulationContext.render().
                    # That method toggles /app/player/playSimulations and can
                    # emit a standalone timeline STOP event at an episode
                    # boundary, closing the Isaac app before the next reset.
                    zero_root_velocity = torch.zeros_like(raw_env._command_velocity_w)
                    raw_env._drone.write_root_velocity_to_sim(zero_root_velocity)
                    hold_deadline = time.monotonic() + args_cli.inter_episode_delay
                    while simulation_app.is_running() and time.monotonic() < hold_deadline:
                        update_visualization()
                        simulation_app.update()
                        time.sleep(1.0 / 30.0)
                    if not simulation_app.is_running():
                        break

                    # Temporarily restore the real reset, recreate the scene,
                    # then suppress only the next terminal auto-reset again.
                    raw_env._reset_idx = original_reset_idx
                    if args_cli.repeat_same_scene:
                        env.seed(args_cli.seed)
                    observations, _ = env.reset()
                    raw_env._reset_idx = suppress_terminal_reset
                    terminal_reset_suppressed = False
                    episode_number += 1
                    episode_steps = 0
                    clear_trajectory()
                    update_visualization()
                    raw_env.sim.render()
                    continue
                if args_cli.hold_final_frame:
                    # Freeze both the UAV and moving obstacles without pausing
                    # the timeline, which would invoke the app-stop callback.
                    print("[V5] Episode complete. The terminal frame is frozen; close Isaac Sim when finished.")
                    while simulation_app.is_running():
                        raw_env.sim.render()
                        time.sleep(1.0 / 30.0)
                else:
                    print("[V5] Episode complete.")
                break
            if args_cli.real_time:
                target_wall_period = raw_env.step_dt / args_cli.playback_speed
                time.sleep(max(0.0, target_wall_period - (time.perf_counter() - step_start)))
    finally:
        if args_cli.single_episode:
            raw_env._reset_idx = original_reset_idx
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
