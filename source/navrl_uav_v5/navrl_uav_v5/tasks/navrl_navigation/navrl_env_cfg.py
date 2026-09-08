"""Configuration for V5 sensor-only, direct-velocity NavRL training."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, MultiMeshRayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from navrl_uav_v5.assets import DRONE_CFG


@configclass
class VoxelMapCfg:
    """Per-environment CUDA voxel-map dimensions in metres."""

    voxel_size = 0.25
    map_size = (20.0, 20.0, 5.0)
    origin_local = (-10.0, -10.0, 0.0)


@configclass
class VirtualRayCfg:
    """NavRL virtual scan geometry; output convention is ``[Nh,Nv]``."""

    horizontal_fov = 360.0
    horizontal_angle_step = 10.0
    vertical_fov = 60.0
    vertical_angle_step = 10.0
    max_distance = 5.0
    no_hit_offset = 0.1


@configclass
class MotionPerceptionCfg:
    """GPU temporal-depth detection and tracking parameters."""

    max_tracks = 5
    motion_threshold = 0.035
    association_distance = 1.25
    velocity_smoothing = 0.70


@configclass
class NavRLGpuEnvCfg(DirectRLEnvCfg):
    """V2 direct plant plus the V5 GPU NavRL observation pipeline.

    The simulator exposes only a depth-ray sensor to the perception module.
    Obstacle transforms are retained privately for scene motion and objective
    collision/evaluation labels, never for policy observations or shaping.
    """

    seed = 5
    decimation = 2
    episode_length_s = 12.0
    action_space = 3
    state_space = 0

    voxel: VoxelMapCfg = VoxelMapCfg()
    raycasting: VirtualRayCfg = VirtualRayCfg()
    motion: MotionPerceptionCfg = MotionPerceptionCfg()
    internal_state_dim = 8
    dynamic_state_dim = 10
    observation_space = {
        "static_obstacles": [
            int(round(raycasting.horizontal_fov / raycasting.horizontal_angle_step)),
            int(round(raycasting.vertical_fov / raycasting.vertical_angle_step)) + 1,
        ],
        "internal_state": internal_state_dim,
        "dynamic_obstacles": [motion.max_tracks, dynamic_state_dim],
    }

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 100.0,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    # RTX 4070 validation: 4096 envs completed both a finite rollout and a full
    # PPO update at ~23.3k collected+optimized steps/s. Lower this on GPUs with
    # less memory; the voxel map alone uses 2.0 GiB at this default.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=22.0,
        replicate_physics=True,
    )
    drone: ArticulationCfg = DRONE_CFG
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Drone/body",
        update_period=0.0,
        history_length=3,
        debug_vis=False,
    )
    depth_sensor: MultiMeshRayCasterCfg = MultiMeshRayCasterCfg(
        prim_path="/World/envs/env_.*/Drone/body",
        mesh_prim_paths=[
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/StaticObstacle_.*",
                is_shared=True,
                # Static means zero velocity during an episode.  The templates
                # are nevertheless moved independently at reset, so Warp must
                # refresh their per-environment poses before ray casting.
                track_mesh_transforms=True,
            ),
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/DynamicObstacle_.*",
                track_mesh_transforms=True,
            ),
        ],
        update_period=0.0,
        ray_alignment="base",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=observation_space["static_obstacles"][1],
            vertical_fov_range=(-0.5 * raycasting.vertical_fov, 0.5 * raycasting.vertical_fov),
            horizontal_fov_range=(0.0, raycasting.horizontal_fov),
            horizontal_res=raycasting.horizontal_angle_step,
        ),
        max_distance=raycasting.max_distance,
        debug_vis=False,
    )

    x_range = (-10.0, 10.0)
    y_range = (-10.0, 10.0)
    # Low-altitude navigation contract.  The Crazyflie body origin starts
    # 0.10 m above the plane so its collider is not spawned interpenetrating
    # the ground, while still appearing and behaving as a ground take-off.
    # Goal heights remain below the obstacle canopy, and moving cuboids are
    # randomized only inside the same operating volume.
    z_range = (0.10, 1.80)
    takeoff_start_height = 0.10
    goal_z_range = (0.80, 1.60)
    altitude_corridor_tolerance = 0.20
    soft_flight_ceiling = 1.80
    min_start_goal_distance = 6.0
    max_start_goal_distance = 15.0
    max_goal_distance = max_start_goal_distance
    endpoint_obstacle_clearance = 1.25
    num_static_obstacles = 40
    # Collider dimensions cannot be changed reliably after PhysX starts.  A
    # larger pre-spawned template pool therefore supplies per-episode width,
    # shape and height randomization without rebuilding the USD stage.
    static_pool_size = 48
    static_obstacle_seed = 53
    static_minimum_separation = 0.50
    # Every static obstacle extends above the 2.20 m hard flight ceiling, even
    # after accounting for the UAV body.  The policy must therefore navigate
    # around the forest rather than learn an above-canopy shortcut.
    static_height_range = (2.60, 5.00)

    dynamic_pool_size = 24
    num_dynamic_obstacles = 15
    dynamic_obstacle_seed = 401
    dynamic_local_range = (4.0, 4.0, 1.5)
    dynamic_speed_range = (0.45, 1.25)
    dynamic_goal_threshold = 0.25
    dynamic_minimum_separation = 0.20
    dynamic_static_clearance = 0.15
    dynamic_sensing_range = 4.0
    dynamic_width_resolution = 0.25

    max_velocity_xy = 2.0
    max_velocity_z = 1.0
    collision_radius = 0.30
    collision_force_threshold = 0.10
    # Ignore brief skid/ground contact during the first 0.2 s of take-off.
    # Persistent contact after this grace period is still a collision.
    contact_reset_grace_steps = 10
    termination_min_z = 0.05
    termination_max_z = 2.20
    goal_threshold = 0.50

    internal_goal_distance_scale = max_goal_distance
    internal_vertical_distance_scale = termination_max_z - termination_min_z
    internal_velocity_scale = max_velocity_xy

    reward_progress_weight = 3.0
    reward_velocity_weight = 0.20
    reward_static_safety_weight = 0.08
    reward_dynamic_safety_weight = 0.08
    reward_action_weight = 0.005
    reward_smoothness_weight = 0.03
    reward_altitude_weight = 4.0
    reward_time_penalty = 0.01
    reward_goal_bonus = 25.0
    reward_collision_penalty = 25.0
    reward_out_of_bounds_penalty = 25.0
    debug_checks = True
    debug_vis = False


__all__ = ["MotionPerceptionCfg", "NavRLGpuEnvCfg", "VirtualRayCfg", "VoxelMapCfg"]
