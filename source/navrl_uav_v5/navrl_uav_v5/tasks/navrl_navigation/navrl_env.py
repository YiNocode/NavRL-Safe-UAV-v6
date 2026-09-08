"""V5 high-throughput NavRL environment built on V2 direct velocity control."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation, RigidObjectCfg, RigidObjectCollection, RigidObjectCollectionCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, MultiMeshRayCaster
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_from_euler_xyz

from navrl_uav_v5.utils.gpu_navrl_perception import GpuNavRLPerception, GpuPerceptionOutput
from navrl_uav_v5.utils.math import goal_frame_to_world
from navrl_uav_v5.utils.navrl_dynamic import DynamicObservation, navrl_dynamic_observation, navrl_internal_state
from navrl_uav_v5.utils.navrl_scene import sample_dynamic_obstacles
from navrl_uav_v5.utils.obstacles import (
    contact_forces_to_collision,
    out_of_bounds_mask,
    sample_obstacle_specs,
)

from .navrl_env_cfg import NavRLGpuEnvCfg


def quaternion_wxyz_to_yaw(quaternion: torch.Tensor) -> torch.Tensor:
    """Return Z-up yaw from Isaac's ``(w,x,y,z)`` quaternion tensor."""
    w, x, y, z = quaternion.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))


class NavRLGpuEnv(DirectRLEnv):
    """Sensor-only NavRL PPO task with a V2 direct-velocity plant.

    The executed command is always the raw bounded policy command.  V5 has no
    action projector, velocity-obstacle layer, deadlock override, PX4 model, or
    ROS2 transport.  Static and dynamic policy observations are produced only
    from depth-ray tensors by :class:`GpuNavRLPerception`.
    """

    cfg: NavRLGpuEnvCfg

    def __init__(self, cfg: NavRLGpuEnvCfg, render_mode: str | None = None, **kwargs):
        expected_static = (
            int(round(cfg.raycasting.horizontal_fov / cfg.raycasting.horizontal_angle_step)),
            int(round(cfg.raycasting.vertical_fov / cfg.raycasting.vertical_angle_step)) + 1,
        )
        if tuple(cfg.observation_space["static_obstacles"]) != expected_static:
            raise ValueError("static observation declaration does not match ray configuration")
        if cfg.static_pool_size < cfg.num_static_obstacles:
            raise ValueError("static_pool_size must be at least num_static_obstacles")
        if cfg.dynamic_pool_size <= 0 or cfg.dynamic_pool_size % 8 != 0:
            raise ValueError("dynamic_pool_size must be a positive multiple of eight")
        if not 0 < cfg.num_dynamic_obstacles <= cfg.dynamic_pool_size:
            raise ValueError("num_dynamic_obstacles must lie in [1, dynamic_pool_size]")
        if not cfg.termination_min_z < cfg.takeoff_start_height < cfg.soft_flight_ceiling:
            raise ValueError(
                "takeoff_start_height must lie between termination_min_z and soft_flight_ceiling"
            )
        if not (
            cfg.takeoff_start_height
            < cfg.goal_z_range[0]
            <= cfg.goal_z_range[1]
            <= cfg.soft_flight_ceiling
            < cfg.termination_max_z
        ):
            raise ValueError(
                "goal_z_range must be above take-off and at/below the soft ceiling, "
                "which must be below termination_max_z"
            )
        if not cfg.z_range[0] <= cfg.takeoff_start_height <= cfg.z_range[1]:
            raise ValueError("takeoff_start_height must lie inside z_range")
        if cfg.altitude_corridor_tolerance < 0.0:
            raise ValueError("altitude_corridor_tolerance must be non-negative")
        if (
            len(cfg.static_height_range) != 2
            or cfg.static_height_range[0] <= cfg.termination_max_z
            or cfg.static_height_range[1] < cfg.static_height_range[0]
            or cfg.static_height_range[1] > cfg.voxel.map_size[2]
        ):
            raise ValueError(
                "static_height_range must lie above termination_max_z and inside the voxel-map height"
            )
        super().__init__(cfg, render_mode, **kwargs)

        init_phase_start = time.perf_counter()
        self._perception = GpuNavRLPerception(
            self.num_envs,
            voxel_size=cfg.voxel.voxel_size,
            map_size=cfg.voxel.map_size,
            map_origin_local=cfg.voxel.origin_local,
            horizontal_fov=cfg.raycasting.horizontal_fov,
            horizontal_angle_step=cfg.raycasting.horizontal_angle_step,
            vertical_fov=cfg.raycasting.vertical_fov,
            vertical_angle_step=cfg.raycasting.vertical_angle_step,
            max_distance=cfg.raycasting.max_distance,
            no_hit_offset=cfg.raycasting.no_hit_offset,
            max_dynamic_tracks=cfg.motion.max_tracks,
            motion_threshold=cfg.motion.motion_threshold,
            association_distance=cfg.motion.association_distance,
            velocity_smoothing=cfg.motion.velocity_smoothing,
            device=self.device,
        )
        if self._perception.output_shape != tuple(cfg.observation_space["static_obstacles"]):
            raise RuntimeError("GPU perception output shape differs from the observation space")
        print(f"[V5 INIT] perception buffers: {time.perf_counter() - init_phase_start:.2f}s", flush=True)

        self._actions = torch.full((self.num_envs, 3), 0.5, device=self.device)
        self._command_velocity_g = torch.zeros((self.num_envs, 3), device=self.device)
        self._command_velocity_w = torch.zeros((self.num_envs, 6), device=self.device)
        self._start_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._goal_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._start_to_goal_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._previous_goal_distance = torch.zeros(self.num_envs, device=self.device)
        self._current_goal_distance = torch.zeros(self.num_envs, device=self.device)
        self._previous_actions = torch.full_like(self._actions, 0.5)
        self._previous_position_w = torch.zeros((self.num_envs, 3), device=self.device)

        static_shape = tuple(cfg.observation_space["static_obstacles"])
        self._static_raw = torch.full(
            (self.num_envs, *static_shape),
            cfg.raycasting.max_distance + cfg.raycasting.no_hit_offset,
            device=self.device,
        )
        self._static_normalized = torch.ones((self.num_envs, *static_shape), device=self.device)
        self._static_hit_mask = torch.zeros_like(self._static_normalized, dtype=torch.bool)
        self._motion_mask = torch.zeros_like(self._static_normalized, dtype=torch.bool)
        self._internal_state = torch.zeros((self.num_envs, cfg.internal_state_dim), device=self.device)
        self._dynamic_observation = self._empty_dynamic_observation()
        self._perception_cache_step = -1
        self._perception_reset_version = 0
        self._perception_cache_reset_version = -1

        self._collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._out_of_bounds = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._time_out = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._episode_return = torch.zeros(self.num_envs, device=self.device)
        self._episode_success = torch.zeros(self.num_envs, device=self.device)
        self._episode_collision = torch.zeros(self.num_envs, device=self.device)
        self._episode_timeout = torch.zeros(self.num_envs, device=self.device)
        self._episode_out_of_bounds = torch.zeros(self.num_envs, device=self.device)
        self._episode_path_length = torch.zeros(self.num_envs, device=self.device)
        self._episode_max_altitude = torch.full(
            (self.num_envs,), cfg.takeoff_start_height, device=self.device
        )
        self._episode_min_static_distance = torch.full(
            (self.num_envs,), cfg.raycasting.max_distance, device=self.device
        )
        self._episode_valid_dynamic_tracks = torch.zeros(self.num_envs, device=self.device)
        self._reward_component_names = (
            "progress",
            "velocity",
            "static_safety",
            "dynamic_safety",
            "action",
            "smoothness",
            "altitude",
            "time",
            "goal",
            "collision",
            "out_of_bounds",
        )
        self._episode_reward_sums = {
            name: torch.zeros(self.num_envs, device=self.device) for name in self._reward_component_names
        }
        self._last_reward_components = {
            name: torch.zeros(self.num_envs, device=self.device) for name in self._reward_component_names
        }

        # Scene generation needs simulator truth internally, but these tensors
        # are never passed to _get_observations or reward shaping.
        static_sizes = [(spec.width, spec.length, spec.height) for spec in self._static_specs]
        static_columns = [spec.kind == "cylinder" for spec in self._static_specs]
        self._static_sizes_truth = torch.tensor(static_sizes, dtype=torch.float32, device=self.device)
        self._static_columns_truth = torch.tensor(static_columns, dtype=torch.bool, device=self.device)
        self._static_positions_local = torch.zeros(
            (self.num_envs, cfg.static_pool_size, 3), device=self.device
        )
        self._static_active_mask = torch.zeros(
            (self.num_envs, cfg.static_pool_size), dtype=torch.bool, device=self.device
        )
        dynamic_sizes = [spec.size for spec in self._dynamic_specs]
        dynamic_columns = [spec.is_column for spec in self._dynamic_specs]
        self._dynamic_sizes_truth = torch.tensor(dynamic_sizes, dtype=torch.float32, device=self.device)
        self._dynamic_columns_truth = torch.tensor(dynamic_columns, dtype=torch.bool, device=self.device)
        self._dynamic_positions_local = torch.zeros(
            (self.num_envs, cfg.dynamic_pool_size, 3), device=self.device
        )
        self._dynamic_goals_local = torch.zeros_like(self._dynamic_positions_local)
        self._dynamic_velocities_truth = torch.zeros_like(self._dynamic_positions_local)
        self._dynamic_speeds = torch.zeros((self.num_envs, cfg.dynamic_pool_size), device=self.device)
        self._dynamic_active_mask = torch.zeros(
            (self.num_envs, cfg.dynamic_pool_size), dtype=torch.bool, device=self.device
        )
        all_env_ids = torch.arange(self.num_envs, device=self.device)
        init_phase_start = time.perf_counter()
        self._randomize_static_obstacles(all_env_ids)
        print(f"[V5 INIT] static randomization: {time.perf_counter() - init_phase_start:.2f}s", flush=True)
        init_phase_start = time.perf_counter()
        self._randomize_dynamic_obstacles(all_env_ids)
        print(f"[V5 INIT] dynamic randomization: {time.perf_counter() - init_phase_start:.2f}s", flush=True)
        init_phase_start = time.perf_counter()
        self._write_static_obstacle_state()
        self._write_dynamic_obstacle_state()
        print(f"[V5 INIT] Isaac state synchronization: {time.perf_counter() - init_phase_start:.2f}s", flush=True)

        if torch.device(self.device).type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

    def _setup_scene(self) -> None:
        self._drone = Articulation(self.cfg.drone)
        self.scene.articulations["drone"] = self._drone
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["drone_contact"] = self._contact_sensor

        self._static_specs = sample_obstacle_specs(
            self.cfg.static_pool_size,
            self.cfg.x_range,
            self.cfg.y_range,
            self.cfg.static_obstacle_seed,
            self.cfg.static_minimum_separation,
            height_range=self.cfg.static_height_range,
        )
        static_cfgs: dict[str, RigidObjectCfg] = {}
        for index, obstacle in enumerate(self._static_specs):
            rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True)
            if obstacle.kind == "cuboid":
                spawn_cfg = sim_utils.CuboidCfg(
                    size=(obstacle.width, obstacle.length, obstacle.height),
                    rigid_props=rigid_props,
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.82, 0.30, 0.08), roughness=0.85
                    ),
                )
            else:
                spawn_cfg = sim_utils.CylinderCfg(
                    radius=obstacle.radius,
                    height=obstacle.height,
                    axis="Z",
                    rigid_props=rigid_props,
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.10, 0.50, 0.82), roughness=0.85
                    ),
                )
            initial_position = obstacle.center if index < self.cfg.num_static_obstacles else (
                obstacle.center[0],
                obstacle.center[1],
                -100.0,
            )
            static_cfgs[f"static_{index:03d}"] = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/StaticObstacle_{index:03d}",
                spawn=spawn_cfg,
                init_state=RigidObjectCfg.InitialStateCfg(pos=initial_position),
            )
        self._static_collection = RigidObjectCollection(
            RigidObjectCollectionCfg(rigid_objects=static_cfgs)
        )
        self.scene.rigid_object_collections["static_obstacles"] = self._static_collection

        self._dynamic_specs = sample_dynamic_obstacles(
            self.cfg.dynamic_pool_size,
            self.cfg.dynamic_obstacle_seed,
            map_half_extent=0.8 * max(abs(self.cfg.x_range[0]), abs(self.cfg.x_range[1])),
            flight_height=self.cfg.z_range[1],
        )
        dynamic_cfgs: dict[str, RigidObjectCfg] = {}
        for index, obstacle in enumerate(self._dynamic_specs):
            rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True)
            collision_props = sim_utils.CollisionPropertiesCfg(collision_enabled=True)
            material = sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.05, 0.90, 0.15), metallic=0.10, roughness=0.55
            )
            if obstacle.is_column:
                spawn_cfg = sim_utils.CylinderCfg(
                    radius=0.5 * obstacle.size[0],
                    height=obstacle.size[2],
                    axis="Z",
                    rigid_props=rigid_props,
                    collision_props=collision_props,
                    visual_material=material,
                )
            else:
                spawn_cfg = sim_utils.CuboidCfg(
                    size=obstacle.size,
                    rigid_props=rigid_props,
                    collision_props=collision_props,
                    visual_material=material,
                )
            dynamic_cfgs[f"dynamic_{index:03d}"] = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/DynamicObstacle_{index:03d}",
                spawn=spawn_cfg,
                init_state=RigidObjectCfg.InitialStateCfg(pos=obstacle.center),
            )
        self._dynamic_collection = RigidObjectCollection(
            RigidObjectCollectionCfg(rigid_objects=dynamic_cfgs)
        )
        self.scene.rigid_object_collections["dynamic_obstacles"] = self._dynamic_collection

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        self._depth_sensor = MultiMeshRayCaster(self.cfg.depth_sensor)
        self.scene.sensors["depth_sensor"] = self._depth_sensor
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _empty_dynamic_observation(self) -> DynamicObservation:
        count = self.cfg.motion.max_tracks
        return DynamicObservation(
            state=torch.zeros((self.num_envs, count, self.cfg.dynamic_state_dim), device=self.device),
            indices=torch.zeros((self.num_envs, count), dtype=torch.int64, device=self.device),
            valid=torch.zeros((self.num_envs, count), dtype=torch.bool, device=self.device),
            positions_w=torch.zeros((self.num_envs, count, 3), device=self.device),
            velocities_w=torch.zeros((self.num_envs, count, 3), device=self.device),
            radii=torch.zeros((self.num_envs, count), device=self.device),
            surface_distances=torch.full(
                (self.num_envs, count), self.cfg.dynamic_sensing_range, device=self.device
            ),
            collision=torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
        )

    @torch.no_grad()
    def _sample_active_mask(self, count: int, pool_size: int, active_count: int) -> torch.Tensor:
        """Return independent uniformly selected template masks, shape ``[count,pool_size]``."""
        scores = torch.rand((count, pool_size), device=self.device)
        indices = torch.topk(scores, k=active_count, dim=1, sorted=False).indices
        mask = torch.zeros((count, pool_size), dtype=torch.bool, device=self.device)
        mask.scatter_(1, indices, True)
        return mask

    @torch.no_grad()
    def _minimum_static_clearance(self, points_local: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        """Exact point-to-active-static-collider distance for ``[E,...,3]`` points."""
        ids = torch.as_tensor(env_ids, dtype=torch.int64, device=self.device)
        if points_local.shape[0] != ids.numel() or points_local.shape[-1] != 3:
            raise ValueError("points_local must have shape [len(env_ids),...,3]")
        extra_dims = points_local.ndim - 2
        position_shape = (ids.numel(),) + (1,) * extra_dims + (self.cfg.static_pool_size, 3)
        scalar_shape = (1,) * (1 + extra_dims) + (self.cfg.static_pool_size,)
        centers = self._static_positions_local[ids].view(position_shape)
        sizes = self._static_sizes_truth.view((1,) * (1 + extra_dims) + self._static_sizes_truth.shape)
        delta = points_local.unsqueeze(-2) - centers
        box_outside = (delta.abs() - 0.5 * sizes).clamp_min(0.0)
        box_distance = torch.linalg.vector_norm(box_outside, dim=-1)
        radius = (0.5 * self._static_sizes_truth[:, 0]).view(scalar_shape)
        half_height = (0.5 * self._static_sizes_truth[:, 2]).view(scalar_shape)
        radial_outside = (torch.linalg.vector_norm(delta[..., :2], dim=-1) - radius).clamp_min(0.0)
        vertical_outside = (delta[..., 2].abs() - half_height).clamp_min(0.0)
        cylinder_distance = torch.sqrt(radial_outside.square() + vertical_outside.square())
        is_column = self._static_columns_truth.view(scalar_shape)
        distance = torch.where(is_column, cylinder_distance, box_distance)
        active_shape = (ids.numel(),) + (1,) * extra_dims + (self.cfg.static_pool_size,)
        active = self._static_active_mask[ids].view(active_shape)
        return distance.masked_fill(~active, torch.inf).amin(dim=-1)

    @torch.no_grad()
    def _randomize_static_obstacles(self, env_ids: torch.Tensor) -> None:
        """Select and place static shape templates independently in every reset environment."""
        ids = torch.as_tensor(env_ids, dtype=torch.int64, device=self.device)
        count = ids.numel()
        if count == 0:
            return
        pool = self.cfg.static_pool_size
        active = self._sample_active_mask(count, pool, self.cfg.num_static_obstacles)
        positions = torch.zeros((count, pool, 3), device=self.device)
        # Dense rejection sampling scales poorly for large obstacle pools.  Randomly map
        # immutable shape templates to a jittered 2-D lattice instead.  The
        # lattice spacing is checked against the largest axis-aligned footprint,
        # so every episode remains collision-free while the shape-to-cell
        # assignment, active subset, and positions all change on reset.
        columns = math.ceil(math.sqrt(pool))
        rows = math.ceil(pool / columns)
        half_xy = 0.5 * self._static_sizes_truth[:, :2]
        column_radius = 0.5 * self._static_sizes_truth[:, :1]
        half_xy = torch.where(self._static_columns_truth[:, None], column_radius.expand(-1, 2), half_xy)
        max_half_xy = half_xy.amax(dim=0)
        border_margin = 0.05
        x_low = self.cfg.x_range[0] + float(max_half_xy[0].item()) + border_margin
        x_high = self.cfg.x_range[1] - float(max_half_xy[0].item()) - border_margin
        y_low = self.cfg.y_range[0] + float(max_half_xy[1].item()) + border_margin
        y_high = self.cfg.y_range[1] - float(max_half_xy[1].item()) - border_margin
        x_grid = torch.linspace(x_low, x_high, columns, device=self.device)
        y_grid = torch.linspace(y_low, y_high, rows, device=self.device)
        grid_y, grid_x = torch.meshgrid(y_grid, x_grid, indexing="ij")
        grid_xy = torch.stack((grid_x.flatten(), grid_y.flatten()), dim=-1)
        spacing_x = float((x_grid[1] - x_grid[0]).item()) if columns > 1 else math.inf
        spacing_y = float((y_grid[1] - y_grid[0]).item()) if rows > 1 else math.inf
        required_x = 2.0 * float(max_half_xy[0].item()) + self.cfg.static_minimum_separation
        required_y = 2.0 * float(max_half_xy[1].item()) + self.cfg.static_minimum_separation
        spacing_slack = min(spacing_x - required_x, spacing_y - required_y)
        if spacing_slack < 0.0:
            raise ValueError("Static obstacle pool is too dense for the configured map and separation")
        jitter = min(0.15, 0.25 * spacing_slack)
        cell_order = torch.rand((count, grid_xy.shape[0]), device=self.device).argsort(dim=1)[:, :pool]
        positions[..., :2] = grid_xy[cell_order]
        if jitter > 0.0:
            positions[..., :2] += (2.0 * torch.rand_like(positions[..., :2]) - 1.0) * jitter
        positions[..., 2] = 0.5 * self._static_sizes_truth[:, 2]
        self._static_active_mask[ids] = active
        self._static_positions_local[ids] = positions

    @torch.no_grad()
    def _randomize_dynamic_obstacles(self, env_ids: torch.Tensor) -> None:
        """Randomize active shapes, starts, goals, speeds and sizes through a template pool."""
        ids = torch.as_tensor(env_ids, dtype=torch.int64, device=self.device)
        count = ids.numel()
        if count == 0:
            return
        pool = self.cfg.dynamic_pool_size
        active = self._sample_active_mask(count, pool, self.cfg.num_dynamic_obstacles)
        positions = torch.zeros((count, pool, 3), device=self.device)
        positions[..., 2] = -100.0
        radii = 0.5 * self._dynamic_sizes_truth[:, :2].amax(dim=-1)
        start_local = self._start_pos_w[ids] - self.scene.env_origins[ids]
        goal_local = self._goal_pos_w[ids] - self.scene.env_origins[ids]
        candidates_per_round = 16
        for obstacle_index in range(pool):
            pending = active[:, obstacle_index].clone()
            size = self._dynamic_sizes_truth[obstacle_index]
            radius = radii[obstacle_index]
            is_column = self._dynamic_columns_truth[obstacle_index]
            for _ in range(8):
                if not torch.any(pending):
                    break
                candidates = torch.rand((count, candidates_per_round, 3), device=self.device)
                candidates[..., 0] = 0.8 * self.cfg.x_range[0] + radius + candidates[..., 0] * (
                    0.8 * (self.cfg.x_range[1] - self.cfg.x_range[0]) - 2.0 * radius
                )
                candidates[..., 1] = 0.8 * self.cfg.y_range[0] + radius + candidates[..., 1] * (
                    0.8 * (self.cfg.y_range[1] - self.cfg.y_range[0]) - 2.0 * radius
                )
                if bool(is_column):
                    candidates[..., 2] = 0.5 * size[2]
                else:
                    half_height = 0.5 * size[2]
                    candidates[..., 2] = half_height + candidates[..., 2] * (
                        self.cfg.z_range[1] - 2.0 * half_height
                    )
                endpoint_distance = torch.minimum(
                    torch.linalg.vector_norm(candidates - start_local[:, None], dim=-1),
                    torch.linalg.vector_norm(candidates - goal_local[:, None], dim=-1),
                )
                valid = endpoint_distance > (self.cfg.endpoint_obstacle_clearance + radius)
                valid &= self._minimum_static_clearance(candidates, ids) > (
                    radius + self.cfg.dynamic_static_clearance
                )
                if obstacle_index:
                    separation = torch.linalg.vector_norm(
                        candidates[..., None, :] - positions[:, None, :obstacle_index, :], dim=-1
                    )
                    required = radius + radii[:obstacle_index] + self.cfg.dynamic_minimum_separation
                    conflicts = (
                        (separation <= required.view(1, 1, -1))
                        & active[:, None, :obstacle_index]
                    ).any(dim=-1)
                    valid &= ~conflicts
                valid &= pending.unsqueeze(1)
                has_valid = valid.any(dim=1)
                first_valid = valid.to(torch.int8).argmax(dim=1)
                selected = candidates[torch.arange(count, device=self.device), first_valid]
                accepted = pending & has_valid
                positions[accepted, obstacle_index] = selected[accepted]
                pending &= ~accepted
            if torch.any(pending):
                raise RuntimeError("Unable to sample separated dynamic obstacles on GPU")

        local_range = torch.tensor(self.cfg.dynamic_local_range, device=self.device)
        goals = positions.clone()
        for obstacle_index in range(pool):
            pending = active[:, obstacle_index].clone()
            size = self._dynamic_sizes_truth[obstacle_index]
            radius = radii[obstacle_index]
            is_column = self._dynamic_columns_truth[obstacle_index]
            for _ in range(8):
                if not torch.any(pending):
                    break
                candidates = positions[:, obstacle_index, None, :] + (
                    2.0 * torch.rand((count, candidates_per_round, 3), device=self.device) - 1.0
                ) * local_range
                candidates[..., 0].clamp_(0.8 * self.cfg.x_range[0] + radius, 0.8 * self.cfg.x_range[1] - radius)
                candidates[..., 1].clamp_(0.8 * self.cfg.y_range[0] + radius, 0.8 * self.cfg.y_range[1] - radius)
                if bool(is_column):
                    candidates[..., 2] = 0.5 * size[2]
                else:
                    half_height = 0.5 * size[2]
                    candidates[..., 2].clamp_(half_height, self.cfg.z_range[1] - half_height)
                endpoint_distance = torch.minimum(
                    torch.linalg.vector_norm(candidates - start_local[:, None], dim=-1),
                    torch.linalg.vector_norm(candidates - goal_local[:, None], dim=-1),
                )
                valid = endpoint_distance > (self.cfg.endpoint_obstacle_clearance + radius)
                valid &= self._minimum_static_clearance(candidates, ids) > (
                    radius + self.cfg.dynamic_static_clearance
                )
                valid &= torch.linalg.vector_norm(
                    candidates - positions[:, obstacle_index, None], dim=-1
                ) > self.cfg.dynamic_goal_threshold
                valid &= pending.unsqueeze(1)
                has_valid = valid.any(dim=1)
                first_valid = valid.to(torch.int8).argmax(dim=1)
                selected = candidates[torch.arange(count, device=self.device), first_valid]
                accepted = pending & has_valid
                goals[accepted, obstacle_index] = selected[accepted]
                pending &= ~accepted
            if torch.any(pending):
                raise RuntimeError("Unable to sample collision-free dynamic goals on GPU")
        speed_low, speed_high = self.cfg.dynamic_speed_range
        speeds = speed_low + (speed_high - speed_low) * torch.rand((count, pool), device=self.device)
        self._dynamic_active_mask[ids] = active
        self._dynamic_positions_local[ids] = positions
        self._dynamic_goals_local[ids] = goals
        self._dynamic_speeds[ids] = speeds
        self._dynamic_velocities_truth[ids] = 0.0

    @torch.no_grad()
    def _move_dynamic_obstacles(self) -> None:
        delta = self._dynamic_goals_local - self._dynamic_positions_local
        distance = torch.linalg.vector_norm(delta, dim=-1)
        reached = distance <= self.cfg.dynamic_goal_threshold
        reached &= self._dynamic_active_mask
        if torch.any(reached):
            local_range = torch.tensor(self.cfg.dynamic_local_range, device=self.device)
            radii = 0.5 * self._dynamic_sizes_truth[:, :2].amax(dim=-1)
            pending = reached.clone()
            ids = torch.arange(self.num_envs, device=self.device)
            start_local = self._start_pos_w - self.scene.env_origins
            goal_local = self._goal_pos_w - self.scene.env_origins
            candidates_per_round = 8
            for _ in range(4):
                if not torch.any(pending):
                    break
                candidates = self._dynamic_positions_local[:, :, None, :] + (
                    2.0
                    * torch.rand(
                        (self.num_envs, self.cfg.dynamic_pool_size, candidates_per_round, 3),
                        device=self.device,
                    )
                    - 1.0
                ) * local_range
                candidates[..., 0].clamp_(
                    0.8 * self.cfg.x_range[0] + radii.view(1, -1, 1),
                    0.8 * self.cfg.x_range[1] - radii.view(1, -1, 1),
                )
                candidates[..., 1].clamp_(
                    0.8 * self.cfg.y_range[0] + radii.view(1, -1, 1),
                    0.8 * self.cfg.y_range[1] - radii.view(1, -1, 1),
                )
                half_height = 0.5 * self._dynamic_sizes_truth[:, 2]
                candidates[..., 2] = candidates[..., 2].clamp(
                    min=half_height.view(1, -1, 1),
                    max=(self.cfg.z_range[1] - half_height).view(1, -1, 1),
                )
                candidates[..., 2] = torch.where(
                    self._dynamic_columns_truth.view(1, -1, 1),
                    half_height.view(1, -1, 1),
                    candidates[..., 2],
                )
                endpoint_distance = torch.minimum(
                    torch.linalg.vector_norm(candidates - start_local[:, None, None], dim=-1),
                    torch.linalg.vector_norm(candidates - goal_local[:, None, None], dim=-1),
                )
                valid = endpoint_distance > (
                    self.cfg.endpoint_obstacle_clearance + radii.view(1, -1, 1)
                )
                valid &= self._minimum_static_clearance(candidates, ids) > (
                    radii.view(1, -1, 1) + self.cfg.dynamic_static_clearance
                )
                valid &= pending.unsqueeze(-1)
                has_valid = valid.any(dim=-1)
                first_valid = valid.to(torch.int8).argmax(dim=-1)
                env_rows = torch.arange(self.num_envs, device=self.device).view(-1, 1)
                obstacle_rows = torch.arange(self.cfg.dynamic_pool_size, device=self.device).view(1, -1)
                selected = candidates[env_rows, obstacle_rows, first_valid]
                accepted = pending & has_valid
                self._dynamic_goals_local[accepted] = selected[accepted]
                pending &= ~accepted
            delta = self._dynamic_goals_local - self._dynamic_positions_local
        direction = delta / torch.linalg.vector_norm(delta, dim=-1, keepdim=True).clamp_min(1.0e-6)
        self._dynamic_velocities_truth[:] = direction * self._dynamic_speeds.unsqueeze(-1)
        self._dynamic_velocities_truth[:, self._dynamic_columns_truth, 2] = 0.0
        self._dynamic_velocities_truth.masked_fill_(~self._dynamic_active_mask.unsqueeze(-1), 0.0)
        self._dynamic_positions_local += self._dynamic_velocities_truth * self.step_dt
        self._write_dynamic_obstacle_state()

    @torch.no_grad()
    def _write_static_obstacle_state(self, env_ids: torch.Tensor | None = None) -> None:
        """Synchronize reset-randomized static templates with Isaac and Warp."""
        ids = (
            torch.arange(self.num_envs, device=self.device)
            if env_ids is None
            else torch.as_tensor(env_ids, dtype=torch.int64, device=self.device)
        )
        pose = torch.zeros((ids.numel(), self.cfg.static_pool_size, 7), device=self.device)
        pose[..., :3] = self._static_positions_local[ids] + self.scene.env_origins[ids].unsqueeze(1)
        pose[..., 3] = 1.0
        pose[..., 2] = torch.where(
            self._static_active_mask[ids], pose[..., 2], torch.full_like(pose[..., 2], -100.0)
        )
        self._static_collection.write_object_pose_to_sim(pose, env_ids=ids)

    @torch.no_grad()
    def _write_dynamic_obstacle_state(self, env_ids: torch.Tensor | None = None) -> None:
        ids = (
            torch.arange(self.num_envs, device=self.device)
            if env_ids is None
            else torch.as_tensor(env_ids, dtype=torch.int64, device=self.device)
        )
        state = torch.zeros(
            (ids.numel(), self.cfg.dynamic_pool_size, 13), device=self.device
        )
        state[..., :3] = self._dynamic_positions_local[ids] + self.scene.env_origins[ids].unsqueeze(1)
        state[..., 3] = 1.0
        state[..., 7:10] = self._dynamic_velocities_truth[ids]
        inactive = ~self._dynamic_active_mask[ids]
        state[..., 2] = torch.where(inactive, torch.full_like(state[..., 2], -100.0), state[..., 2])
        state[..., 7:10].masked_fill_(inactive.unsqueeze(-1), 0.0)
        self._dynamic_collection.write_object_state_to_sim(state, env_ids=ids)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if self.cfg.debug_checks:
            assert actions.shape == (self.num_envs, 3)
            assert torch.isfinite(actions).all()
        self._actions.copy_(actions.clamp(0.0, 1.0))
        centered = 2.0 * self._actions - 1.0
        self._command_velocity_g[:, :2] = centered[:, :2] * self.cfg.max_velocity_xy
        self._command_velocity_g[:, 2] = centered[:, 2] * self.cfg.max_velocity_z
        self._command_velocity_w.zero_()
        self._command_velocity_w[:, :3] = goal_frame_to_world(
            self._command_velocity_g, self._start_to_goal_w
        )
        self._move_dynamic_obstacles()

    def _apply_action(self) -> None:
        """Apply the policy command directly, exactly as the V2 plant does."""
        self._drone.write_root_velocity_to_sim(self._command_velocity_w)

    @torch.no_grad()
    def _update_perception(self) -> None:
        if (
            self._perception_cache_step == self.common_step_counter
            and self._perception_cache_reset_version == self._perception_reset_version
        ):
            return
        heading = quaternion_wxyz_to_yaw(self._drone.data.root_quat_w)
        output: GpuPerceptionOutput = self._perception.update(
            self._depth_sensor.data.ray_hits_w,
            self._depth_sensor._ray_directions_w,
            self._depth_sensor.data.pos_w,
            self.scene.env_origins,
            self._drone.data.root_pos_w,
            heading,
            self.step_dt,
        )
        self._static_raw.copy_(output.static_raw)
        self._static_normalized.copy_(output.static_normalized)
        self._static_hit_mask.copy_(output.static_hit_mask)
        self._motion_mask.copy_(output.motion_mask)
        self._internal_state.copy_(
            navrl_internal_state(
                self._drone.data.root_pos_w,
                self._goal_pos_w,
                self._drone.data.root_lin_vel_w,
                self._start_to_goal_w,
                distance_scale_xy=self.cfg.internal_goal_distance_scale,
                distance_scale_z=self.cfg.internal_vertical_distance_scale,
                velocity_scale=self.cfg.internal_velocity_scale,
            )
        )
        perceived_columns = torch.zeros_like(output.dynamic_valid)
        self._dynamic_observation = navrl_dynamic_observation(
            self._drone.data.root_pos_w,
            self._start_to_goal_w,
            output.dynamic_positions_w,
            output.dynamic_velocities_w,
            output.dynamic_sizes,
            perceived_columns,
            None,
            num_closest=self.cfg.motion.max_tracks,
            sensing_range=self.cfg.dynamic_sensing_range,
            width_resolution=self.cfg.dynamic_width_resolution,
            robot_radius=self.cfg.collision_radius,
            active_mask=output.dynamic_valid,
        )
        self._perception_cache_step = self.common_step_counter
        self._perception_cache_reset_version = self._perception_reset_version

    def _get_observations(self) -> dict[str, torch.Tensor]:
        self._update_perception()
        observations = {
            "static_obstacles": self._static_normalized,
            "internal_state": self._internal_state,
            "dynamic_obstacles": self._dynamic_observation.state,
        }
        if self.cfg.debug_checks:
            assert observations["static_obstacles"].shape == (
                self.num_envs,
                *self.cfg.observation_space["static_obstacles"],
            )
            assert observations["internal_state"].shape == (
                self.num_envs,
                self.cfg.internal_state_dim,
            )
            assert observations["dynamic_obstacles"].shape == (
                self.num_envs,
                self.cfg.motion.max_tracks,
                self.cfg.dynamic_state_dim,
            )
            assert all(torch.isfinite(value).all() for value in observations.values())
        return observations

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.extras.pop("log", None)
        self._update_perception()
        contact = contact_forces_to_collision(
            self._contact_sensor.data.net_forces_w_history,
            self.cfg.collision_force_threshold,
        )
        contact &= self.episode_length_buf > self.cfg.contact_reset_grace_steps
        self._collision.copy_(contact)
        position_local = self._drone.data.root_pos_w - self.scene.env_origins
        self._episode_max_altitude.copy_(
            torch.maximum(self._episode_max_altitude, position_local[:, 2])
        )
        self._out_of_bounds.copy_(
            out_of_bounds_mask(
                position_local,
                x_limit=max(abs(self.cfg.x_range[0]), abs(self.cfg.x_range[1])),
                y_limit=max(abs(self.cfg.y_range[0]), abs(self.cfg.y_range[1])),
                min_z=self.cfg.termination_min_z,
                max_z=self.cfg.termination_max_z,
            )
        )
        self._current_goal_distance.copy_(
            torch.linalg.vector_norm(self._goal_pos_w - self._drone.data.root_pos_w, dim=-1)
        )
        self._success.copy_(self._current_goal_distance <= self.cfg.goal_threshold)
        self._out_of_bounds &= ~self._success
        terminated = self._success | self._collision | self._out_of_bounds
        self._time_out.copy_((self.episode_length_buf >= self.max_episode_length) & ~terminated)

        step_distance = torch.linalg.vector_norm(
            self._drone.data.root_pos_w - self._previous_position_w, dim=-1
        )
        self._episode_path_length += step_distance
        self._previous_position_w.copy_(self._drone.data.root_pos_w)
        minimum_static = self._static_raw.amin(dim=(1, 2)).clamp_max(self.cfg.raycasting.max_distance)
        self._episode_min_static_distance.copy_(
            torch.minimum(self._episode_min_static_distance, minimum_static)
        )
        self._episode_valid_dynamic_tracks += self._dynamic_observation.valid.float().sum(dim=1)
        self.extras.update(
            {
                "success": self._success.clone(),
                "collision": self._collision.clone(),
                "out_of_bounds": self._out_of_bounds.clone(),
                "timeout": self._time_out.clone(),
                "final_goal_distance": self._current_goal_distance.clone(),
                "min_static_distance": minimum_static.clone(),
                "valid_dynamic_tracks": self._dynamic_observation.valid.sum(dim=1).clone(),
                "altitude_m": position_local[:, 2].clone(),
                "episode_max_altitude_m": self._episode_max_altitude.clone(),
                "above_soft_ceiling": (position_local[:, 2] > self.cfg.soft_flight_ceiling).clone(),
                "simulator_truth_policy_input": torch.zeros(
                    self.num_envs, dtype=torch.bool, device=self.device
                ),
            }
        )
        return terminated, self._time_out

    def _get_rewards(self) -> torch.Tensor:
        self._update_perception()
        relative_goal = self._goal_pos_w - self._drone.data.root_pos_w
        goal_direction = relative_goal / self._current_goal_distance.clamp_min(1.0e-6).unsqueeze(-1)
        progress = self._previous_goal_distance - self._current_goal_distance
        goal_velocity = torch.sum(self._drone.data.root_lin_vel_w * goal_direction, dim=-1)
        static_clearance = self._static_raw.clamp_max(self.cfg.raycasting.max_distance)
        static_safety = torch.log(
            static_clearance.clamp_min(self.cfg.collision_radius)
            / self.cfg.raycasting.max_distance
        ).mean(dim=(1, 2))
        dynamic_clearance = self._dynamic_observation.surface_distances.clamp(
            min=self.cfg.collision_radius,
            max=self.cfg.dynamic_sensing_range,
        )
        dynamic_safety = torch.log(dynamic_clearance / self.cfg.dynamic_sensing_range).mean(dim=1)
        centered_action = 2.0 * self._actions - 1.0
        previous_centered_action = 2.0 * self._previous_actions - 1.0
        position_local = self._drone.data.root_pos_w - self.scene.env_origins
        start_height = self._start_pos_w[:, 2] - self.scene.env_origins[:, 2]
        goal_height = self._goal_pos_w[:, 2] - self.scene.env_origins[:, 2]
        # NavRL-style height corridor: flying above the endpoint corridor is
        # penalized before reaching the global ceiling.  The global cap makes
        # the constraint explicit even for future endpoint distributions.
        corridor_ceiling = torch.minimum(
            torch.maximum(start_height, goal_height) + self.cfg.altitude_corridor_tolerance,
            torch.full_like(goal_height, self.cfg.soft_flight_ceiling),
        )
        altitude_excess = (position_local[:, 2] - corridor_ceiling).clamp_min(0.0)
        components = {
            "progress": self.cfg.reward_progress_weight * progress,
            "velocity": self.cfg.reward_velocity_weight * goal_velocity,
            "static_safety": self.cfg.reward_static_safety_weight * static_safety,
            "dynamic_safety": self.cfg.reward_dynamic_safety_weight * dynamic_safety,
            "action": -self.cfg.reward_action_weight * centered_action.square().sum(dim=-1),
            "smoothness": -self.cfg.reward_smoothness_weight
            * (centered_action - previous_centered_action).square().sum(dim=-1),
            "altitude": -self.cfg.reward_altitude_weight * altitude_excess.square(),
            "time": torch.full_like(progress, -self.cfg.reward_time_penalty),
            "goal": self.cfg.reward_goal_bonus * self._success.float(),
            "collision": -self.cfg.reward_collision_penalty * self._collision.float(),
            "out_of_bounds": -self.cfg.reward_out_of_bounds_penalty * self._out_of_bounds.float(),
        }
        reward = torch.stack(tuple(components.values())).sum(dim=0)
        for name, value in components.items():
            self._last_reward_components[name].copy_(value)
            self._episode_reward_sums[name] += value
        self._episode_return += reward
        self._episode_success.copy_(torch.maximum(self._episode_success, self._success.float()))
        self._episode_collision.copy_(torch.maximum(self._episode_collision, self._collision.float()))
        self._episode_timeout.copy_(torch.maximum(self._episode_timeout, self._time_out.float()))
        self._episode_out_of_bounds.copy_(
            torch.maximum(self._episode_out_of_bounds, self._out_of_bounds.float())
        )
        self._previous_goal_distance.copy_(self._current_goal_distance)
        self._previous_actions.copy_(self._actions)
        self.extras["reward_components"] = {
            name: value.clone() for name, value in components.items()
        }
        if self.cfg.debug_checks:
            assert torch.isfinite(reward).all()
        return reward

    @torch.no_grad()
    def _sample_start_goal(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample per-environment endpoints clear of that environment's randomized geometry."""
        ids = torch.as_tensor(env_ids, dtype=torch.int64, device=self.device)
        count = ids.numel()
        lower_xy = torch.tensor(
            (self.cfg.x_range[0] + 1.0, self.cfg.y_range[0] + 1.0),
            dtype=torch.float32,
            device=self.device,
        )
        upper_xy = torch.tensor(
            (self.cfg.x_range[1] - 1.0, self.cfg.y_range[1] - 1.0),
            dtype=torch.float32,
            device=self.device,
        )
        starts = torch.zeros((count, 3), device=self.device)
        goals = torch.zeros_like(starts)
        unresolved = torch.ones(count, dtype=torch.bool, device=self.device)
        candidates_per_round = 16
        extent_xy = upper_xy - lower_xy
        for _ in range(32):
            if not torch.any(unresolved):
                break
            candidate_starts = torch.empty(
                (count, candidates_per_round, 3), device=self.device
            )
            candidate_starts[..., :2] = lower_xy + torch.rand(
                (count, candidates_per_round, 2), device=self.device
            ) * extent_xy
            candidate_starts[..., 2] = self.cfg.takeoff_start_height
            candidate_goals = torch.empty_like(candidate_starts)
            candidate_goals[..., :2] = lower_xy + torch.rand(
                (count, candidates_per_round, 2), device=self.device
            ) * extent_xy
            candidate_goals[..., 2] = self.cfg.goal_z_range[0] + torch.rand(
                (count, candidates_per_round), device=self.device
            ) * (self.cfg.goal_z_range[1] - self.cfg.goal_z_range[0])
            pair_distance = torch.linalg.vector_norm(candidate_goals - candidate_starts, dim=-1)
            valid = (
                (pair_distance >= self.cfg.min_start_goal_distance)
                & (pair_distance <= self.cfg.max_start_goal_distance)
                & (
                    self._minimum_static_clearance(candidate_starts, ids)
                    > self.cfg.endpoint_obstacle_clearance
                )
                & (
                    self._minimum_static_clearance(candidate_goals, ids)
                    > self.cfg.endpoint_obstacle_clearance
                )
                & unresolved.unsqueeze(1)
            )
            has_valid = valid.any(dim=1)
            first_valid = valid.to(torch.int8).argmax(dim=1)
            rows = torch.arange(count, device=self.device)
            selected_starts = candidate_starts[rows, first_valid]
            selected_goals = candidate_goals[rows, first_valid]
            accepted = unresolved & has_valid
            starts[accepted] = selected_starts[accepted]
            goals[accepted] = selected_goals[accepted]
            unresolved &= ~accepted
        if torch.any(unresolved):
            raise RuntimeError("Unable to sample endpoints clear of randomized static geometry")
        return starts, goals

    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        self._log_episode_statistics(env_ids)
        ids = torch.as_tensor(env_ids, dtype=torch.int64, device=self.device)
        self._drone.reset(ids)
        self._contact_sensor.reset(ids)
        super()._reset_idx(env_ids)

        # Geometry is written before sensor/tracker reset.  Consequently the
        # first post-reset depth frame, voxel timestamp and motion history all
        # refer to the same episode layout.
        self._randomize_static_obstacles(ids)
        self._write_static_obstacle_state(ids)
        start_local, goal_local = self._sample_start_goal(ids)
        self._start_pos_w[ids] = start_local + self.scene.env_origins[ids]
        self._goal_pos_w[ids] = goal_local + self.scene.env_origins[ids]
        self._start_to_goal_w[ids] = self._goal_pos_w[ids] - self._start_pos_w[ids]
        self._randomize_dynamic_obstacles(ids)
        self._write_dynamic_obstacle_state(ids)

        root_state = self._drone.data.default_root_state[ids].clone()
        root_state[:, :3] = self._start_pos_w[ids]
        yaw = torch.atan2(self._start_to_goal_w[ids, 1], self._start_to_goal_w[ids, 0])
        zeros = torch.zeros_like(yaw)
        root_state[:, 3:7] = quat_from_euler_xyz(zeros, zeros, yaw)
        root_state[:, 7:] = 0.0
        self._drone.write_root_pose_to_sim(root_state[:, :7], ids)
        self._drone.write_root_velocity_to_sim(root_state[:, 7:], ids)
        self._drone.write_joint_state_to_sim(
            self._drone.data.default_joint_pos[ids].clone(),
            self._drone.data.default_joint_vel[ids].clone(),
            None,
            ids,
        )
        self._depth_sensor.reset(ids)
        self._perception.reset(ids)

        initial_distance = torch.linalg.vector_norm(self._start_to_goal_w[ids], dim=-1)
        self._previous_goal_distance[ids] = initial_distance
        self._current_goal_distance[ids] = initial_distance
        self._previous_position_w[ids] = self._start_pos_w[ids]
        self._actions[ids] = 0.5
        self._previous_actions[ids] = 0.5
        self._command_velocity_g[ids] = 0.0
        self._command_velocity_w[ids] = 0.0
        self._collision[ids] = False
        self._out_of_bounds[ids] = False
        self._success[ids] = False
        self._time_out[ids] = False
        self._static_raw[ids] = self.cfg.raycasting.max_distance + self.cfg.raycasting.no_hit_offset
        self._static_normalized[ids] = 1.0
        self._static_hit_mask[ids] = False
        self._motion_mask[ids] = False
        self._internal_state[ids] = 0.0
        self._perception_reset_version += 1
        self._episode_return[ids] = 0.0
        self._episode_success[ids] = 0.0
        self._episode_collision[ids] = 0.0
        self._episode_timeout[ids] = 0.0
        self._episode_out_of_bounds[ids] = 0.0
        self._episode_path_length[ids] = 0.0
        self._episode_max_altitude[ids] = self.cfg.takeoff_start_height
        self._episode_min_static_distance[ids] = self.cfg.raycasting.max_distance
        self._episode_valid_dynamic_tracks[ids] = 0.0
        for tensor in self._episode_reward_sums.values():
            tensor[ids] = 0.0
        for tensor in self._last_reward_components.values():
            tensor[ids] = 0.0

    def _log_episode_statistics(self, env_ids: Sequence[int]) -> None:
        ids = torch.as_tensor(env_ids, dtype=torch.int64, device=self.device)
        ids = ids[self.episode_length_buf[ids] > 0]
        if ids.numel() == 0:
            return
        steps = self.episode_length_buf[ids].float().clamp_min(1.0)
        log = {
            "Episode/return": self._episode_return[ids].mean(),
            "Episode/episode_len": steps.mean(),
            "Episode/success": self._episode_success[ids].mean(),
            "Episode/collision": self._episode_collision[ids].mean(),
            "Episode/timeout": self._episode_timeout[ids].mean(),
            "Episode/out_of_bounds": self._episode_out_of_bounds[ids].mean(),
            "Episode/final_goal_distance": self._current_goal_distance[ids].mean(),
            "Episode/path_length": self._episode_path_length[ids].mean(),
            "Episode/max_altitude_m": self._episode_max_altitude[ids].mean(),
            "Perception/min_static_distance": self._episode_min_static_distance[ids].mean(),
            "Perception/mean_valid_dynamic_tracks": (
                self._episode_valid_dynamic_tracks[ids] / steps
            ).mean(),
            "Perception/simulator_truth_policy_input": 0.0,
            "Performance/parallel_envs": float(self.num_envs),
            "Performance/voxel_map_mib": self._perception.map.memory_bytes / (1024.0**2),
        }
        for name in self._reward_component_names:
            log[f"Episode_Reward/{name}"] = self._episode_reward_sums[name][ids].mean()
        self.extras["log"] = log

    def get_perception_state(self) -> dict[str, torch.Tensor | str | int]:
        """Return policy-side tensors and auditable source metadata."""
        self._update_perception()
        return {
            "source": "isaac_warp_depth_sensor_then_gpu_voxel_motion_tracking",
            "simulator_truth_policy_input": "false",
            "static_raw": self._static_raw.clone(),
            "static_normalized": self._static_normalized.clone(),
            "static_hit_mask": self._static_hit_mask.clone(),
            "motion_mask": self._motion_mask.clone(),
            "dynamic_state": self._dynamic_observation.state.clone(),
            "dynamic_valid": self._dynamic_observation.valid.clone(),
            "internal_state": self._internal_state.clone(),
            "voxel_map_memory_bytes": self._perception.map.memory_bytes,
        }

    def get_randomized_scene_state(self) -> dict[str, torch.Tensor | int]:
        """Return simulator-private domain state for reset validation only."""
        return {
            "static_positions_local": self._static_positions_local.clone(),
            "static_sizes": self._static_sizes_truth.clone(),
            "static_active_mask": self._static_active_mask.clone(),
            "dynamic_positions_local": self._dynamic_positions_local.clone(),
            "dynamic_goals_local": self._dynamic_goals_local.clone(),
            "dynamic_sizes": self._dynamic_sizes_truth.clone(),
            "dynamic_speeds": self._dynamic_speeds.clone(),
            "dynamic_active_mask": self._dynamic_active_mask.clone(),
            "static_positions_w": self._static_collection.data.object_pos_w.clone(),
            "dynamic_positions_w": self._dynamic_collection.data.object_pos_w.clone(),
            "perception_reset_version": self._perception_reset_version,
        }

    def get_control_state(self) -> dict[str, torch.Tensor | str]:
        """Expose the one-to-one policy/direct-velocity command mapping."""
        return {
            "backend": "direct_root_velocity",
            "policy_action": self._actions.clone(),
            "command_velocity_goal": self._command_velocity_g.clone(),
            "executed_velocity_world": self._command_velocity_w[:, :3].clone(),
        }


__all__ = ["NavRLGpuEnv", "quaternion_wxyz_to_yaw"]
