"""V6 finite-FOV depth navigation task with the validated V5 task mechanics."""
from __future__ import annotations

import torch
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.utils import configclass

from navrl_uav_v5.assets import DRONE_CFG
from navrl_uav_v5.utils.gpu_camera_motion_tracker import GpuCameraMotionTracker
from navrl_uav_v5.utils.navrl_dynamic import navrl_dynamic_observation, navrl_internal_state
from .navrl_env import NavRLGpuEnv
from .navrl_env_cfg import NavRLGpuEnvCfg


@configclass
class V6FrontDepthCameraEnvCfg(NavRLGpuEnvCfg):
    """Formal V6 navigation configuration using a front axial-depth image."""
    seed = 6
    episode_length_s = 12.0
    camera_height = 96
    camera_width = 160
    camera_near_m = 0.10
    camera_far_m = 5.0
    camera_horizontal_fov_deg = 90.0
    stereo_baseline_m = 0.10
    static_embedding_dim = 128
    internal_state_dim = 8
    dynamic_state_dim = 10
    max_dynamic_tracks = 5
    motion_threshold_m = 0.015
    association_distance_m = 0.75
    velocity_smoothing = 0.70
    max_missed_frames = 10
    motion_nms_kernel = 9

    # V6 starts airborne at the requested 1.5 m while retaining Crazyflie and
    # the V5 high-level direct-root-velocity plant.
    takeoff_start_height = 1.50
    z_range = (0.10, 2.10)
    goal_z_range = (1.55, 1.90)
    soft_flight_ceiling = 2.10
    termination_max_z = 2.40
    internal_vertical_distance_scale = termination_max_z - 0.05

    observation_space = {
        "front_depth": [camera_height, camera_width, 1],
        "internal_state": internal_state_dim,
        "dynamic_obstacles": [max_dynamic_tracks, dynamic_state_dim],
    }
    scene = InteractiveSceneCfg(num_envs=64, env_spacing=22.0, replicate_physics=True)
    drone = DRONE_CFG.copy()
    drone.init_state.pos = (0.0, 0.0, takeoff_start_height)
    front_depth_camera = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Drone/body/front_depth_camera",
        update_period=0.0,
        height=camera_height,
        width=camera_width,
        data_types=["distance_to_image_plane"],
        depth_clipping_behavior="none",
        update_latest_camera_pose=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            focus_distance=5.0,
            horizontal_aperture=24.0,
            clipping_range=(camera_near_m, camera_far_m),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.05, 0.0, 0.05), rot=(1.0, 0.0, 0.0, 0.0), convention="world"
        ),
    )


class V6FrontDepthCameraEnv(NavRLGpuEnv):
    """V5 navigation dynamics/reward/termination with V6 camera observations."""
    cfg: V6FrontDepthCameraEnvCfg

    def _initialize_perception(self) -> None:
        cfg = self.cfg
        self._motion_tracker = GpuCameraMotionTracker(
            self.num_envs,
            (cfg.camera_height, cfg.camera_width),
            cfg.max_dynamic_tracks,
            near_m=cfg.camera_near_m,
            far_m=cfg.camera_far_m,
            motion_threshold_m=cfg.motion_threshold_m,
            association_distance_m=cfg.association_distance_m,
            velocity_smoothing=cfg.velocity_smoothing,
            max_missed_frames=cfg.max_missed_frames,
            nms_kernel=cfg.motion_nms_kernel,
            width_resolution_m=cfg.dynamic_width_resolution,
            device=self.device,
        )
        self._camera_dynamic_observation = None

    def _static_buffer_shape(self) -> tuple[int, ...]:
        return (self.cfg.camera_height, self.cfg.camera_width)

    def _setup_perception_sensor(self) -> None:
        self._camera = TiledCamera(self.cfg.front_depth_camera)
        self.scene.sensors["front_depth_camera"] = self._camera

    def _reset_perception(self, env_ids: torch.Tensor) -> None:
        self._camera.reset(env_ids)
        self._motion_tracker.reset(env_ids)

    def _perception_memory_bytes(self) -> int:
        tensors = (
            self._motion_tracker.previous_depth,
            self._motion_tracker.previous_position_w,
            self._motion_tracker.previous_quaternion_w_ros,
            self._motion_tracker.positions_w,
            self._motion_tracker.velocities_w,
            self._motion_tracker.sizes,
            self._motion_tracker.valid,
            self._motion_tracker.missed_frames,
        )
        return sum(t.numel() * t.element_size() for t in tensors)

    @torch.no_grad()
    def _update_perception(self) -> None:
        if (self._perception_cache_step == self.common_step_counter
                and self._perception_cache_reset_version == self._perception_reset_version):
            return
        depth = self._camera.data.output["distance_to_image_plane"]
        expected = (self.num_envs, self.cfg.camera_height, self.cfg.camera_width, 1)
        if tuple(depth.shape) != expected:
            raise RuntimeError(f"unexpected depth shape {tuple(depth.shape)}, expected {expected}")
        raw_depth = depth[..., 0]
        valid = torch.isfinite(raw_depth)
        valid &= raw_depth >= self.cfg.camera_near_m
        valid &= raw_depth <= self.cfg.camera_far_m
        metric = torch.nan_to_num(
            raw_depth, nan=self.cfg.camera_far_m,
            posinf=self.cfg.camera_far_m, neginf=self.cfg.camera_near_m,
        ).clamp(self.cfg.camera_near_m, self.cfg.camera_far_m)
        self._static_raw.copy_(metric)
        self._static_normalized.copy_(metric / self.cfg.camera_far_m)
        self._static_hit_mask.copy_(valid)

        data = self._camera.data
        tracked = self._motion_tracker.update(
            depth, data.intrinsic_matrices, data.pos_w, data.quat_w_ros,
            self._drone.data.root_pos_w, self._start_to_goal_w, self.step_dt,
        )
        self._camera_dynamic_observation = tracked
        self._motion_mask.copy_(tracked.motion_mask)
        self._internal_state.copy_(navrl_internal_state(
            self._drone.data.root_pos_w, self._goal_pos_w,
            self._drone.data.root_lin_vel_w, self._start_to_goal_w,
            distance_scale_xy=self.cfg.internal_goal_distance_scale,
            distance_scale_z=self.cfg.internal_vertical_distance_scale,
            velocity_scale=self.cfg.internal_velocity_scale,
        ))
        perceived_columns = torch.zeros_like(tracked.valid)
        self._dynamic_observation = navrl_dynamic_observation(
            self._drone.data.root_pos_w, self._start_to_goal_w,
            tracked.positions_w, tracked.velocities_w, tracked.sizes,
            perceived_columns, None,
            num_closest=self.cfg.max_dynamic_tracks,
            sensing_range=self.cfg.dynamic_sensing_range,
            width_resolution=self.cfg.dynamic_width_resolution,
            robot_radius=self.cfg.collision_radius,
            active_mask=tracked.valid,
        )
        self._perception_cache_step = self.common_step_counter
        self._perception_cache_reset_version = self._perception_reset_version

    def _get_observations(self) -> dict[str, torch.Tensor]:
        self._update_perception()
        observations = {
            "front_depth": self._camera.data.output["distance_to_image_plane"].clone(),
            "internal_state": self._internal_state,
            "dynamic_obstacles": self._dynamic_observation.state,
        }
        if self.cfg.debug_checks:
            assert observations["front_depth"].shape == (
                self.num_envs, self.cfg.camera_height, self.cfg.camera_width, 1
            )
            assert observations["internal_state"].shape == (self.num_envs, 8)
            assert observations["dynamic_obstacles"].shape == (self.num_envs, 5, 10)
            assert torch.isfinite(observations["internal_state"]).all()
            assert torch.isfinite(observations["dynamic_obstacles"]).all()
        return observations

    def get_camera_state(self) -> dict[str, torch.Tensor | str | object]:
        self._update_perception()
        data = self._camera.data
        return {
            "source": "rtx_tiled_camera_distance_to_image_plane",
            "depth_m": data.output["distance_to_image_plane"].clone(),
            "intrinsics": data.intrinsic_matrices.clone(),
            "position_w": data.pos_w.clone(),
            "quaternion_w_world": data.quat_w_world.clone(),
            "quaternion_w_ros": data.quat_w_ros.clone(),
            "dynamic_observation": self._camera_dynamic_observation,
        }

    def get_perception_state(self) -> dict[str, torch.Tensor | str | int]:
        self._update_perception()
        return {
            "source": "isaac_rtx_front_depth_then_gpu_temporal_tracking",
            "simulator_truth_policy_input": "false",
            "front_depth": self._camera.data.output["distance_to_image_plane"].clone(),
            "static_raw": self._static_raw.clone(),
            "static_normalized": self._static_normalized.clone(),
            "static_hit_mask": self._static_hit_mask.clone(),
            "motion_mask": self._motion_mask.clone(),
            "dynamic_state": self._dynamic_observation.state.clone(),
            "dynamic_valid": self._dynamic_observation.valid.clone(),
            "internal_state": self._internal_state.clone(),
            "perception_memory_bytes": self._perception_memory_bytes(),
        }


__all__ = ["V6FrontDepthCameraEnv", "V6FrontDepthCameraEnvCfg"]
