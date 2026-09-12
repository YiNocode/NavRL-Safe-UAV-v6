"""V4 front-depth navigation with ego-compensated temporal voxel memory."""

from __future__ import annotations

import torch
from isaaclab.utils import configclass

from navrl_uav_v5.utils.gpu_temporal_front_depth_voxel import GpuTemporalFrontDepthVoxelizer
from .v6_camera_env import V6FrontDepthCameraEnv
from .v6_voxel_reward_v3_env import (
    V6FrontDepthVoxelRewardV3Env,
    V6FrontDepthVoxelRewardV3EnvCfg,
)


@configclass
class V6TemporalVoxelEnvCfg(V6FrontDepthVoxelRewardV3EnvCfg):
    """V3 reward and task configuration with a 1.5-second local map."""

    temporal_voxel_history_s = 1.5
    observation_space = {
        "front_voxel": [4, 16, 32, 32],
        "internal_state": 8,
        "dynamic_obstacles": [5, 10],
    }


class V6TemporalVoxelEnv(V6FrontDepthVoxelRewardV3Env):
    """Fuse current depth with ego-warped finite-FOV voxel history."""

    cfg: V6TemporalVoxelEnvCfg

    def _initialize_perception(self) -> None:
        V6FrontDepthCameraEnv._initialize_perception(self)
        cfg = self.cfg
        self._voxelizer = GpuTemporalFrontDepthVoxelizer(
            self.num_envs,
            (cfg.camera_height, cfg.camera_width),
            cfg.voxel_grid_shape,
            history_duration_s=cfg.temporal_voxel_history_s,
            forward_range_m=cfg.voxel_forward_range_m,
            lateral_range_m=cfg.voxel_lateral_range_m,
            vertical_range_m=cfg.voxel_vertical_range_m,
            near_m=cfg.camera_near_m,
            far_m=cfg.camera_far_m,
            pixel_stride=cfg.voxel_pixel_stride,
            free_samples=cfg.voxel_free_samples,
            device=self.device,
        )
        self._voxel_cache_step = -1
        self._voxel_cache_reset_version = -1

    def _reset_perception(self, env_ids: torch.Tensor) -> None:
        V6FrontDepthCameraEnv._reset_perception(self, env_ids)
        self._voxelizer.reset(env_ids)

    def _perception_memory_bytes(self) -> int:
        return (
            V6FrontDepthCameraEnv._perception_memory_bytes(self)
            + self._voxelizer.memory_bytes
        )

    @torch.no_grad()
    def _update_perception(self) -> None:
        V6FrontDepthCameraEnv._update_perception(self)
        if (
            self._voxel_cache_step == self.common_step_counter
            and self._voxel_cache_reset_version == self._perception_reset_version
        ):
            return
        camera_data = self._camera.data
        self._voxelizer.update(
            camera_data.output["distance_to_image_plane"],
            camera_data.intrinsic_matrices,
            camera_data.pos_w,
            camera_data.quat_w_ros,
            self.step_dt,
            dynamic_mask=self._motion_mask,
        )
        self._voxel_cache_step = self.common_step_counter
        self._voxel_cache_reset_version = self._perception_reset_version

    def _get_observations(self) -> dict[str, torch.Tensor]:
        self._update_perception()
        observations = {
            "front_voxel": self._voxelizer.state,
            "internal_state": self._internal_state,
            "dynamic_obstacles": self._dynamic_observation.state,
        }
        if self.cfg.debug_checks:
            assert observations["front_voxel"].shape == (
                self.num_envs, 4, *self.cfg.voxel_grid_shape
            )
            assert torch.isfinite(observations["front_voxel"]).all()
            assert observations["internal_state"].shape == (self.num_envs, 8)
            assert observations["dynamic_obstacles"].shape == (self.num_envs, 5, 10)
        return observations

    def get_perception_state(self) -> dict[str, torch.Tensor | str | int]:
        state = V6FrontDepthCameraEnv.get_perception_state(self)
        state.update(
            source="isaac_rtx_front_depth_then_gpu_ego_compensated_temporal_voxel",
            front_voxel=self._voxelizer.state.clone(),
            temporal_voxel_history_s=self.cfg.temporal_voxel_history_s,
        )
        return state


__all__ = ["V6TemporalVoxelEnv", "V6TemporalVoxelEnvCfg"]
