"""V6 camera-depth navigation variant with a finite-FOV voxel policy input."""

from __future__ import annotations

import torch
from isaaclab.utils import configclass

from navrl_uav_v5.utils.gpu_front_depth_voxel import GpuFrontDepthVoxelizer
from .v6_camera_env import V6FrontDepthCameraEnv, V6FrontDepthCameraEnvCfg


@configclass
class V6FrontDepthVoxelEnvCfg(V6FrontDepthCameraEnvCfg):
    """Same experiment configuration as V6 depth-image, changing only static input."""

    voxel_grid_shape = (16, 32, 32)  # Z, Y(right), X(forward)
    voxel_forward_range_m = (0.0, 5.0)
    voxel_lateral_range_m = (-2.5, 2.5)
    voxel_vertical_range_m = (-2.0, 2.0)
    voxel_pixel_stride = 4
    voxel_free_samples = 16
    observation_space = {
        "front_voxel": [3, *voxel_grid_shape],
        "internal_state": 8,
        "dynamic_obstacles": [5, 10],
    }


class V6FrontDepthVoxelEnv(V6FrontDepthCameraEnv):
    """Use current front depth to construct explicit occupied/free/observed voxels."""

    cfg: V6FrontDepthVoxelEnvCfg

    def _initialize_perception(self) -> None:
        super()._initialize_perception()
        cfg = self.cfg
        self._voxelizer = GpuFrontDepthVoxelizer(
            self.num_envs,
            (cfg.camera_height, cfg.camera_width),
            cfg.voxel_grid_shape,
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
        super()._reset_perception(env_ids)
        self._voxelizer.reset(env_ids)

    def _perception_memory_bytes(self) -> int:
        return super()._perception_memory_bytes() + self._voxelizer.memory_bytes

    @torch.no_grad()
    def _update_perception(self) -> None:
        super()._update_perception()
        if (
            self._voxel_cache_step == self.common_step_counter
            and self._voxel_cache_reset_version == self._perception_reset_version
        ):
            return
        self._voxelizer.update(
            self._camera.data.output["distance_to_image_plane"],
            self._camera.data.intrinsic_matrices,
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
                self.num_envs, 3, *self.cfg.voxel_grid_shape
            )
            assert torch.isfinite(observations["front_voxel"]).all()
            assert observations["internal_state"].shape == (self.num_envs, 8)
            assert observations["dynamic_obstacles"].shape == (self.num_envs, 5, 10)
        return observations

    def get_perception_state(self) -> dict[str, torch.Tensor | str | int]:
        state = super().get_perception_state()
        state.update(
            source="isaac_rtx_front_depth_then_gpu_local_voxel_and_temporal_tracking",
            front_voxel=self._voxelizer.state.clone(),
        )
        return state


__all__ = ["V6FrontDepthVoxelEnv", "V6FrontDepthVoxelEnvCfg"]
