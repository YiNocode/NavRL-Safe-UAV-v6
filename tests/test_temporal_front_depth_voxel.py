from pathlib import Path

import torch

from navrl_uav_v5.utils.gpu_temporal_front_depth_voxel import GpuTemporalFrontDepthVoxelizer


ROOT = Path(__file__).resolve().parents[1]


def _inputs(depth_value: float = 3.5):
    depth = torch.full((1, 4, 4, 1), torch.inf)
    depth[0, 0, 0, 0] = depth_value
    intrinsics = torch.tensor([[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]]])
    position = torch.zeros((1, 3))
    quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    return depth, intrinsics, position, quaternion


def _voxelizer():
    return GpuTemporalFrontDepthVoxelizer(
        1,
        (4, 4),
        (4, 4, 4),
        history_duration_s=1.0,
        forward_range_m=(0.0, 4.0),
        lateral_range_m=(-2.0, 2.0),
        vertical_range_m=(-2.0, 2.0),
        far_m=4.0,
        pixel_stride=4,
        free_samples=3,
    )


def test_temporal_voxel_retains_unobserved_history_with_decay():
    voxelizer = _voxelizer()
    depth, intrinsics, position, quaternion = _inputs()
    first = voxelizer.update(depth, intrinsics, position, quaternion, 0.1).clone()
    assert first[:, 0].sum() == 1.0
    empty_depth = torch.full_like(depth, torch.inf)
    second = voxelizer.update(empty_depth, intrinsics, position, quaternion, 0.1).clone()
    assert 0.0 < second[:, 0].max() < 1.0
    assert second[:, 2].sum() > 0.0
    assert 0.0 < second[:, 3].max() < 1.0


def test_temporal_voxel_ego_translation_moves_history_in_current_frame():
    voxelizer = _voxelizer()
    depth, intrinsics, position, quaternion = _inputs()
    first = voxelizer.update(depth, intrinsics, position, quaternion, 0.1).clone()
    first_x = int(first[0, 0].flatten().argmax().item()) % 4
    position[:, 2] = 1.0  # ROS optical forward is +Z in world for identity quaternion.
    empty_depth = torch.full_like(depth, torch.inf)
    second = voxelizer.update(empty_depth, intrinsics, position, quaternion, 0.1).clone()
    second_x = int(second[0, 0].flatten().argmax().item()) % 4
    assert second_x == first_x - 1


def test_dynamic_mask_prevents_persistent_occupied_write_and_reset_clears_state():
    voxelizer = _voxelizer()
    depth, intrinsics, position, quaternion = _inputs()
    dynamic_mask = torch.zeros((1, 4, 4), dtype=torch.bool)
    dynamic_mask[0, 0, 0] = True
    state = voxelizer.update(
        depth, intrinsics, position, quaternion, 0.1, dynamic_mask=dynamic_mask
    )
    assert state[:, 0].sum() == 0.0
    voxelizer.reset(torch.tensor([0]))
    assert voxelizer.state.sum() == 0.0
    assert not voxelizer.has_previous.any()


def test_v4_contract_and_four_channel_encoder_configuration():
    task = ROOT / "source/navrl_uav_v5/navrl_uav_v5/tasks/navrl_navigation"
    environment = (task / "v6_temporal_voxel_env.py").read_text()
    runner = (task / "agents/v6_temporal_voxel_rsl_rl_ppo_cfg.py").read_text()
    registry = (task / "__init__.py").read_text()
    assert '"front_voxel": [4, 16, 32, 32]' in environment
    assert "temporal_voxel_history_s = 1.5" in environment
    assert "dynamic_mask=self._motion_mask" in environment
    assert "voxel_channels = 4" in runner
    assert "Isaac-UAV-NavRL-V6-Front-Depth-TemporalVoxel-V4-v0" in registry
