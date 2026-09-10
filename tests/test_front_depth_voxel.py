import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def _load(relative_path: str, module_name: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


VOXEL = _load(
    "source/navrl_uav_v5/navrl_uav_v5/utils/gpu_front_depth_voxel.py",
    "gpu_front_depth_voxel",
)
ENCODER = _load(
    "source/navrl_uav_v5/navrl_uav_v5/tasks/navrl_navigation/agents/front_voxel_encoder.py",
    "front_voxel_encoder",
)


def test_voxel_contract_preserves_unknown_and_separates_free_from_occupied():
    voxelizer = VOXEL.GpuFrontDepthVoxelizer(
        1, (4, 4), (4, 4, 4),
        forward_range_m=(0.0, 4.0), lateral_range_m=(-2.0, 2.0),
        vertical_range_m=(-2.0, 2.0), far_m=4.0, pixel_stride=4, free_samples=3,
    )
    depth = torch.full((1, 4, 4, 1), torch.inf)
    intrinsics = torch.tensor([[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]]])
    empty = voxelizer.update(depth, intrinsics)
    assert empty.shape == (1, 3, 4, 4, 4)
    assert empty.sum() == 0.0

    depth[0, 0, 0, 0] = 3.5
    state = voxelizer.update(depth, intrinsics)
    occupied, free, observed = state[0]
    assert occupied.sum() == 1.0
    assert free.sum() > 0.0
    assert not torch.any((occupied > 0.0) & (free > 0.0))
    assert torch.equal(observed, torch.maximum(occupied, free))
    assert observed.sum() < observed.numel()


def test_front_voxel_encoder_outputs_128_features():
    encoder = ENCODER.FrontVoxelEncoder(embedding_dim=128)
    output = encoder(torch.zeros((2, 3, 16, 32, 32)))
    assert output.shape == (2, 128)
    assert torch.isfinite(output).all()


def test_voxel_variant_keeps_image_variant_and_uses_distinct_task():
    registry = (ROOT / "source/navrl_uav_v5/navrl_uav_v5/tasks/navrl_navigation/__init__.py").read_text()
    environment = (ROOT / "source/navrl_uav_v5/navrl_uav_v5/tasks/navrl_navigation/v6_voxel_env.py").read_text()
    runner = (ROOT / "source/navrl_uav_v5/navrl_uav_v5/tasks/navrl_navigation/agents/v6_voxel_rsl_rl_ppo_cfg.py").read_text()
    assert "Isaac-UAV-NavRL-V6-Front-Depth-M1-v0" in registry
    assert "Isaac-UAV-NavRL-V6-Front-Depth-Voxel-v0" in registry
    assert '"front_voxel": [3, *voxel_grid_shape]' in environment
    assert '"internal_state": 8' in environment
    assert '"dynamic_obstacles": [5, 10]' in environment
    assert "V6FrontDepthCameraEnvCfg" in environment
    assert '"dynamic_obstacles"' in environment
    assert 'experiment_name = "uav_v6_front_depth_voxel"' in runner
    assert '["front_voxel", "internal_state", "dynamic_obstacles"]' in runner
