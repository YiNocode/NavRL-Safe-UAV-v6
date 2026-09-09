from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/"source"/"navrl_uav_v5"/"navrl_uav_v5"/"tasks"/"navrl_navigation"
def test_v6_m1_contract():
    cfg=(TASK/"v6_camera_env.py").read_text()
    reg=(TASK/"__init__.py").read_text()
    assert "Isaac-UAV-NavRL-V5-GPU-Direct-v0" in reg
    assert "Isaac-UAV-NavRL-V6-Front-Depth-M1-v0" in reg
    assert "TiledCameraCfg" in cfg
    assert 'data_types=["distance_to_image_plane"]' in cfg
    assert '"front_depth":[camera_height,camera_width,1]' in cfg
    assert "MultiMeshRayCaster" not in cfg and "static_obstacles" not in cfg
    geometry=(ROOT/"source"/"navrl_uav_v5"/"navrl_uav_v5"/"utils"/"gpu_camera_geometry.py").read_text()
    assert "backproject_axial_depth" in geometry
    assert "optical_points_to_world" in geometry

def test_v6_m3_batched_contract():
    cfg=(TASK/"v6_camera_env.py").read_text()
    smoke=(ROOT/"scripts"/"random_agent_v6.py").read_text()
    assert "M1 requires exactly one environment" not in cfg
    assert "(self.num_envs,6)" in cfg
    assert "(self.num_envs,self.cfg.camera_height,self.cfg.camera_width,1)" in cfg
    assert "num_envs=args.num_envs" in smoke
    assert "env_frames_s=" in smoke
    assert "cuda_peak_allocated_mib=" in smoke

def test_v6_m4_finite_fov_static_contract():
    cfg=(TASK/"v6_camera_env.py").read_text()
    encoder=(TASK/"agents"/"front_depth_encoder.py").read_text()
    assert 'observation_space={"front_depth":[camera_height,camera_width,1]}' in cfg
    assert "static_embedding_dim=128" in cfg
    assert "torch.isfinite(depth)" in encoder
    assert "torch.stack((proximity, valid.to(depth.dtype)), dim=1)" in encoder
    assert "nn.Conv2d(2, 16" in encoder
    assert "[B,128]" in encoder
    assert "36,7" not in encoder and "StaticObstacleEncoder" not in encoder
