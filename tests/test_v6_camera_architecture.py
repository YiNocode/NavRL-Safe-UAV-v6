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
