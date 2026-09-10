from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/"source"/"navrl_uav_v5"/"navrl_uav_v5"/"tasks"/"navrl_navigation"
def test_v6_m1_contract():
    cfg=(TASK/"v6_camera_env.py").read_text()
    reg=(TASK/"__init__.py").read_text()
    assert "Isaac-UAV-NavRL-V5-GPU-Direct-v0" in reg
    assert "Isaac-UAV-NavRL-V6-Front-Depth-M1-v0" in reg
    assert "TiledCameraCfg" in cfg
    assert 'data_types=["distance_to_image_plane"]' in cfg.replace(" ", "")
    assert '"front_depth":[camera_height,camera_width,1]' in cfg.replace(" ", "")
    assert "MultiMeshRayCaster" not in cfg and "static_obstacles" not in cfg
    geometry=(ROOT/"source"/"navrl_uav_v5"/"navrl_uav_v5"/"utils"/"gpu_camera_geometry.py").read_text()
    assert "backproject_axial_depth" in geometry
    assert "optical_points_to_world" in geometry

def test_v6_m3_batched_contract():
    cfg=(TASK/"v6_camera_env.py").read_text()
    smoke=(ROOT/"scripts"/"random_agent_v6.py").read_text()
    assert "M1 requires exactly one environment" not in cfg
    compact=cfg.replace(" ", "").replace("\n", "")
    assert "(self.num_envs,self.cfg.camera_height,self.cfg.camera_width,1)" in compact
    assert "num_envs=args.num_envs" in smoke
    assert "env_frames_s=" in smoke
    assert "cuda_peak_allocated_mib=" in smoke

def test_v6_m4_finite_fov_static_contract():
    cfg=(TASK/"v6_camera_env.py").read_text()
    encoder=(TASK/"agents"/"front_depth_encoder.py").read_text()
    compact=cfg.replace(" ", "")
    assert '"front_depth":[camera_height,camera_width,1]' in compact
    assert "static_embedding_dim=128" in compact
    assert "torch.isfinite(depth)" in encoder
    assert "torch.stack((proximity, valid.to(depth.dtype)), dim=1)" in encoder
    assert "nn.Conv2d(2, 16" in encoder
    assert "[B,128]" in encoder
    assert "36,7" not in encoder and "StaticObstacleEncoder" not in encoder

def test_v6_m5_dynamic_depth_contract():
    cfg=(TASK/"v6_camera_env.py").read_text()
    tracker=(ROOT/"source"/"navrl_uav_v5"/"navrl_uav_v5"/"utils"/"gpu_camera_motion_tracker.py").read_text()
    compact=cfg.replace(" ", "")
    assert '"dynamic_obstacles":[max_dynamic_tracks,dynamic_state_dim]' in compact
    assert "max_dynamic_tracks=5" in compact and "dynamic_state_dim=10" in compact
    assert "_previous_depth_in_current_camera" in tracker
    assert "scatter_reduce_" in tracker
    assert "motion_mask" in tracker and "_select_detections" in tracker
    assert "association_distance_m" in tracker and "max_missed_frames" in tracker
    assert "obstacle_positions" not in tracker and "semantic_labels" not in tracker

def test_v6_m6_actor_fusion_contract():
    cfg=(TASK/"v6_camera_env.py").read_text()
    actor=(TASK/"agents"/"v6_actor_critic.py").read_text()
    runner=(TASK/"agents"/"v6_rsl_rl_ppo_cfg.py").read_text()
    registry=(TASK/"__init__.py").read_text()
    assert '"internal_state":internal_state_dim' in cfg.replace(" ", "")
    assert "navrl_internal_state(" in cfg
    assert "FrontDepthEncoder(" in actor
    assert "_mlp((self.dynamic_dim, 128, 64))" in actor
    assert "self.fused_feature_dim = self.static_embedding_dim + 64 + self.internal_state_dim" in actor
    assert "self.fused_feature_dim != 200" in actor
    assert "_mlp((self.fused_feature_dim, 256, 256))" in actor
    assert '["front_depth", "internal_state", "dynamic_obstacles"]' in runner
    assert 'class_name = "V6NavRLActorCritic"' in runner
    assert "rsl_rl_cfg_entry_point" in registry and "V6NavRLGpuPPORunnerCfg" in registry

def test_v6_m6_does_not_claim_v5_checkpoint_compatibility():
    actor=(TASK/"agents"/"v6_actor_critic.py").read_text()
    assert "StaticObstacleEncoder" not in actor
    assert "static_obstacles" not in actor
    assert "front_depth" in actor

def test_v6_m7_scaling_benchmark_contract():
    benchmark=(ROOT/"scripts"/"benchmark_scaling.py").read_text()
    assert "default=V6_TASK_ID" in benchmark
    assert '"environment_fps"' in benchmark and '"camera_fps"' in benchmark
    assert '"gpu_memory_mib"' in benchmark and '"memory_stable"' in benchmark
    assert "torch.cuda.mem_get_info" in benchmark
    assert '"device_used"' in benchmark
    assert '"step_latency_ms"' in benchmark
    assert '"policy_inference_ms"' in benchmark
    assert '"pre_physics"' in benchmark
    assert '"apply_action"' in benchmark
    assert '"observation"' in benchmark
    assert '"physics_render_wrapper_residual"' in benchmark
    assert "runner.learn" not in benchmark and "optimizer" not in benchmark

def test_v6_formal_navigation_gates_are_wired():
    cfg=(TASK/"v6_camera_env.py").read_text()
    base=(TASK/"navrl_env.py").read_text()
    runner=(TASK/"agents"/"v6_rsl_rl_ppo_cfg.py").read_text()
    assert "class V6FrontDepthCameraEnv(NavRLGpuEnv)" in cfg
    assert "takeoff_start_height = 1.50" in cfg
    assert "self._command_velocity_w" in base and "write_root_velocity_to_sim" in base
    assert '"progress"' in base and '"goal"' in base and '"collision"' in base
    assert "self._success | self._collision | self._out_of_bounds" in base
    assert "_randomize_static_obstacles" in base and "_randomize_dynamic_obstacles" in base
    assert '"Episode/success"' in base
    assert "max_iterations = 10_000" in runner
    assert "num_steps_per_env = 32" in runner

def test_v6_deterministic_evaluation_contract():
    evaluation=(ROOT/"scripts"/"evaluate_v6.py").read_text()
    environment=(TASK/"navrl_env.py").read_text()
    assert 'TASK_ID = "Isaac-UAV-NavRL-V6-Front-Depth-M1-v0"' in evaluation
    assert "get_inference_policy" in evaluation
    assert '"deterministic_policy": True' in evaluation
    assert '"success_rate"' in evaluation and '"collision_rate"' in evaluation
    assert '"timeout_rate"' in evaluation and '"out_of_bounds_rate"' in evaluation
    assert '"mean_path_length_m"' in evaluation
    assert '"p05_minimum_front_depth_m"' in evaluation
    assert '"checkpoint_sha256"' in evaluation
    assert '"episode_path_length"' in environment
    assert '"episode_min_static_distance"' in environment
