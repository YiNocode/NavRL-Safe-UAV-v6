from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "source/navrl_uav_v5/navrl_uav_v5/tasks/navrl_navigation"


def test_safety_reward_is_an_isolated_voxel_task():
    registry = (TASK / "__init__.py").read_text()
    environment = (TASK / "v6_voxel_safety_env.py").read_text()
    runner = (TASK / "agents/v6_voxel_safety_rsl_rl_ppo_cfg.py").read_text()
    assert "Isaac-UAV-NavRL-V6-Front-Depth-Voxel-v0" in registry
    assert "Isaac-UAV-NavRL-V6-Front-Depth-Voxel-SafetyReward-v0" in registry
    assert "class V6FrontDepthVoxelSafetyEnv(V6FrontDepthVoxelEnv)" in environment
    assert "class V6FrontDepthVoxelSafetyEnvCfg(V6FrontDepthVoxelEnvCfg)" in environment
    assert 'experiment_name = "uav_v6_front_depth_voxel_safety_reward"' in runner


def test_safety_reward_uses_sensor_clearance_tail_and_tracked_ttc():
    environment = (TASK / "v6_voxel_safety_env.py").read_text()
    assert "torch.topk(clearance" in environment
    assert "static_risk_pixel_fraction = 0.01" in environment
    assert "tracked.positions_w" in environment
    assert "tracked.velocities_w" in environment
    assert "dynamic_ttc_horizon_s = 2.00" in environment
    assert "_dynamic_observation.surface_distances" in environment
    assert "_static_positions_local" not in environment
    assert "_dynamic_positions_truth" not in environment


def test_reward_rebalances_speed_collision_and_goal_without_changing_policy_contract():
    environment = (TASK / "v6_voxel_safety_env.py").read_text()
    actor = (TASK / "agents/v6_voxel_actor_critic.py").read_text()
    assert "reward_velocity_weight = 0.10" in environment
    assert "reward_collision_penalty = 40.0" in environment
    assert "reward_goal_bonus = 30.0" in environment
    assert 'required = {"front_voxel", "internal_state", "dynamic_obstacles"}' in actor
    assert "self.fused_feature_dim != 200" in actor
