from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "source/navrl_uav_v5/navrl_uav_v5/tasks/navrl_navigation"


def test_reward_v3_is_registered_as_an_isolated_experiment():
    registry = (TASK / "__init__.py").read_text()
    environment = (TASK / "v6_voxel_reward_v3_env.py").read_text()
    runner = (TASK / "agents/v6_voxel_reward_v3_rsl_rl_ppo_cfg.py").read_text()
    assert "Isaac-UAV-NavRL-V6-Front-Depth-Voxel-RewardV3-v0" in registry
    assert "class V6FrontDepthVoxelRewardV3Env(V6FrontDepthVoxelEnv)" in environment
    assert "class V6FrontDepthVoxelRewardV3EnvCfg(V6FrontDepthVoxelEnvCfg)" in environment
    assert 'experiment_name = "uav_v6_front_depth_voxel_reward_v3"' in runner


def test_reward_v3_is_action_conditioned_and_has_no_absolute_velocity_reward():
    environment = (TASK / "v6_voxel_reward_v3_env.py").read_text()
    assert "reward_velocity_weight = 0.0" in environment
    assert "command_velocity_g" in environment
    assert "closing_speed" in environment
    assert "proximity * normalized_closing" in environment
    assert "ttc_risk * normalized_closing" in environment
    assert '"velocity": torch.zeros_like(progress)' in environment
    assert "dynamic_top_risks = 2" in environment


def test_reward_v3_stall_penalty_is_safety_gated_and_sensor_only():
    environment = (TASK / "v6_voxel_reward_v3_env.py").read_text()
    assert "safe_to_advance" in environment
    assert "stall_safe_static_clearance_m" in environment
    assert "stall_safe_dynamic_clearance_m" in environment
    assert '"stall": self.cfg.reward_stall_weight * stall' in environment
    assert "_static_positions_local" not in environment
    assert "_dynamic_positions_truth" not in environment


def test_train_supports_exact_non_overwriting_result_directory():
    training = (ROOT / "scripts/train.py").read_text()
    assert '"--log_dir"' in training
    assert "--log_dir must be empty or absent" in training
