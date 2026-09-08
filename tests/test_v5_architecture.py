"""Static architecture checks for forbidden V5 runtime dependencies."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = PROJECT_ROOT / "source" / "navrl_uav_v5" / "navrl_uav_v5"


def test_only_v5_gpu_task_is_registered() -> None:
    tasks_init = (PACKAGE / "tasks" / "__init__.py").read_text()
    registry = (PACKAGE / "tasks" / "navrl_navigation" / "__init__.py").read_text()
    assert "navrl_navigation" in tasks_init
    assert "Isaac-UAV-NavRL-V5-GPU-Direct-v0" in registry
    assert "static_navigation" not in tasks_init


def test_runtime_has_no_shield_ros2_or_px4_module() -> None:
    runtime_files = tuple(PACKAGE.rglob("*.py"))
    names = {path.name for path in runtime_files}
    assert "safety_shield.py" not in names
    assert "ros2_bridge.py" not in names
    assert "px4_control.py" not in names
    environment = (PACKAGE / "tasks" / "navrl_navigation" / "navrl_env.py").read_text()
    assert "project_safe_velocity" not in environment
    assert "safety_shield_enabled" not in environment
    assert "write_root_velocity_to_sim" in environment


def test_structured_policy_keeps_ray_matrix_for_cnn() -> None:
    actor = (PACKAGE / "tasks" / "navrl_navigation" / "agents" / "navrl_actor_critic.py").read_text()
    environment = (PACKAGE / "tasks" / "navrl_navigation" / "navrl_env.py").read_text()
    assert "static.unsqueeze(-3)" in actor
    assert '"static_obstacles": self._static_normalized' in environment
    assert '"dynamic_obstacles": self._dynamic_observation.state' in environment
    assert "ray_hits_w" not in actor


def test_policy_observation_is_sensor_derived() -> None:
    environment = (PACKAGE / "tasks" / "navrl_navigation" / "navrl_env.py").read_text()
    perception = (PACKAGE / "utils" / "gpu_navrl_perception.py").read_text()
    assert "self._depth_sensor.data.ray_hits_w" in environment
    assert "self._perception.update" in environment
    assert "integrate_hits" in perception
    assert "ray_cast" in perception
    assert "simulator_truth_policy_input" in environment


def test_task_requires_ground_takeoff_and_low_altitude_navigation() -> None:
    config = (PACKAGE / "tasks" / "navrl_navigation" / "navrl_env_cfg.py").read_text()
    environment = (PACKAGE / "tasks" / "navrl_navigation" / "navrl_env.py").read_text()
    drone = (PACKAGE / "assets" / "drone.py").read_text()
    assert "takeoff_start_height = 0.10" in config
    assert "goal_z_range = (0.80, 1.60)" in config
    assert "soft_flight_ceiling = 1.80" in config
    assert "termination_max_z = 2.20" in config
    assert "static_height_range = (2.60, 5.00)" in config
    assert "candidate_starts[..., 2] = self.cfg.takeoff_start_height" in environment
    assert "candidate_goals[..., 2] = self.cfg.goal_z_range[0]" in environment
    assert '"altitude": -self.cfg.reward_altitude_weight' in environment
    assert "height_range=self.cfg.static_height_range" in environment
    assert "DRONE_CFG.init_state.pos = (0.0, 0.0, 0.10)" in drone


def test_visualization_synchronizes_obstacle_render_and_physx_poses() -> None:
    play = (PROJECT_ROOT / "scripts" / "play.py").read_text()
    assert '"StaticObstacle", raw_env.cfg.static_pool_size' in play
    assert '"DynamicObstacle", raw_env.cfg.dynamic_pool_size' in play
    assert "raw_env._static_collection.data.object_pos_w[0]" in play
    assert "raw_env._static_collection.data.object_quat_w[0]" in play
    assert "raw_env._dynamic_collection.data.object_pos_w[0]" in play
    assert "raw_env._dynamic_collection.data.object_quat_w[0]" in play
    assert "render/PhysX position error" in play
    assert "render/PhysX quaternion error" in play
