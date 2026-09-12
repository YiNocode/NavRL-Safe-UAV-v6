from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_deterministic_evaluator_has_sensor_attributed_failure_split():
    evaluation = (ROOT / "scripts/evaluate_v6.py").read_text()
    assert '"failure_diagnostics"' in evaluation
    assert '"sensor_attributed_not_contact_object_ground_truth"' in evaluation
    assert '"collision_attribution_counts"' in evaluation
    assert '"timeout_behavior_counts"' in evaluation
    assert '"leadup_means_by_outcome"' in evaluation
    assert "raw._static_raw" in evaluation
    assert "raw._dynamic_observation.surface_distances" in evaluation
    assert "_static_positions_local" not in evaluation
    assert "_dynamic_positions_truth" not in evaluation


def test_collision_categories_cover_front_dynamic_edge_and_unobserved():
    evaluation = (ROOT / "scripts/evaluate_v6.py").read_text()
    for category in (
        "tracked_dynamic_front_confirmed",
        "tracked_dynamic_no_front_depth",
        "front_fov_center_only",
        "front_fov_edge_only",
        "unobserved_or_outside_front_fov",
    ):
        assert category in evaluation


def test_timeout_categories_and_one_second_command_history_are_reported():
    evaluation = (ROOT / "scripts/evaluate_v6.py").read_text()
    assert '"--diagnostic_window_s"' in evaluation
    assert "history_commands" in evaluation
    assert "window_goal_progress_m" in evaluation
    assert "mean_forward_command_mps" in evaluation
    assert "mean_abs_lateral_command_mps" in evaluation
    assert "mean_abs_vertical_command_mps" in evaluation
    assert "hover_stall" in evaluation
    assert "maneuver_without_goal_progress" in evaluation
    assert "slow_progress_timeout" in evaluation
