"""Regression tests for episode-weighted best-checkpoint statistics."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / (
    "source/navrl_uav_v5/navrl_uav_v5/tasks/navrl_navigation/agents/stability_runner.py"
)


def _load_runner(monkeypatch):
    runners = types.ModuleType("rsl_rl.runners")
    runners.OnPolicyRunner = type("OnPolicyRunner", (), {})
    package = types.ModuleType("rsl_rl")
    package.runners = runners
    monkeypatch.setitem(sys.modules, "rsl_rl", package)
    monkeypatch.setitem(sys.modules, "rsl_rl.runners", runners)
    spec = importlib.util.spec_from_file_location("stability_runner_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.StabilityOnPolicyRunner


def test_extract_episode_metrics_preserves_each_episode_outcome(monkeypatch):
    StabilityOnPolicyRunner = _load_runner(monkeypatch)
    runner = StabilityOnPolicyRunner.__new__(StabilityOnPolicyRunner)
    runner.best_metric_key = "Episode/success"
    runner.device = "cpu"

    metrics = runner._extract_episode_metrics(
        [
            {"Episode/success": torch.tensor([1.0])},
            {"Episode/success": torch.tensor([0.0, 0.0, 1.0])},
        ]
    )

    assert metrics == [1.0, 0.0, 0.0, 1.0]
    assert sum(metrics) / len(metrics) == 0.5


def test_extract_episode_metrics_ignores_updates_without_completed_episodes(monkeypatch):
    StabilityOnPolicyRunner = _load_runner(monkeypatch)
    runner = StabilityOnPolicyRunner.__new__(StabilityOnPolicyRunner)
    runner.best_metric_key = "Episode/success"
    runner.device = "cpu"

    assert runner._extract_episode_metrics([{"Episode/return": torch.tensor([1.0])}]) == []
