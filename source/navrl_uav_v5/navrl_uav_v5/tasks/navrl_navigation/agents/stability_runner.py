"""Best-checkpoint and early-stopping support for V5 on-policy training."""

from __future__ import annotations

import json
import math
import shutil
from collections import deque
from pathlib import Path
from typing import Any

import torch
from rsl_rl.runners import OnPolicyRunner


class TrainingEarlyStopped(RuntimeError):
    """Raised after the runner safely checkpoints an early-stopped policy."""


class StabilityOnPolicyRunner(OnPolicyRunner):
    """Track a rolling episode metric and preserve the best policy.

    The base RSL-RL rollout/update loop remains unchanged.  This subclass uses
    the episode metrics already produced by that loop, so no simulator-truth
    observation is added to the policy.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.best_metric_key = str(self.cfg.get("best_metric_key", "Episode/success"))
        self.best_metric_window_episodes = int(
            self.cfg.get("best_metric_window_episodes", 500)
        )
        self.best_metric_min_delta = float(self.cfg.get("best_metric_min_delta", 0.002))
        self.early_stopping_min_episodes = int(
            self.cfg.get("early_stopping_min_episodes", 5_000)
        )
        self.early_stopping_patience_episodes = int(
            self.cfg.get("early_stopping_patience_episodes", 10_000)
        )
        self.early_stopping_max_drop = float(self.cfg.get("early_stopping_max_drop", 0.08))
        self.early_stopping_degradation_patience_episodes = int(
            self.cfg.get("early_stopping_degradation_patience_episodes", 2_000)
        )
        if self.best_metric_window_episodes < 1:
            raise ValueError("best_metric_window_episodes must be positive")
        if self.early_stopping_min_episodes < self.best_metric_window_episodes:
            raise ValueError("early_stopping_min_episodes must cover the metric window")
        if (
            self.early_stopping_patience_episodes < 1
            or self.early_stopping_degradation_patience_episodes < 1
        ):
            raise ValueError("episode-based early-stopping patience values must be positive")
        if self.best_metric_min_delta < 0.0 or self.early_stopping_max_drop <= 0.0:
            raise ValueError("metric delta must be non-negative and max drop must be positive")

        self._metric_history: deque[float] = deque(maxlen=self.best_metric_window_episodes)
        self._completed_episodes = 0
        self._best_metric = -math.inf
        self._best_iteration = -1
        self._last_improvement_iteration = -1
        self._last_improvement_episode = 0
        self._degradation_episodes = 0
        self._best_source_checkpoint: str | None = None

    def initialize_best_checkpoint(self, source: Path, metric: float) -> None:
        """Seed best tracking from an externally evaluated checkpoint."""
        if self.log_dir is None:
            raise RuntimeError("A log directory is required for best-checkpoint tracking")
        source = source.expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        if not math.isfinite(metric):
            raise ValueError("Initial best metric must be finite")
        destination = Path(self.log_dir) / "model_best.pt"
        shutil.copy2(source, destination)
        self._best_metric = float(metric)
        self._best_iteration = self.current_learning_iteration
        self._last_improvement_iteration = self.current_learning_iteration
        self._last_improvement_episode = self._completed_episodes
        self._best_source_checkpoint = str(source)
        self._write_stability_state("initialized")
        print(
            f"[STABILITY] Initialized model_best.pt from {source.name}: "
            f"{self.best_metric_key}={self._best_metric:.4f}",
            flush=True,
        )

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        """Run normal logging, then update best/early-stop state."""
        super().log(locs, width=width, pad=pad)
        episode_metrics = self._extract_episode_metrics(locs.get("ep_infos", []))
        if not episode_metrics:
            return
        iteration = int(locs["it"])
        self._metric_history.extend(episode_metrics)
        completed_now = len(episode_metrics)
        self._completed_episodes += completed_now
        if len(self._metric_history) < self.best_metric_window_episodes:
            return

        rolling_metric = sum(self._metric_history) / len(self._metric_history)
        if self.writer is not None:
            self.writer.add_scalar("Stability/rolling_success", rolling_metric, iteration)
            self.writer.add_scalar("Stability/best_success", self._best_metric, iteration)
            self.writer.add_scalar(
                "Stability/completed_episodes", self._completed_episodes, iteration
            )

        if rolling_metric > self._best_metric + self.best_metric_min_delta:
            self._best_metric = rolling_metric
            self._best_iteration = iteration
            self._last_improvement_iteration = iteration
            self._last_improvement_episode = self._completed_episodes
            self._degradation_episodes = 0
            self.save(
                str(Path(self.log_dir) / "model_best.pt"),
                infos={
                    "best_metric": self._best_metric,
                    "best_iteration": iteration,
                    "completed_episodes": self._completed_episodes,
                },
            )
            self._write_stability_state("improved")
            print(
                f"[STABILITY] New best at iteration {iteration}: "
                f"rolling {self.best_metric_key}={rolling_metric:.4f} "
                f"over {len(self._metric_history)} episodes",
                flush=True,
            )
        elif rolling_metric < self._best_metric - self.early_stopping_max_drop:
            self._degradation_episodes += completed_now
        else:
            self._degradation_episodes = 0

        if self._completed_episodes < self.early_stopping_min_episodes:
            return
        if self._last_improvement_iteration < 0:
            self._last_improvement_iteration = iteration

        reason: str | None = None
        if self._degradation_episodes >= self.early_stopping_degradation_patience_episodes:
            reason = (
                f"rolling metric {rolling_metric:.4f} remained more than "
                f"{self.early_stopping_max_drop:.4f} below best {self._best_metric:.4f} for "
                f"{self._degradation_episodes} completed episodes"
            )
        elif (
            self._completed_episodes - self._last_improvement_episode
            >= self.early_stopping_patience_episodes
        ):
            reason = (
                f"no rolling-metric improvement greater than {self.best_metric_min_delta:.4f} "
                f"for {self._completed_episodes - self._last_improvement_episode} "
                "completed episodes"
            )
        if reason is None:
            return

        self.save(
            str(Path(self.log_dir) / "model_early_stop.pt"),
            infos={"reason": reason, "best_metric": self._best_metric, "best_iteration": self._best_iteration},
        )
        self._write_stability_state("early_stopped", reason=reason, rolling_metric=rolling_metric)
        if self.writer is not None and hasattr(self.writer, "flush"):
            self.writer.flush()
        raise TrainingEarlyStopped(reason)

    def _extract_episode_metrics(self, ep_infos: list[dict]) -> list[float]:
        values: list[torch.Tensor] = []
        for episode_info in ep_infos:
            if self.best_metric_key not in episode_info:
                continue
            value = episode_info[self.best_metric_key]
            value_tensor = torch.as_tensor(value, device=self.device, dtype=torch.float32).reshape(-1)
            values.append(value_tensor)
        if not values:
            return []
        return torch.cat(values).detach().cpu().tolist()

    def _write_stability_state(self, status: str, **extra: Any) -> None:
        if self.log_dir is None:
            return
        state = {
            "status": status,
            "metric_key": self.best_metric_key,
            "best_metric": self._best_metric,
            "best_iteration": self._best_iteration,
            "last_improvement_iteration": self._last_improvement_iteration,
            "last_improvement_episode": self._last_improvement_episode,
            "completed_episodes": self._completed_episodes,
            "metric_window_episodes": self.best_metric_window_episodes,
            "source_checkpoint": self._best_source_checkpoint,
            **extra,
        }
        Path(self.log_dir, "stability_state.json").write_text(json.dumps(state, indent=2) + "\n")


__all__ = ["StabilityOnPolicyRunner", "TrainingEarlyStopped"]
