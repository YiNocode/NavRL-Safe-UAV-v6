"""Performance-based obstacle curriculum with no automatic regression."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class CurriculumSnapshot:
    """Metrics for the current level after consuming completed episodes."""

    level: int
    num_obstacles: int
    completed_at_level: int
    window_episodes: int
    success_rate: float
    collision_rate: float
    timeout_rate: float
    mean_return: float
    promoted: bool
    next_level: int
    next_num_obstacles: int


class ObstacleCurriculumManager:
    """Promote when the latest complete-episode window reaches a target rate."""

    def __init__(
        self,
        levels: tuple[int, ...] = (0, 5, 10, 20, 30, 40),
        window_episodes: int = 200,
        min_completed_episodes: int = 200,
        promotion_success_rate: float = 0.80,
        initial_num_obstacles: int = 0,
    ) -> None:
        if not levels or levels[0] < 0 or tuple(sorted(set(levels))) != levels:
            raise ValueError("levels must be a strictly increasing tuple of non-negative obstacle counts")
        if window_episodes <= 0 or min_completed_episodes <= 0:
            raise ValueError("episode window sizes must be positive")
        if min_completed_episodes < window_episodes:
            raise ValueError("min_completed_episodes must be at least window_episodes")
        if not 0.0 <= promotion_success_rate <= 1.0:
            raise ValueError("promotion_success_rate must lie in [0, 1]")
        if initial_num_obstacles not in levels:
            raise ValueError("initial_num_obstacles must be one of the configured levels")
        self.levels = levels
        self.window_episodes = window_episodes
        self.min_completed_episodes = min_completed_episodes
        self.promotion_success_rate = promotion_success_rate
        self._level_index = levels.index(initial_num_obstacles)
        self._completed_at_level = 0
        self._success = deque(maxlen=window_episodes)
        self._collision = deque(maxlen=window_episodes)
        self._timeout = deque(maxlen=window_episodes)
        self._returns = deque(maxlen=window_episodes)
        self._promotion_history: list[dict[str, int | float | bool]] = []

    @property
    def level(self) -> int:
        return self._level_index

    @property
    def num_obstacles(self) -> int:
        return self.levels[self._level_index]

    @property
    def is_final_level(self) -> bool:
        return self._level_index == len(self.levels) - 1

    def update(
        self,
        success: Iterable[float],
        collision: Iterable[float],
        timeout: Iterable[float],
        returns: Iterable[float],
        *,
        completed_count: int | None = None,
    ) -> CurriculumSnapshot:
        """Consume terminal episode results and promote at most one level."""
        success_values = [float(value) for value in success]
        collision_values = [float(value) for value in collision]
        timeout_values = [float(value) for value in timeout]
        return_values = [float(value) for value in returns]
        batch_size = len(success_values)
        if not (
            len(collision_values) == batch_size
            and len(timeout_values) == batch_size
            and len(return_values) == batch_size
        ):
            raise ValueError("all curriculum metric batches must have equal length")
        completed_count = batch_size if completed_count is None else completed_count
        if completed_count < batch_size:
            raise ValueError("completed_count cannot be smaller than the supplied metric batch")
        self._success.extend(success_values)
        self._collision.extend(collision_values)
        self._timeout.extend(timeout_values)
        self._returns.extend(return_values)
        self._completed_at_level += completed_count

        window_size = len(self._success)
        success_rate = sum(self._success) / window_size if window_size else 0.0
        collision_rate = sum(self._collision) / window_size if window_size else 0.0
        timeout_rate = sum(self._timeout) / window_size if window_size else 0.0
        mean_return = sum(self._returns) / window_size if window_size else 0.0
        old_level = self._level_index
        old_obstacles = self.num_obstacles
        promoted = (
            not self.is_final_level
            and self._completed_at_level >= self.min_completed_episodes
            and window_size >= self.window_episodes
            and success_rate >= self.promotion_success_rate
        )
        if promoted:
            self._level_index += 1

        snapshot = CurriculumSnapshot(
            level=old_level,
            num_obstacles=old_obstacles,
            completed_at_level=self._completed_at_level,
            window_episodes=window_size,
            success_rate=success_rate,
            collision_rate=collision_rate,
            timeout_rate=timeout_rate,
            mean_return=mean_return,
            promoted=promoted,
            next_level=self._level_index,
            next_num_obstacles=self.num_obstacles,
        )
        if promoted:
            self._promotion_history.append(asdict(snapshot))
            self._completed_at_level = 0
            self._success.clear()
            self._collision.clear()
            self._timeout.clear()
            self._returns.clear()
        return snapshot

    def state_dict(self) -> dict[str, object]:
        """Return JSON-serializable state and promotion history."""
        return {
            "levels": list(self.levels),
            "level": self.level,
            "num_obstacles": self.num_obstacles,
            "is_final_level": self.is_final_level,
            "completed_at_level": self._completed_at_level,
            "window_episodes": len(self._success),
            "promotion_success_rate": self.promotion_success_rate,
            "promotion_history": list(self._promotion_history),
        }


__all__ = ["CurriculumSnapshot", "ObstacleCurriculumManager"]
