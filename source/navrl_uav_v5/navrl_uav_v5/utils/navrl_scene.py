"""Deterministic scene generators matching NavRL's public Isaac setup."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StaticForestObstacle:
    center: tuple[float, float, float]
    size: tuple[float, float, float]


@dataclass(frozen=True)
class DynamicObstacleSpec:
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    is_column: bool


@dataclass(frozen=True)
class EpisodeSceneLayout:
    """One deterministic episode layout expressed in the local map frame.

    Geometry is selected from pre-spawned shape templates.  This lets Isaac
    move kinematic actors at reset without rebuilding PhysX collision shapes,
    while the selected template subset changes obstacle widths and heights
    between episodes.
    """

    static_template_indices: np.ndarray
    static_positions: np.ndarray
    dynamic_template_indices: np.ndarray
    dynamic_positions: np.ndarray
    dynamic_goals: np.ndarray
    dynamic_speeds: np.ndarray


def sample_static_forest(
    count: int,
    seed: int,
    map_half_extent: float = 20.0,
    horizontal_scale: float = 0.1,
) -> tuple[StaticForestObstacle, ...]:
    """Reproduce the modified discrete-height-field sampler used by NavRL.

    Widths are sampled on the original 0.4 m grid and heights from the four
    published intervals with probabilities ``[0.1, 0.15, 0.2, 0.55]``.
    """
    if count < 0:
        raise ValueError("count must be non-negative")
    generator = np.random.RandomState(seed)
    width_pixels = int((2.0 * map_half_extent) / horizontal_scale)
    shape_choices = np.arange(4, 11, 4)
    position_choices = np.arange(0, width_pixels, 4)
    height_intervals = ((1.0, 1.5), (1.5, 2.0), (2.0, 4.0), (4.0, 6.0))
    height_probabilities = (0.10, 0.15, 0.20, 0.55)
    history: list[tuple[int, int, int, int]] = []
    obstacles: list[StaticForestObstacle] = []

    def acceptable(x: int, y: int, width: int, length: int) -> bool:
        # This intentionally follows the public Orbit fork's pixel-space
        # ``good_distance`` helper, including its overlap allowance.
        for previous_x, previous_y, _, _ in history:
            dx = abs(previous_x - x) - width
            dy = abs(previous_y - y) - length
            if dx < 0 and dy < 0:
                continue
            distance = math.hypot(dx, dy)
            if 2 <= distance <= 10:
                return False
        return True

    for _ in range(count):
        interval_index = int(generator.choice(len(height_probabilities), 1, p=height_probabilities)[0])
        height = float(generator.uniform(*height_intervals[interval_index]))
        for _attempt in range(100_000):
            width_px = int(generator.choice(shape_choices))
            length_px = int(generator.choice(shape_choices))
            x_start = min(int(generator.choice(position_choices)), width_pixels - width_px)
            y_start = min(int(generator.choice(position_choices)), width_pixels - length_px)
            if acceptable(x_start, y_start, width_px, length_px) or not history:
                break
        history.append((x_start, y_start, width_px, length_px))
        width = width_px * horizontal_scale
        length = length_px * horizontal_scale
        center = (
            -map_half_extent + (x_start + 0.5 * width_px) * horizontal_scale,
            -map_half_extent + (y_start + 0.5 * length_px) * horizontal_scale,
            0.5 * height,
        )
        obstacles.append(StaticForestObstacle(center=center, size=(width, length, height)))
    return tuple(obstacles)


def sample_dynamic_obstacles(
    count: int,
    seed: int,
    map_half_extent: float = 20.0,
    flight_height: float = 4.5,
    width_scale_range: tuple[float, float] = (1.0, 1.0),
    height_scale_range: tuple[float, float] = (1.0, 1.0),
) -> tuple[DynamicObstacleSpec, ...]:
    """Generate NavRL's eight size/shape categories in deterministic order."""
    if count <= 0 or count % 8 != 0:
        raise ValueError("NavRL dynamic obstacle count must be a positive multiple of eight")
    for name, value in (("width_scale_range", width_scale_range), ("height_scale_range", height_scale_range)):
        if len(value) != 2 or value[0] <= 0.0 or value[1] < value[0]:
            raise ValueError(f"{name} must be a positive ordered pair")
    generator = np.random.RandomState(seed)
    per_category = count // 8
    preferred_distance = 2.0 * math.sqrt(map_half_extent * map_half_extent / count)
    accepted_xy: list[tuple[float, float]] = []
    obstacles: list[DynamicObstacleSpec] = []

    for category in range(8):
        base_width = 0.25 * (category % 4 + 1)
        is_column = category >= 4
        base_height = 5.0 if is_column else 1.0
        for _ in range(per_category):
            separation = preferred_distance
            attempts = 0
            while True:
                x = float(generator.uniform(-map_half_extent, map_half_extent))
                y = float(generator.uniform(-map_half_extent, map_half_extent))
                if all(math.hypot(x - px, y - py) > separation for px, py in accepted_xy):
                    break
                attempts += 1
                if attempts % 2000 == 0:
                    separation *= 0.8
            accepted_xy.append((x, y))
            width = base_width * float(generator.uniform(*width_scale_range))
            height = base_height * float(generator.uniform(*height_scale_range))
            z = 0.5 * height if is_column else float(generator.uniform(0.0, flight_height))
            obstacles.append(
                DynamicObstacleSpec(
                    center=(x, y, z),
                    size=(width, width, height),
                    is_column=is_column,
                )
            )
    return tuple(obstacles)


def sample_episode_scene_layout(
    *,
    seed: int,
    start: np.ndarray | tuple[float, float, float],
    goal: np.ndarray | tuple[float, float, float],
    static_template_sizes: np.ndarray,
    static_count: int,
    dynamic_template_sizes: np.ndarray,
    dynamic_template_columns: np.ndarray,
    dynamic_count: int,
    map_half_extent: float,
    flight_height: float,
    endpoint_clearance: float,
    static_minimum_separation: float,
    dynamic_minimum_separation: float,
    dynamic_local_range: tuple[float, float, float],
    dynamic_speed_range: tuple[float, float],
    maximum_attempts: int = 100_000,
) -> EpisodeSceneLayout:
    """Sample a collision-separated scene for one episode.

    The function is simulator independent and deterministic for ``seed``.
    Static and dynamic shapes come from template pools because changing a
    PhysX collider's scale at runtime is not reliable.  Random template subsets
    plus new poses produce different position/size/height combinations on every
    episode.  Dynamic start-to-goal segments are kept away from the UAV episode
    endpoints, not merely their initial positions.
    """

    start_array = np.asarray(start, dtype=np.float64)
    goal_array = np.asarray(goal, dtype=np.float64)
    static_sizes = np.asarray(static_template_sizes, dtype=np.float64)
    dynamic_sizes = np.asarray(dynamic_template_sizes, dtype=np.float64)
    dynamic_columns = np.asarray(dynamic_template_columns, dtype=np.bool_)
    if start_array.shape != (3,) or goal_array.shape != (3,):
        raise ValueError("start and goal must have shape (3,)")
    if static_sizes.ndim != 2 or static_sizes.shape[1:] != (3,):
        raise ValueError("static_template_sizes must have shape (P,3)")
    if dynamic_sizes.ndim != 2 or dynamic_sizes.shape[1:] != (3,):
        raise ValueError("dynamic_template_sizes must have shape (P,3)")
    if dynamic_columns.shape != (dynamic_sizes.shape[0],):
        raise ValueError("dynamic_template_columns must have shape (P,)")
    if not 0 <= static_count <= static_sizes.shape[0]:
        raise ValueError("static_count is outside the template pool")
    if not 0 <= dynamic_count <= dynamic_sizes.shape[0]:
        raise ValueError("dynamic_count is outside the template pool")
    if map_half_extent <= 0.0 or flight_height <= 0.0 or maximum_attempts <= 0:
        raise ValueError("map extents and maximum_attempts must be positive")
    if endpoint_clearance < 0.0 or static_minimum_separation < 0.0 or dynamic_minimum_separation < 0.0:
        raise ValueError("clearance and separation values must be non-negative")
    if len(dynamic_local_range) != 3 or any(value <= 0.0 for value in dynamic_local_range):
        raise ValueError("dynamic_local_range must contain three positive values")
    speed_low, speed_high = map(float, dynamic_speed_range)
    if speed_low <= 0.0 or speed_high < speed_low:
        raise ValueError("dynamic_speed_range must be a positive ordered pair")

    generator = np.random.RandomState(int(seed))
    static_indices = generator.choice(static_sizes.shape[0], static_count, replace=False).astype(np.int64)
    chosen_static_sizes = static_sizes[static_indices]
    static_positions: list[np.ndarray] = []

    def point_box_distance(point: np.ndarray, center: np.ndarray, size: np.ndarray) -> float:
        outside = np.maximum(np.abs(point - center) - 0.5 * size, 0.0)
        return float(np.linalg.norm(outside))

    def boxes_separated(
        candidate: np.ndarray,
        candidate_size: np.ndarray,
        centers: list[np.ndarray],
        sizes: np.ndarray,
        minimum: float,
    ) -> bool:
        for other_center, other_size in zip(centers, sizes, strict=False):
            outside_xy = np.maximum(
                np.abs(candidate[:2] - other_center[:2]) - 0.5 * (candidate_size[:2] + other_size[:2]),
                0.0,
            )
            if float(np.linalg.norm(outside_xy)) < minimum:
                return False
        return True

    for obstacle_index, size in enumerate(chosen_static_sizes):
        half_xy = 0.5 * size[:2]
        if np.any(half_xy >= map_half_extent):
            raise ValueError("static template is larger than the configured map")
        for _ in range(maximum_attempts):
            center = np.array(
                [
                    generator.uniform(-map_half_extent + half_xy[0], map_half_extent - half_xy[0]),
                    generator.uniform(-map_half_extent + half_xy[1], map_half_extent - half_xy[1]),
                    0.5 * size[2],
                ],
                dtype=np.float64,
            )
            endpoint_distance = min(
                point_box_distance(start_array, center, size),
                point_box_distance(goal_array, center, size),
            )
            if endpoint_distance < endpoint_clearance:
                continue
            if boxes_separated(
                center,
                size,
                static_positions,
                chosen_static_sizes[:obstacle_index],
                static_minimum_separation,
            ):
                static_positions.append(center)
                break
        else:
            raise RuntimeError("Could not sample a separated static-obstacle layout")
    static_position_array = np.asarray(static_positions, dtype=np.float64).reshape(static_count, 3)

    # Select approximately equal numbers from the eight NavRL shape groups.
    category_count = 8
    category_ids = np.arange(dynamic_sizes.shape[0], dtype=np.int64) * category_count // max(
        dynamic_sizes.shape[0], 1
    )
    dynamic_index_values: list[int] = []
    quota, remainder = divmod(dynamic_count, category_count)
    extra_categories = set(
        generator.choice(category_count, remainder, replace=False).tolist()
        if remainder
        else ()
    )
    for category in range(category_count):
        candidates = np.flatnonzero(category_ids == category)
        category_quota = quota + int(category in extra_categories)
        if category_quota > candidates.size:
            raise ValueError("dynamic template pool cannot satisfy balanced category selection")
        if category_quota:
            dynamic_index_values.extend(generator.choice(candidates, category_quota, replace=False).tolist())
    generator.shuffle(dynamic_index_values)
    dynamic_indices = np.asarray(dynamic_index_values, dtype=np.int64)
    chosen_dynamic_sizes = dynamic_sizes[dynamic_indices]
    chosen_columns = dynamic_columns[dynamic_indices]
    dynamic_positions: list[np.ndarray] = []
    dynamic_goals: list[np.ndarray] = []

    def point_segment_distance(point: np.ndarray, start_value: np.ndarray, end_value: np.ndarray) -> float:
        segment = end_value - start_value
        denominator = float(np.dot(segment, segment))
        if denominator <= 1.0e-12:
            return float(np.linalg.norm(point - start_value))
        fraction = float(np.clip(np.dot(point - start_value, segment) / denominator, 0.0, 1.0))
        return float(np.linalg.norm(point - (start_value + fraction * segment)))

    local_range = np.asarray(dynamic_local_range, dtype=np.float64)
    endpoints = (start_array, goal_array)
    for obstacle_index, (size, is_column) in enumerate(zip(chosen_dynamic_sizes, chosen_columns, strict=False)):
        radius = 0.5 * float(max(size[0], size[1]))
        for _ in range(maximum_attempts):
            position = np.array(
                [
                    generator.uniform(-map_half_extent + radius, map_half_extent - radius),
                    generator.uniform(-map_half_extent + radius, map_half_extent - radius),
                    0.5 * size[2] if is_column else generator.uniform(radius, max(radius, flight_height - radius)),
                ],
                dtype=np.float64,
            )
            offset = generator.uniform(-local_range, local_range)
            target = position + offset
            target[:2] = np.clip(target[:2], -map_half_extent + radius, map_half_extent - radius)
            target[2] = 0.5 * size[2] if is_column else np.clip(target[2], radius, max(radius, flight_height - radius))
            required_endpoint_distance = endpoint_clearance + radius
            if any(
                point_segment_distance(endpoint, position, target) < required_endpoint_distance
                for endpoint in endpoints
            ):
                continue
            if any(
                point_box_distance(position, center, static_size) < radius + dynamic_minimum_separation
                or point_box_distance(target, center, static_size) < radius + dynamic_minimum_separation
                for center, static_size in zip(static_position_array, chosen_static_sizes, strict=False)
            ):
                continue
            if any(
                float(np.linalg.norm(position - previous))
                < radius + 0.5 * max(chosen_dynamic_sizes[previous_index, :2]) + dynamic_minimum_separation
                for previous_index, previous in enumerate(dynamic_positions)
            ):
                continue
            dynamic_positions.append(position)
            dynamic_goals.append(target)
            break
        else:
            raise RuntimeError("Could not sample a separated dynamic-obstacle layout")

    return EpisodeSceneLayout(
        static_template_indices=static_indices,
        static_positions=static_position_array,
        dynamic_template_indices=dynamic_indices,
        dynamic_positions=np.asarray(dynamic_positions, dtype=np.float64).reshape(dynamic_count, 3),
        dynamic_goals=np.asarray(dynamic_goals, dtype=np.float64).reshape(dynamic_count, 3),
        dynamic_speeds=generator.uniform(speed_low, speed_high, size=dynamic_count),
    )


__all__ = [
    "DynamicObstacleSpec",
    "EpisodeSceneLayout",
    "StaticForestObstacle",
    "sample_episode_scene_layout",
    "sample_dynamic_obstacles",
    "sample_static_forest",
]
