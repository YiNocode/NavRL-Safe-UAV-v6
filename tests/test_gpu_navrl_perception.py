"""Deterministic tensor tests for V5's sensor-only GPU perception modules."""

from __future__ import annotations

import inspect

import torch
from navrl_uav_v5.utils.gpu_navrl_perception import (
    BatchedOccupancyVoxelMap,
    GpuDepthMotionTracker,
    make_body_ray_directions,
)


def test_body_ray_direction_matrix_convention() -> None:
    directions = make_body_ray_directions(360.0, 90.0, 60.0, 30.0)
    assert directions.shape == (4, 3, 3)
    torch.testing.assert_close(directions[0, 1], torch.tensor([1.0, 0.0, 0.0]), atol=1.0e-6, rtol=0.0)
    assert directions[0, 2, 2] > 0.0
    assert directions[0, 0, 2] < 0.0


def test_batched_depth_hits_are_voxelized_then_raycast() -> None:
    voxel_map = BatchedOccupancyVoxelMap(
        2,
        voxel_size=0.25,
        map_size=(10.0, 10.0, 4.0),
        origin_local=(-5.0, -5.0, 0.0),
        device="cpu",
    )
    hits = torch.tensor([[[3.0, 0.0, 1.0]], [[0.0, 3.0, 1.0]]])
    origins = torch.zeros((2, 3))
    voxel_map.integrate_hits(hits, origins, torch.ones((2, 1), dtype=torch.bool))
    directions = make_body_ray_directions(360.0, 90.0, 0.0, 10.0)
    raw, normalized, hit = voxel_map.ray_cast(
        torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        origins,
        torch.zeros(2),
        directions,
        max_distance=4.0,
        ray_step_size=0.25,
        no_hit_offset=0.1,
    )
    assert raw.shape == (2, 4, 1)
    assert hit[0, 0, 0]
    assert hit[1, 1, 0]
    assert abs(raw[0, 0, 0].item() - 3.0) <= 0.25
    assert abs(raw[0, 1, 0].item() - 4.1) <= 1.0e-6
    assert normalized[0, 1, 0].item() == 1.0


def test_uav_heading_rotates_virtual_rays() -> None:
    voxel_map = BatchedOccupancyVoxelMap(
        1,
        voxel_size=0.25,
        map_size=(10.0, 10.0, 4.0),
        origin_local=(-5.0, -5.0, 0.0),
        device="cpu",
    )
    voxel_map.integrate_hits(
        torch.tensor([[[0.0, 2.0, 1.0]]]),
        torch.zeros((1, 3)),
        torch.ones((1, 1), dtype=torch.bool),
    )
    directions = make_body_ray_directions(360.0, 90.0, 0.0, 10.0)
    raw, _, hit = voxel_map.ray_cast(
        torch.tensor([[0.0, 0.0, 1.0]]),
        torch.zeros((1, 3)),
        torch.tensor([torch.pi / 2.0]),
        directions,
        max_distance=4.0,
        ray_step_size=0.25,
        no_hit_offset=0.1,
    )
    assert hit[0, 0, 0]
    assert abs(raw[0, 0, 0].item() - 2.0) <= 0.25


def test_temporal_depth_motion_produces_gpu_track() -> None:
    tracker = GpuDepthMotionTracker(
        1,
        (4, 1),
        1,
        max_distance=5.0,
        motion_threshold=0.05,
        association_distance=1.5,
        velocity_smoothing=0.5,
        horizontal_angle_step=90.0,
        device="cpu",
    )
    directions = make_body_ray_directions(360.0, 90.0, 0.0, 10.0)
    # Reorder [Nh,Nv] to Isaac's flattened [Nv,Nh].
    sensor_directions = directions.transpose(0, 1).reshape(1, 4, 3)
    origin = torch.zeros((1, 3))
    depth_1 = torch.full((1, 4, 1), 5.0)
    depth_1[0, 0, 0] = 3.0
    hits_1 = sensor_directions * depth_1.transpose(1, 2).reshape(1, 4, 1)
    tracker.update(depth_1, hits_1, sensor_directions, origin, 0.1)
    depth_2 = depth_1.clone()
    depth_2[0, 0, 0] = 2.7
    hits_2 = sensor_directions * depth_2.transpose(1, 2).reshape(1, 4, 1)
    positions, velocities, sizes, valid, motion = tracker.update(
        depth_2, hits_2, sensor_directions, origin, 0.1
    )
    assert motion[0, 0, 0]
    assert valid[0, 0]
    assert torch.isfinite(positions).all()
    assert torch.isfinite(velocities).all()
    assert torch.all(sizes[valid] > 0.0)


def test_perception_api_cannot_accept_simulator_obstacle_truth() -> None:
    parameters = set(inspect.signature(GpuDepthMotionTracker.update).parameters)
    forbidden = {
        "obstacle_positions",
        "obstacle_velocities",
        "obstacle_sizes",
        "mesh_ids",
        "semantic_labels",
    }
    assert parameters.isdisjoint(forbidden)
