"""Deterministic tests for V6 finite-FOV temporal depth tracking."""

import inspect

import torch

from navrl_uav_v5.utils.gpu_camera_motion_tracker import GpuCameraMotionTracker


def _camera_inputs(batch: int, height: int, width: int):
    intrinsics=torch.tensor(
        [[[10.0,0.0,(width-1)/2],[0.0,10.0,(height-1)/2],[0.0,0.0,1.0]]]
    ).expand(batch,-1,-1).clone()
    position=torch.zeros((batch,3))
    quaternion=torch.tensor([[1.0,0.0,0.0,0.0]]).expand(batch,-1).clone()
    goal_frame=torch.tensor([[1.0,0.0,0.0]]).expand(batch,-1).clone()
    return intrinsics,position,quaternion,goal_frame


def test_temporal_residual_creates_track_and_preserves_10d_contract():
    tracker=GpuCameraMotionTracker(
        2,(5,5),max_tracks=5,motion_threshold_m=0.05,
        association_distance_m=1.0,velocity_smoothing=0.5,
        max_missed_frames=2,nms_kernel=3,device="cpu")
    intrinsics,position,quaternion,goal_frame=_camera_inputs(2,5,5)
    first=torch.full((2,5,5,1),4.0)
    initial=tracker.update(first,intrinsics,position,quaternion,position,goal_frame,0.1)
    assert not initial.valid.any()
    second=first.clone(); second[:,2,2,0]=2.0
    tracked=tracker.update(second,intrinsics,position,quaternion,position,goal_frame,0.1)
    assert tracked.state.shape==(2,5,10)
    assert tracked.valid[:,0].all() and tracked.motion_mask[:,2,2].all()
    assert torch.isfinite(tracked.state).all()


def test_ego_translation_is_compensated_before_residual():
    tracker=GpuCameraMotionTracker(
        1,(1,1),max_tracks=1,motion_threshold_m=0.01,
        association_distance_m=1.0,velocity_smoothing=0.5,
        max_missed_frames=1,nms_kernel=1,device="cpu")
    intrinsics,position,quaternion,goal_frame=_camera_inputs(1,1,1)
    tracker.update(torch.tensor([[[[2.0]]]]),intrinsics,position,quaternion,position,goal_frame,0.1)
    moved_position=position.clone(); moved_position[:,2]=0.1
    result=tracker.update(
        torch.tensor([[[[1.9]]]]),intrinsics,moved_position,quaternion,
        moved_position,goal_frame,0.1)
    assert not result.motion_mask.any()
    assert not result.valid.any()


def test_track_survives_short_occlusion_then_expires():
    tracker=GpuCameraMotionTracker(
        1,(3,3),max_tracks=1,motion_threshold_m=0.05,
        association_distance_m=1.0,velocity_smoothing=0.5,
        max_missed_frames=2,nms_kernel=1,device="cpu")
    intrinsics,position,quaternion,goal_frame=_camera_inputs(1,3,3)
    background=torch.full((1,3,3,1),4.0)
    tracker.update(background,intrinsics,position,quaternion,position,goal_frame,0.1)
    moving=background.clone(); moving[:,1,1,0]=2.0
    assert tracker.update(moving,intrinsics,position,quaternion,position,goal_frame,0.1).valid.any()
    assert tracker.update(moving,intrinsics,position,quaternion,position,goal_frame,0.1).valid.any()
    assert tracker.update(moving,intrinsics,position,quaternion,position,goal_frame,0.1).valid.any()
    expired=tracker.update(moving,intrinsics,position,quaternion,position,goal_frame,0.1)
    assert not expired.valid.any()
    assert torch.count_nonzero(expired.state)==0


def test_tracker_api_cannot_accept_simulator_obstacle_truth():
    parameters=set(inspect.signature(GpuCameraMotionTracker.update).parameters)
    forbidden={"obstacle_positions","obstacle_velocities","obstacle_sizes","semantic_labels"}
    assert parameters.isdisjoint(forbidden)
