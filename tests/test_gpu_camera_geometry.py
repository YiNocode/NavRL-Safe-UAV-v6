"""Deterministic tensor tests for V6 camera geometry."""
import torch

from navrl_uav_v5.utils.gpu_camera_geometry import backproject_axial_depth, optical_points_to_world


def test_axial_depth_backprojection_pixel_geometry():
    depth=torch.full((1,3,5,1),2.0)
    k=torch.tensor([[[2.0,0.0,2.0],[0.0,2.0,1.0],[0.0,0.0,1.0]]])
    points=backproject_axial_depth(depth,k)
    torch.testing.assert_close(points[0,1,2],torch.tensor([0.0,0.0,2.0]))
    torch.testing.assert_close(points[0,0,0],torch.tensor([-2.0,-1.0,2.0]))


def test_optical_to_world_identity_and_translation():
    points=torch.tensor([[[[1.0,2.0,3.0]]]])
    position=torch.tensor([[4.0,5.0,6.0]])
    quaternion=torch.tensor([[1.0,0.0,0.0,0.0]])
    world=optical_points_to_world(points,position,quaternion)
    torch.testing.assert_close(world,torch.tensor([[[[5.0,7.0,9.0]]]]))


def test_geometry_rejects_mismatched_batches():
    depth=torch.ones((2,3,4,1))
    try:
        backproject_axial_depth(depth,torch.eye(3).unsqueeze(0))
    except ValueError:
        pass
    else:
        raise AssertionError("batch mismatch was not rejected")
