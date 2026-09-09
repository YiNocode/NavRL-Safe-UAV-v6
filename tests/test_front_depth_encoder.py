"""V6 finite-FOV static observation and encoder tests."""

import importlib.util
from pathlib import Path

import pytest
import torch

MODULE_PATH=(
    Path(__file__).resolve().parents[1]
    /"source"/"navrl_uav_v5"/"navrl_uav_v5"/"tasks"/"navrl_navigation"/"agents"
    /"front_depth_encoder.py"
)
SPEC=importlib.util.spec_from_file_location("front_depth_encoder",MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
FrontDepthEncoder=MODULE.FrontDepthEncoder


def test_front_depth_contract_and_embedding_shape():
    encoder=FrontDepthEncoder(embedding_dim=128,near_m=0.1,far_m=5.0)
    depth=torch.full((3,96,160,1),2.0)
    features=encoder.prepare_input(depth)
    embedding=encoder(depth)
    assert features.shape==(3,2,96,160)
    assert embedding.shape==(3,128)
    assert torch.isfinite(features).all() and torch.isfinite(embedding).all()


def test_invalid_depth_has_zero_proximity_and_invalid_mask():
    encoder=FrontDepthEncoder(near_m=0.1,far_m=5.0)
    depth=torch.tensor([[[[float("inf")],[float("nan")],[5.0],[0.1],[2.55]]]])
    features=encoder.prepare_input(depth)
    torch.testing.assert_close(features[0,0,0],torch.tensor([0.0,0.0,0.0,1.0,0.5]))
    torch.testing.assert_close(features[0,1,0],torch.tensor([0.0,0.0,1.0,1.0,1.0]))


def test_front_depth_encoder_rejects_v5_ray_matrix():
    encoder=FrontDepthEncoder()
    with pytest.raises(ValueError,match="B,H,W,1"):
        encoder(torch.ones((2,36,7)))
