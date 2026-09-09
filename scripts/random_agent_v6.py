"""V6 M1/M2 front-depth acquisition and geometry smoke."""
import argparse
from isaaclab.app import AppLauncher
TASK_ID="Isaac-UAV-NavRL-V6-Front-Depth-M1-v0"
parser=argparse.ArgumentParser()
parser.add_argument("--num_envs",type=int,default=1)
parser.add_argument("--steps",type=int,default=20)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
app=AppLauncher(args).app
def main():
    import gymnasium as gym
    import torch
    import navrl_uav_v5
    from isaaclab.utils.math import unproject_depth
    from isaaclab_tasks.utils import parse_env_cfg
    from navrl_uav_v5.utils.gpu_camera_geometry import backproject_axial_depth, optical_points_to_world
    if args.num_envs != 1: raise ValueError("M1 requires --num_envs 1")
    cfg=parse_env_cfg(TASK_ID,device=args.device,num_envs=1)
    env=gym.make(TASK_ID,cfg=cfg)
    try:
        obs,_=env.reset()
        for _ in range(args.steps):
            obs,*_=env.step(torch.zeros((1,3),device=env.unwrapped.device))
        state=env.unwrapped.get_camera_state(); depth=state["depth_m"]
        valid=torch.isfinite(depth)
        assert depth.shape==(1,96,160,1) and depth.dtype==torch.float32
        assert depth.device.type=="cuda" and valid.any()
        assert state["intrinsics"].shape==(1,3,3)
        assert state["position_w"].shape==(1,3)
        assert state["quaternion_w_world"].shape==(1,4)
        points_c=backproject_axial_depth(depth,state["intrinsics"])
        isaac_points_c=(
            unproject_depth(depth,state["intrinsics"],is_ortho=True)
            .reshape(1,cfg.camera_width,cfg.camera_height,3)
            .transpose(1,2)
        )
        valid_xyz=valid.expand_as(points_c)
        torch.testing.assert_close(
            points_c[valid_xyz],isaac_points_c[valid_xyz],rtol=1e-5,atol=1e-5
        )
        points_w=optical_points_to_world(points_c,state["position_w"],state["quaternion_w_ros"])
        center=points_w[0,cfg.camera_height//2,cfg.camera_width//2]
        target_front_x=state["target_position_w"][0,0]-0.25
        assert torch.isfinite(center).all()
        assert torch.isclose(center[0],target_front_x,atol=0.08,rtol=0.0), (
            f"center x={center[0].item():.4f}, expected target front x={target_front_x.item():.4f}"
        )
        assert torch.abs(center[1]-state["target_position_w"][0,1])<=0.05, (
            f"center y={center[1].item():.4f}, target y={state['target_position_w'][0,1].item():.4f}"
        )
        assert torch.all(depth[valid]>=cfg.camera_near_m) and torch.all(depth[valid]<=cfg.camera_far_m)
        values=depth[valid]
        print(f"[PASS] V6 M1 shape={tuple(depth.shape)} dtype={depth.dtype} device={depth.device} "
              f"valid_pixels={int(valid.sum())} range_m=({values.min().item():.3f},{values.max().item():.3f}) "
              f"center_hit_w=({center[0].item():.3f},{center[1].item():.3f},{center[2].item():.3f}) "
              "intrinsics=(1,3,3) pose=(1,3)+(1,4) geometry=validated "
              "isaac_unproject_match=true finite_fov_not_v5_360=true", flush=True)
    finally: env.close()
if __name__=="__main__":
    try: main()
    finally: app.close()
