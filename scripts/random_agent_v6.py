"""V6 M1 front depth camera smoke."""
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
    from isaaclab_tasks.utils import parse_env_cfg
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
        assert torch.all(depth[valid]>=cfg.camera_near_m) and torch.all(depth[valid]<=cfg.camera_far_m)
        values=depth[valid]
        print(f"[PASS] V6 M1 shape={tuple(depth.shape)} dtype={depth.dtype} device={depth.device} "
              f"valid_pixels={int(valid.sum())} range_m=({values.min().item():.3f},{values.max().item():.3f}) "
              "intrinsics=(1,3,3) pose=(1,3)+(1,4) finite_fov_not_v5_360=true", flush=True)
    finally: env.close()
if __name__=="__main__":
    try: main()
    finally: app.close()
