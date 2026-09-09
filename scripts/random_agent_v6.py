"""V6 M1-M6 camera perception and structured-observation smoke."""
import argparse
import time
import traceback
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
    from navrl_uav_v5.tasks.navrl_navigation.agents.front_depth_encoder import FrontDepthEncoder
    from navrl_uav_v5.utils.gpu_camera_geometry import backproject_axial_depth, optical_points_to_world
    if args.num_envs < 1: raise ValueError("--num_envs must be positive")
    cfg=parse_env_cfg(TASK_ID,device=args.device,num_envs=args.num_envs)
    env=gym.make(TASK_ID,cfg=cfg)
    try:
        obs,_=env.reset()
        torch.cuda.reset_peak_memory_stats(env.unwrapped.device)
        torch.cuda.synchronize(env.unwrapped.device)
        start=time.perf_counter()
        max_tracked_speed=torch.zeros((),device=env.unwrapped.device)
        total_track_matches=0
        total_detections=0
        detection_frames=0
        min_association_distance=float("inf")
        max_measured_speed=0.0
        previous_truth_position=None
        max_truth_step=torch.zeros((),device=env.unwrapped.device)
        for _ in range(args.steps):
            obs,*_=env.step(torch.zeros((args.num_envs,3),device=env.unwrapped.device))
            tracked=env.unwrapped._dynamic_observation
            if tracked is not None and tracked.valid.any():
                speed=torch.linalg.vector_norm(tracked.velocities_w,dim=-1)
                max_tracked_speed=torch.maximum(max_tracked_speed,speed[tracked.valid].max())
            tracker=env.unwrapped._motion_tracker
            total_track_matches+=tracker.last_matched_count
            total_detections+=tracker.last_detection_count
            detection_frames+=int(tracker.last_detection_count>0)
            min_association_distance=min(
                min_association_distance,tracker.last_min_association_distance
            )
            max_measured_speed=max(max_measured_speed,tracker.last_max_measured_speed)
            truth_position=env.unwrapped._dynamic_target.data.root_pos_w.clone()
            if previous_truth_position is not None:
                truth_step=torch.linalg.vector_norm(truth_position-previous_truth_position,dim=-1).max()
                max_truth_step=torch.maximum(max_truth_step,truth_step)
            previous_truth_position=truth_position
        torch.cuda.synchronize(env.unwrapped.device)
        elapsed_s=time.perf_counter()-start
        state=env.unwrapped.get_camera_state(); depth=state["depth_m"]
        valid=torch.isfinite(depth)
        expected_depth=(args.num_envs,cfg.camera_height,cfg.camera_width,1)
        assert depth.shape==expected_depth and depth.dtype==torch.float32
        assert depth.device.type=="cuda" and valid.reshape(args.num_envs,-1).any(dim=1).all()
        assert state["intrinsics"].shape==(args.num_envs,3,3)
        assert state["position_w"].shape==(args.num_envs,3)
        assert state["quaternion_w_world"].shape==(args.num_envs,4)
        assert state["quaternion_w_ros"].shape==(args.num_envs,4)
        points_c=backproject_axial_depth(depth,state["intrinsics"])
        isaac_points_c=(
            unproject_depth(depth,state["intrinsics"],is_ortho=True)
            .reshape(args.num_envs,cfg.camera_width,cfg.camera_height,3)
            .transpose(1,2)
        )
        valid_xyz=valid.expand_as(points_c)
        torch.testing.assert_close(
            points_c[valid_xyz],isaac_points_c[valid_xyz],rtol=1e-5,atol=1e-5
        )
        points_w=optical_points_to_world(points_c,state["position_w"],state["quaternion_w_ros"])
        centers=points_w[:,cfg.camera_height//2,cfg.camera_width//2]
        target_front_x=state["target_position_w"][:,0]-0.25
        assert torch.isfinite(centers).all()
        assert torch.allclose(centers[:,0],target_front_x,atol=0.08,rtol=0.0), (
            f"max center x error={torch.max(torch.abs(centers[:,0]-target_front_x)).item():.4f}"
        )
        center_y_error=torch.abs(centers[:,1]-state["target_position_w"][:,1])
        assert torch.all(center_y_error<=0.05), (
            f"max center y error={center_y_error.max().item():.4f}"
        )
        assert torch.all(depth[valid]>=cfg.camera_near_m) and torch.all(depth[valid]<=cfg.camera_far_m)
        static_encoder=FrontDepthEncoder(
            embedding_dim=cfg.static_embedding_dim,
            near_m=cfg.camera_near_m,
            far_m=cfg.camera_far_m,
        ).to(env.unwrapped.device)
        with torch.inference_mode():
            encoder_input=static_encoder.prepare_input(depth)
            static_embedding=static_encoder(depth)
        assert encoder_input.shape==(args.num_envs,2,cfg.camera_height,cfg.camera_width)
        assert torch.isfinite(encoder_input).all()
        assert static_embedding.shape==(args.num_envs,128)
        assert torch.isfinite(static_embedding).all()
        assert obs["internal_state"].shape==(args.num_envs,8)
        assert torch.isfinite(obs["internal_state"]).all()
        dynamic=state["dynamic_observation"]
        assert dynamic.state.shape==(args.num_envs,5,10)
        assert torch.isfinite(dynamic.state).all()
        assert dynamic.valid.any(dim=1).all()
        assert dynamic.motion_mask.reshape(args.num_envs,-1).any(dim=1).all()
        truth=state["dynamic_target_position_w"]
        track_error=torch.linalg.vector_norm(dynamic.positions_w-truth[:,None,:],dim=-1)
        track_error=torch.where(dynamic.valid,track_error,torch.full_like(track_error,torch.inf))
        nearest_track_error=track_error.min(dim=1).values
        assert torch.all(nearest_track_error<=0.50), (
            f"max nearest dynamic track error={nearest_track_error.max().item():.4f}"
        )
        assert max_tracked_speed>0.01, (
            f"max tracked speed={max_tracked_speed.item():.4f}, matches={total_track_matches}, "
            f"detections={total_detections}/{detection_frames} frames, "
            f"min association={min_association_distance:.4f}, max measured speed={max_measured_speed:.4f}, "
            f"max truth step={max_truth_step.item():.4f}"
        )
        values=depth[valid]
        allocated_mib=torch.cuda.max_memory_allocated(env.unwrapped.device)/(1024**2)
        reserved_mib=torch.cuda.max_memory_reserved(env.unwrapped.device)/(1024**2)
        sim_steps_s=args.steps/elapsed_s
        env_frames_s=args.steps*args.num_envs/elapsed_s
        max_center_error=torch.max(torch.abs(centers[:,0]-target_front_x)).item()
        print(f"[PASS] V6 M6 envs={args.num_envs} depth_shape={tuple(depth.shape)} "
              f"encoder_input_shape={tuple(encoder_input.shape)} static_embedding_shape={tuple(static_embedding.shape)} "
              f"internal_shape={tuple(obs['internal_state'].shape)} "
              f"dynamic_shape={tuple(dynamic.state.shape)} dynamic_valid={int(dynamic.valid.sum())} "
              f"max_nearest_track_error_m={nearest_track_error.max().item():.4f} "
              f"max_tracked_speed_mps={max_tracked_speed.item():.4f} "
              f"dtype={depth.dtype} device={depth.device} "
              f"valid_pixels={int(valid.sum())} range_m=({values.min().item():.3f},{values.max().item():.3f}) "
              f"max_center_error_m={max_center_error:.5f} steps={args.steps} elapsed_s={elapsed_s:.3f} "
              f"sim_steps_s={sim_steps_s:.2f} env_frames_s={env_frames_s:.2f} "
              f"cuda_peak_allocated_mib={allocated_mib:.1f} cuda_peak_reserved_mib={reserved_mib:.1f} "
              "geometry=validated isaac_unproject_match=true ego_compensation=true "
              "track_lifecycle=true simulator_truth_policy_input=false finite_fov_not_v5_360=true", flush=True)
    finally: env.close()
if __name__=="__main__":
    try:
        main()
    except BaseException as error:
        print(f"[FAIL] V6 M6 {type(error).__name__}: {error}",flush=True)
        traceback.print_exc()
        raise
    finally: app.close()
