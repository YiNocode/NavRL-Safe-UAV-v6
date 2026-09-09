"""V6 front depth camera task with batched GPU tensors."""
from __future__ import annotations
from collections.abc import Sequence
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from navrl_uav_v5.assets import DRONE_CFG
from navrl_uav_v5.utils.gpu_camera_motion_tracker import GpuCameraMotionTracker

@configclass
class V6FrontDepthCameraEnvCfg(DirectRLEnvCfg):
    seed=6; decimation=2; episode_length_s=2.0
    action_space=3; state_space=0
    camera_height=96; camera_width=160
    camera_near_m=0.10; camera_far_m=5.0
    camera_horizontal_fov_deg=90.0; stereo_baseline_m=0.10
    static_embedding_dim=128
    dynamic_state_dim=10; max_dynamic_tracks=5
    motion_threshold_m=0.015; association_distance_m=0.75
    velocity_smoothing=0.70; max_missed_frames=10; motion_nms_kernel=9
    # Policy-side V6 static contract: channel-last axial depth in metres.
    # Invalid/no-return pixels remain non-finite and are masked by the encoder.
    observation_space={
        "front_depth":[camera_height,camera_width,1],
        "dynamic_obstacles":[max_dynamic_tracks,dynamic_state_dim],
    }
    sim=SimulationCfg(dt=0.01, render_interval=decimation)
    scene=InteractiveSceneCfg(num_envs=1, env_spacing=8.0, replicate_physics=True)
    drone=DRONE_CFG
    target=RigidObjectCfg(
        prim_path="/World/envs/env_.*/CameraTarget",
        spawn=sim_utils.CuboidCfg(
            size=(0.5,1.0,1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True,disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85,0.15,0.05))),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0,0.0,0.60)))
    dynamic_target=RigidObjectCfg(
        prim_path="/World/envs/env_.*/DynamicCameraTarget",
        spawn=sim_utils.CuboidCfg(
            size=(0.30,0.30,0.50),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True,disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10,0.25,0.90))),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.50,0.85,0.55)))
    front_depth_camera=TiledCameraCfg(
        prim_path="/World/envs/env_.*/Drone/body/front_depth_camera",
        update_period=0.0, height=camera_height, width=camera_width,
        data_types=["distance_to_image_plane"], depth_clipping_behavior="none",
        update_latest_camera_pose=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0, focus_distance=5.0, horizontal_aperture=24.0,
            clipping_range=(camera_near_m,camera_far_m)),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.05,0.0,0.05), rot=(1.0,0.0,0.0,0.0), convention="world"))

class V6FrontDepthCameraEnv(DirectRLEnv):
    cfg: V6FrontDepthCameraEnvCfg
    def __init__(self,cfg,render_mode=None,**kwargs):
        super().__init__(cfg,render_mode,**kwargs)
        self._zero_velocity=torch.zeros((self.num_envs,6),device=self.device)
        self._start_to_goal_w=torch.zeros((self.num_envs,3),device=self.device)
        self._start_to_goal_w[:,0]=1.0
        self._motion_tracker=GpuCameraMotionTracker(
            self.num_envs,(cfg.camera_height,cfg.camera_width),cfg.max_dynamic_tracks,
            near_m=cfg.camera_near_m,far_m=cfg.camera_far_m,
            motion_threshold_m=cfg.motion_threshold_m,
            association_distance_m=cfg.association_distance_m,
            velocity_smoothing=cfg.velocity_smoothing,
            max_missed_frames=cfg.max_missed_frames,nms_kernel=cfg.motion_nms_kernel,
            device=self.device)
        self._dynamic_observation=None
    def _setup_scene(self):
        self._drone=Articulation(self.cfg.drone)
        self._target=RigidObject(self.cfg.target)
        self._dynamic_target=RigidObject(self.cfg.dynamic_target)
        self._camera=TiledCamera(self.cfg.front_depth_camera)
        spawn_ground_plane("/World/ground",GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["drone"]=self._drone
        self.scene.rigid_objects["camera_target"]=self._target
        self.scene.rigid_objects["dynamic_camera_target"]=self._dynamic_target
        self.scene.sensors["front_depth_camera"]=self._camera
        light=sim_utils.DomeLightCfg(intensity=2000.0)
        light.func("/World/Light",light)
    def _pre_physics_step(self,actions):
        if actions.shape != (self.num_envs,3):
            raise ValueError(f"actions must have shape ({self.num_envs},3)")
        phase=self.episode_length_buf.to(torch.float32)*self.step_dt*4.0
        state=self._dynamic_target.data.default_root_state.clone()
        state[:,:3]+=self.scene.env_origins
        state[:,0]+=0.35*torch.sin(phase)
        state[:,1]+=0.15*torch.cos(phase)
        state[:,7:]=0.0
        self._dynamic_target.write_root_pose_to_sim(state[:,:7])
        self._dynamic_target.write_root_velocity_to_sim(state[:,7:])
    def _apply_action(self):
        self._drone.write_root_velocity_to_sim(self._zero_velocity)
    def _get_observations(self):
        depth=self._camera.data.output["distance_to_image_plane"]
        expected=(self.num_envs,self.cfg.camera_height,self.cfg.camera_width,1)
        if tuple(depth.shape) != expected: raise RuntimeError(f"unexpected depth shape {tuple(depth.shape)}")
        data=self._camera.data
        self._dynamic_observation=self._motion_tracker.update(
            depth,data.intrinsic_matrices,data.pos_w,data.quat_w_ros,
            self._drone.data.root_pos_w,self._start_to_goal_w,self.step_dt)
        return {
            "front_depth":depth.clone(),
            "dynamic_obstacles":self._dynamic_observation.state,
        }
    def _get_rewards(self):
        return torch.zeros(self.num_envs,device=self.device)
    def _get_dones(self):
        return torch.zeros(self.num_envs,dtype=torch.bool,device=self.device), self.episode_length_buf>=self.max_episode_length-1
    def _reset_idx(self,env_ids:Sequence[int]|None):
        if env_ids is None: env_ids=self._drone._ALL_INDICES
        super()._reset_idx(env_ids)
        ids=torch.as_tensor(env_ids,dtype=torch.int64,device=self.device)
        state=self._drone.data.default_root_state[ids].clone()
        state[:,:3]+=self.scene.env_origins[ids]; state[:,7:]=0.0
        self._drone.write_root_pose_to_sim(state[:,:7],ids)
        self._drone.write_root_velocity_to_sim(state[:,7:],ids)
        dynamic_state=self._dynamic_target.data.default_root_state[ids].clone()
        dynamic_state[:,:3]+=self.scene.env_origins[ids]; dynamic_state[:,7:]=0.0
        self._dynamic_target.write_root_pose_to_sim(dynamic_state[:,:7],ids)
        self._dynamic_target.write_root_velocity_to_sim(dynamic_state[:,7:],ids)
        self._camera.reset(ids)
        self._motion_tracker.reset(ids)
    def get_camera_state(self):
        data=self._camera.data
        return {"source":"rtx_tiled_camera_distance_to_image_plane",
            "depth_m":data.output["distance_to_image_plane"].clone(),
            "intrinsics":data.intrinsic_matrices.clone(),"position_w":data.pos_w.clone(),
            "quaternion_w_world":data.quat_w_world.clone(),
            "quaternion_w_ros":data.quat_w_ros.clone(),
            "target_position_w":self._target.data.root_pos_w.clone(),
            "dynamic_target_position_w":self._dynamic_target.data.root_pos_w.clone(),
            "dynamic_observation":self._dynamic_observation}

__all__=["V6FrontDepthCameraEnv","V6FrontDepthCameraEnvCfg"]
