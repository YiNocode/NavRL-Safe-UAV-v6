"""V6 M1 single-environment front depth camera task."""
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

@configclass
class V6FrontDepthCameraEnvCfg(DirectRLEnvCfg):
    seed=6; decimation=2; episode_length_s=2.0
    action_space=3; state_space=0
    camera_height=96; camera_width=160
    camera_near_m=0.10; camera_far_m=5.0
    camera_horizontal_fov_deg=90.0; stereo_baseline_m=0.10
    observation_space={"front_depth":[camera_height,camera_width,1]}
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
        if cfg.scene.num_envs != 1:
            raise ValueError("M1 requires exactly one environment")
        super().__init__(cfg,render_mode,**kwargs)
        self._zero_velocity=torch.zeros((1,6),device=self.device)
    def _setup_scene(self):
        self._drone=Articulation(self.cfg.drone)
        self._target=RigidObject(self.cfg.target)
        self._camera=TiledCamera(self.cfg.front_depth_camera)
        spawn_ground_plane("/World/ground",GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["drone"]=self._drone
        self.scene.rigid_objects["camera_target"]=self._target
        self.scene.sensors["front_depth_camera"]=self._camera
        light=sim_utils.DomeLightCfg(intensity=2000.0)
        light.func("/World/Light",light)
    def _pre_physics_step(self,actions):
        if actions.shape != (1,3): raise ValueError("actions must have shape (1,3)")
    def _apply_action(self):
        self._drone.write_root_velocity_to_sim(self._zero_velocity)
    def _get_observations(self):
        depth=self._camera.data.output["distance_to_image_plane"]
        expected=(1,self.cfg.camera_height,self.cfg.camera_width,1)
        if tuple(depth.shape) != expected: raise RuntimeError(f"unexpected depth shape {tuple(depth.shape)}")
        return {"front_depth":depth.clone()}
    def _get_rewards(self):
        return torch.zeros(1,device=self.device)
    def _get_dones(self):
        return torch.zeros(1,dtype=torch.bool,device=self.device), self.episode_length_buf>=self.max_episode_length-1
    def _reset_idx(self,env_ids:Sequence[int]|None):
        if env_ids is None: env_ids=self._drone._ALL_INDICES
        super()._reset_idx(env_ids)
        ids=torch.as_tensor(env_ids,dtype=torch.int64,device=self.device)
        state=self._drone.data.default_root_state[ids].clone()
        state[:,:3]+=self.scene.env_origins[ids]; state[:,7:]=0.0
        self._drone.write_root_pose_to_sim(state[:,:7],ids)
        self._drone.write_root_velocity_to_sim(state[:,7:],ids)
        self._camera.reset(ids)
    def get_camera_state(self):
        data=self._camera.data
        return {"source":"rtx_tiled_camera_distance_to_image_plane",
            "depth_m":data.output["distance_to_image_plane"].clone(),
            "intrinsics":data.intrinsic_matrices.clone(),"position_w":data.pos_w.clone(),
            "quaternion_w_world":data.quat_w_world.clone(),
            "quaternion_w_ros":data.quat_w_ros.clone(),
            "target_position_w":self._target.data.root_pos_w.clone()}

__all__=["V6FrontDepthCameraEnv","V6FrontDepthCameraEnvCfg"]
