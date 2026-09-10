"""V5 NavRL policy definitions and RSL-RL class registration."""

from .navrl_actor_critic import NavRLActorCritic, register_navrl_rsl_rl_class
from .navrl_ppo import NavRLPPO, register_navrl_ppo_class
from .front_depth_encoder import FrontDepthEncoder
from .static_obstacle_encoder import StaticObstacleEncoder
from .v6_actor_critic import V6NavRLActorCritic, register_v6_navrl_rsl_rl_class
from .v6_voxel_actor_critic import V6VoxelNavRLActorCritic, register_v6_voxel_navrl_rsl_rl_class

register_navrl_rsl_rl_class()
register_navrl_ppo_class()
register_v6_navrl_rsl_rl_class()
register_v6_voxel_navrl_rsl_rl_class()

__all__ = [
    "NavRLActorCritic",
    "NavRLPPO",
    "FrontDepthEncoder",
    "StaticObstacleEncoder",
    "V6NavRLActorCritic",
    "register_navrl_ppo_class",
    "register_navrl_rsl_rl_class",
    "register_v6_navrl_rsl_rl_class",
    "V6VoxelNavRLActorCritic",
    "register_v6_voxel_navrl_rsl_rl_class",
]
