"""V5 NavRL policy definitions and RSL-RL class registration."""

from .navrl_actor_critic import NavRLActorCritic, register_navrl_rsl_rl_class
from .navrl_ppo import NavRLPPO, register_navrl_ppo_class
from .front_depth_encoder import FrontDepthEncoder
from .static_obstacle_encoder import StaticObstacleEncoder

register_navrl_rsl_rl_class()
register_navrl_ppo_class()

__all__ = [
    "NavRLActorCritic",
    "NavRLPPO",
    "FrontDepthEncoder",
    "StaticObstacleEncoder",
    "register_navrl_ppo_class",
    "register_navrl_rsl_rl_class",
]
