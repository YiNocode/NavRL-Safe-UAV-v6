"""Crazyflie asset configuration for high-level velocity-control experiments."""

from isaaclab_assets import CRAZYFLIE_CFG

DRONE_RADIUS = 0.30
"""Conservative navigation radius in metres; collision logic is added later."""

DRONE_CFG = CRAZYFLIE_CFG.copy()
DRONE_CFG.prim_path = "/World/envs/env_.*/Drone"
# Match the task's near-ground reset.  The environment overwrites XY and Z at
# every episode reset, but this avoids a one-frame airborne default pose while
# Isaac initializes the articulation.
DRONE_CFG.init_state.pos = (0.0, 0.0, 0.10)

# Prompt 3 validates a high-level velocity command, not motor-level flight
# dynamics. Gravity is disabled so a zero velocity command is a stable hover.
DRONE_CFG.spawn.rigid_props.disable_gravity = True
DRONE_CFG.spawn.activate_contact_sensors = True
"""Official Isaac Lab Crazyflie configured for kinematic-like velocity control."""
