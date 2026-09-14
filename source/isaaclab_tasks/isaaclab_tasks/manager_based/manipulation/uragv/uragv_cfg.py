import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Configuration
##

URAGV_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=(
            "/WorkSpace/LiuBang/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/uragv/URAGV.usd"
            # f"{ISAAC_NUCLEUS_DIR}/Robots/Idealworks/iw_hub.usd"
        ),
        # activate_contact_sensors=True,
        # rigid_props=sim_utils.RigidBlseodyPropertiesCfg(
        #     disable_gravity=False,
        #     retain_accelerations=False,
        #     linear_damping=0.0,
        #     angular_damping=0.0,
        #     max_linear_velocity=1000.0,
        #     max_angular_velocity=1000.0,
        #     max_depenetration_velocity=1.0,
        # ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        "agv": ImplicitActuatorCfg(
            joint_names_expr=[
                "dummy_.*",
            ],
            effort_limit=200,
            velocity_limit=80,
            stiffness=1000,
            damping=800,
        ),
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "ur_arm_.*",
            ],
            velocity_limit=100.0,
            effort_limit=100,
            stiffness=1000,
            damping=100,
        ),
        "gripper":ImplicitActuatorCfg(
            joint_names_expr=[
                "Slider_.*",
            ],
            velocity_limit=100,
            effort_limit=100,
            stiffness=1000,
            damping=100,
        ),
    },
)
"""Configuration for the UR+AGV robot."""