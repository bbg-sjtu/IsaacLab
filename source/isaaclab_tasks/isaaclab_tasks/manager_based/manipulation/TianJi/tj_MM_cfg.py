import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Configuration
##

TJ_MM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=(
            "/home/bbgsjtu/Research/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/TianJi/robot_model/MM.usd"
            # f"{ISAAC_NUCLEUS_DIR}/Robots/Idealworks/iw_hub.usd"
        ),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        # enabled_self_collisions一定要False！否则关节h控制总是有错误！！！机械模型的装配感觉有问题
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        "left_arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "Joint.*L",
            ],
            velocity_limit=10.0,
            effort_limit=100,
            stiffness=5000,
            damping=100,
        ),
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "Joint.*R",
            ],
            velocity_limit=10.0,
            effort_limit=100,
            stiffness=5000,
            damping=100,
        ),
        "torso": ImplicitActuatorCfg(
            joint_names_expr=[
                "joint.*",
            ],
            velocity_limit=10.0,
            effort_limit=1000,
            stiffness=5000,
            damping=200,
        ),
        "agv": ImplicitActuatorCfg(
            joint_names_expr=[
                "dummy_.*",
            ],
            velocity_limit=10.0,
            effort_limit=1000,
            stiffness=5000,
            damping=200,
        ),
    },
)
"""Configuration for the TianJi+AGV robot."""