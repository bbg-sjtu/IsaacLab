# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Articulation configuration for the Spirit AI Moz1 Pink-WBC model."""

from __future__ import annotations

import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

MOZ1_DIR = Path(__file__).resolve().parent
MOZ1_ROBOT_MODEL_DIR = MOZ1_DIR / "robot_model"
MOZ1_PACKAGE_DIR = MOZ1_ROBOT_MODEL_DIR / "spirit01_model"
MOZ1_URDF_PATH = MOZ1_PACKAGE_DIR / "urdf" / "moz1_wbc.urdf"
MOZ1_USD_PATH = MOZ1_PACKAGE_DIR / "usd" / "moz1_wbc" / "moz1_wbc.usda"

MOZ1_LEFT_EEF_LINK = "left07"
MOZ1_RIGHT_EEF_LINK = "right07"
MOZ1_BASE_REFERENCE_LINK = "world_base"

# This ordering is also the ordering of the posture suffix in every WBC action.
MOZ1_POSTURE_JOINT_NAMES = (
    "dummy_x_joint",
    "dummy_y_joint",
    "dummy_yaw_joint",
    "LegWaist_0",
    "LegWaist_1",
    "LegWaist_2",
    "LegWaist_3",
    "LegWaist_4",
    "LegWaist_5",
)

# User-specified reset pose [m or rad, depending on joint type].
# All values are within the limits published in the vendor URDF.
MOZ1_DEFAULT_JOINT_POSITIONS = {
    "dummy_x_joint": 0.0,
    "dummy_y_joint": 0.0,
    "dummy_yaw_joint": 0.0,
    "LegWaist_0": 0.0,
    "LegWaist_1": 1.0,
    "LegWaist_2": -1.5,
    "LegWaist_3": 0.5,
    "LegWaist_4": 0.0,
    "LegWaist_5": 0.0,
    "LeftArm_0": 0.0,
    "LeftArm_1": -0.4,
    "LeftArm_2": 0.0,
    "LeftArm_3": -1.5,
    "LeftArm_4": -1.2,
    "LeftArm_5": 0.0,
    "LeftArm_6": 0.0,
    "RightArm_0": 0.0,
    "RightArm_1": -0.4,
    "RightArm_2": 0.0,
    "RightArm_3": 1.5,
    "RightArm_4": 1.2,
    "RightArm_5": 0.0,
    "RightArm_6": 0.0,
}

MOZ1_DEFAULT_POSTURE = tuple(MOZ1_DEFAULT_JOINT_POSITIONS[name] for name in MOZ1_POSTURE_JOINT_NAMES)


MOZ1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(MOZ1_USD_PATH),
        activate_contact_sensors=False,
        # The base is commanded through virtual joints, without wheel contacts.
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # The public wheel mesh has a radius of approximately 0.105 m.
        pos=(0.0, 0.0, 0.105),
        joint_pos=MOZ1_DEFAULT_JOINT_POSITIONS,
        joint_vel={".*": 0.0},
    ),
    actuators={
        "virtual_base_xy": ImplicitActuatorCfg(
            joint_names_expr=["dummy_[xy]_joint"],
            velocity_limit_sim=1.0,
            effort_limit_sim=5000.0,
            stiffness=2500.0,
            damping=250.0,
        ),
        "virtual_base_yaw": ImplicitActuatorCfg(
            joint_names_expr=["dummy_yaw_joint"],
            velocity_limit_sim=1.5,
            effort_limit_sim=1000.0,
            stiffness=2500.0,
            damping=200.0,
        ),
        "leg_waist": ImplicitActuatorCfg(
            joint_names_expr=["LegWaist_.*"],
            velocity_limit_sim=math.radians(65.0),
            # The public package does not provide torque limits. This is a
            # simulation-only ceiling and must not be copied to hardware.
            effort_limit_sim=1000.0,
            stiffness=4000.0,
            damping=120.0,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=["LeftArm_.*", "RightArm_.*"],
            velocity_limit_sim={
                "(Left|Right)Arm_[01]": math.radians(210.0),
                "(Left|Right)Arm_[23]": math.radians(240.0),
                "(Left|Right)Arm_[4-6]": math.radians(300.0),
            },
            # The public package does not provide torque limits. This is a
            # simulation-only ceiling and must not be copied to hardware.
            effort_limit_sim=1000.0,
            stiffness=4000.0,
            damping=100.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Moz1 configured for kinematic whole-body tracking with a virtual planar base."""
