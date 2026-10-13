# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import tempfile

import torch

import carb

import isaaclab.controllers.utils as ControllerUtils
import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.pink_ik import FrameTaskCfg, PinkIKControllerCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import ManusViveCfg, OpenXRDeviceCfg, XrCfg
from isaaclab.devices.openxr.retargeters.humanoid.fourier.gr1t2_retargeter import GR1T2RetargeterCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg_NOHAND
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils.configclass import configclass
from isaaclab.envs.common import ViewerCfg
from .tj_lr_cfg import TJ_LR_CFG
from . import mdp


##
# Scene definition
##
@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the GR1T2 Pick Place Base Scene."""

    robot: ArticulationCfg = TJ_LR_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0, 0.5),
            joint_pos={
                "Joint1_L":0.0,
                "Joint2_L":-1.5,
                "Joint3_L":1.57,
                "Joint4_L":-1.57,
                "Joint5_L":0.0,
                "Joint6_L":0.0,
                "Joint7_L":0.0,

                "Joint1_R":0.0,
                "Joint2_R":-1.5,
                "Joint3_R":-1.57,
                "Joint4_R":-1.57,
                "Joint5_R":0.0,
                "Joint6_R":0.0,
                "Joint7_R":0.0,
            },
        ),
    )

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )

    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    upper_body_ik = PinkInverseKinematicsActionCfg_NOHAND(
        enable_gravity_compensation=True,
        pink_controlled_joint_names=[
                "Joint1_L",
                "Joint2_L",
                "Joint3_L",
                "Joint4_L",
                "Joint5_L",
                "Joint6_L",
                "Joint7_L",

                "Joint1_R",
                "Joint2_R",
                "Joint3_R",
                "Joint4_R",
                "Joint5_R",
                "Joint6_R",
                "Joint7_R",
        ],
        hand_joint_names=[],
        target_eef_link_names={
            "left_eef": "ee_L",
            "right_eef": "ee_R",
        },
        # the robot in the sim scene we are controlling
        asset_name="robot",
        # Configuration for the IK controller
        # The frames names are the ones present in the URDF file
        # The urdf has to be generated from the USD that is being used in the scene
        controller=PinkIKControllerCfg(
            articulation_name="robot",
            base_link_name="Base",
            num_hand_joints=0,
            show_ik_warnings=False,
            # Determines whether Pink IK solver will fail due to a joint limit violation
            fail_on_joint_limit_violation=False,
            variable_input_tasks=[
                FrameTaskCfg(
                    frame="ee_L",
                    position_cost=8.0,  # [cost] / [m]
                    orientation_cost=1.0,  # [cost] / [rad]
                    lm_damping=0,  # dampening for solver for step jumps
                    gain=0.5,
                ),
                FrameTaskCfg(
                    frame="ee_R",
                    position_cost=8.0,  # [cost] / [m]
                    orientation_cost=1.0,  # [cost] / [rad]
                    lm_damping=0,  # dampening for solver for step jumps
                    gain=0.5,
                ),
                # DampingTask(
                #     cost=0.5,  # [cost] * [s] / [rad]
                # ),
                # NullSpacePostureTask(
                #     cost=0.5,
                #     lm_damping=1,
                #     controlled_frames=[
                #         "GR1T2_fourier_hand_6dof_left_hand_pitch_link",
                #         "GR1T2_fourier_hand_6dof_right_hand_pitch_link",
                #     ],
                #     controlled_joints=[
                #         "left_shoulder_pitch_joint",
                #         "left_shoulder_roll_joint",
                #         "left_shoulder_yaw_joint",
                #         "left_elbow_pitch_joint",
                #         "right_shoulder_pitch_joint",
                #         "right_shoulder_roll_joint",
                #         "right_shoulder_yaw_joint",
                #         "right_elbow_pitch_joint",
                #         "waist_yaw_joint",
                #         "waist_pitch_joint",
                #         "waist_roll_joint",
                #     ],
                # ),
            ],
            fixed_input_tasks=[],
            xr_enabled=bool(carb.settings.get_settings().get("/app/xr/enabled")),
        ),
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        states = ObsTerm(func=mdp.get_joint_pos)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


# @configclass
# class TerminationsCfg:
#     """Termination terms for the MDP."""

#     time_out = DoneTerm(func=mdp.time_out, time_out=True)

#     object_dropping = DoneTerm(
#         func=mdp.root_height_below_minimum, params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("object")}
#     )

#     success = DoneTerm(func=mdp.task_done_pick_place, params={"task_link_name": "right_hand_roll_link"})


# @configclass
# class EventCfg:
#     """Configuration for events."""

#     reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

#     reset_object = EventTerm(
#         func=mdp.reset_root_state_uniform,
#         mode="reset",
#         params={
#             "pose_range": {
#                 "x": [-0.01, 0.01],
#                 "y": [-0.01, 0.01],
#             },
#             "velocity_range": {},
#             "asset_cfg": SceneEntityCfg("object"),
#         },
#     )


@configclass
class TJ_LR_EnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the GR1T2 environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=1, env_spacing=10, replicate_physics=True)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    # terminations: TerminationsCfg = TerminationsCfg()
    # events = EventCfg()

    viewer: ViewerCfg = ViewerCfg(eye=(2,2,2),lookat=(0,0,0))

    # Temporary directory for URDF files
    temp_urdf_dir = tempfile.gettempdir()

    # Unused managers
    commands = None
    rewards = None
    curriculum = None

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2 # decimation=5的意思是，env层动作指令的频率是sim实际仿真频率的1/5，假设sim仿真步长是0.01s，那么env层动作指令的频率就是0.05s，
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 1 / 100  # 60Hz
        self.sim.render_interval = 2

        # Convert USD to URDF and change revolute joints to fixed
        temp_urdf_output_path, temp_urdf_meshes_output_path = ControllerUtils.convert_usd_to_urdf(
            self.scene.robot.spawn.usd_path, self.temp_urdf_dir, force_conversion=True
        )

        # Set the URDF and mesh paths for the IK controller
        self.actions.upper_body_ik.controller.urdf_path = temp_urdf_output_path
        self.actions.upper_body_ik.controller.mesh_path = temp_urdf_meshes_output_path
