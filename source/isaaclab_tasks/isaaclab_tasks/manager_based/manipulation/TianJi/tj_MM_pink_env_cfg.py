# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import tempfile
import torch

from pink.tasks import FrameTask,DampingTask

import isaaclab.controllers.utils as ControllerUtils
import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg_NullSpacePostureControl
from isaaclab.controllers.pink_ik import NullSpacePostureTask, PinkIKControllerCfg,JointCouplingTask
from . import mdp


from .tj_MM_cfg import TJ_MM_CFG

from isaaclab.sensors import CameraCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.envs.common import ViewerCfg
##
# Scene definition
##
@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):

    # Humanoid robot w/ arms higher
    robot: ArticulationCfg = TJ_MM_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0, 0),
            joint_pos={
                "Joint1_L":0.0,
                "Joint2_L":-1.5,
                "Joint3_L":1.5,
                "Joint4_L":-1.5,
                "Joint5_L":0.0,
                "Joint6_L":0.0,
                "Joint7_L":0.0,

                "Joint1_R":0.0,
                "Joint2_R":-1.5,
                "Joint3_R":-1.5,
                "Joint4_R":-1.5,
                "Joint5_R":0.0,
                "Joint6_R":0.0,
                "Joint7_R":0.0,

                "joint1":0.0,
                "joint2":-0.7,
                "joint3":1.2,
                "joint4":-0.5,

                "dummy_x_joint":0.0,
                "dummy_y_joint":0.0,
            },
        ),
    )

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(
            color=(0, 0, 0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, -0.025),
        ),
    )

    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75,0.75,0.75), intensity=500.0),
    )

    # camera_top = CameraCfg(
    #     prim_path="/World/envs/env_.*/PackingTable/camera_top",
    #     update_period=0.1,
    #     height=480,
    #     width=640,
    #     data_types=["rgb"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=11.9, focus_distance=400.0, horizontal_aperture=20.955,vertical_aperture=15.2908, clipping_range=(0.1, 1.0e5)
    #     ),
    #     offset=CameraCfg.OffsetCfg(pos=(-0.568, -0.852, 3.546), rot=(0,-1,0,0), convention="ros"),
    # )
    # camera_1 = CameraCfg(
    #     prim_path="/World/envs/env_.*/PackingTable/camera_1",
    #     update_period=0.1,
    #     height=480,
    #     width=640,
    #     data_types=["rgb"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=18.148, focus_distance=400.0, horizontal_aperture=20.955,vertical_aperture=15.2908, clipping_range=(0.1, 1.0e5)
    #     ),
    #     offset=CameraCfg.OffsetCfg(pos=(-3.337, -1.428, 2.278), rot=(0.36652,-0.6047,0.6047,-0.36652), convention="ros"),
    #     # 原：0.6047,0.36652,-0.36652,-0.6047，需要变换一下次序，奇怪的bug
    #     # (x1,x2,x3,x4) -> (x2,-x1,x4,-x3)
    # )
    # camera_2 = CameraCfg(
    #     prim_path="/World/envs/env_.*/PackingTable/camera_2",
    #     update_period=0.1,
    #     height=480,
    #     width=640,
    #     data_types=["rgb"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=18.148, focus_distance=400.0, horizontal_aperture=20.955,vertical_aperture=15.2908, clipping_range=(0.1, 1.0e5)
    #     ),
    #     offset=CameraCfg.OffsetCfg(pos=(-0.082,1.955,2.277), rot=(0.0, 0.0, -0.85408,0.52015), convention="ros"),
    # )
    # camera_wrist = CameraCfg(
    #     prim_path="/World/envs/env_.*/Robot/gripper1/RobotIq_Hand_E_base/ee_link/camera_wrist",
    #     update_period=0.1,
    #     height=480,
    #     width=640,
    #     data_types=["rgb"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=14.5, focus_distance=400.0, horizontal_aperture=20.955,vertical_aperture=15.2908, clipping_range=(0.1, 1.0e5)
    #     ),
    #     offset=CameraCfg.OffsetCfg(pos=(-0.13688,0.0,0.09234), rot=(-0.40343,0.58072,-0.58072,0.40343), convention="ros"),
    # )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    upper_body_ik = PinkInverseKinematicsActionCfg_NullSpacePostureControl(
        enable_gravity_compensation=True,
        pink_controlled_joint_names=[
                "Joint.*",
                "joint.*",
                "dummy.*",
        ],
        hand_joint_names=[],
        target_eef_link_names={
            "left_eef": "ee_L",
            "right_eef": "ee_R",
        },
        # the robot in the sim scene we are controlling‘’
        asset_name="robot",
        # Configuration for the IK controller
        # The frames names are the ones present in the URDF file
        # The urdf has to be generated from the USD that is being used in the scene
        controller=PinkIKControllerCfg(
            articulation_name="robot",
            base_link_name="world_base",
            num_posture_controlled_joints=6,
            show_ik_warnings=False,
            # Determines whether Pink IK solver will fail due to a joint limit violation
            fail_on_joint_limit_violation=False,
            variable_input_tasks=[
                FrameTask(
                    "ee_L",
                    position_cost=10.0,  # [cost] / [m]
                    orientation_cost=1.0,  # [cost] / [rad]
                    lm_damping=0,  # dampening for solver for step jumps
                    gain=0.5,
                ),
                FrameTask(
                    "ee_R",
                    position_cost=10.0,  # [cost] / [m]
                    orientation_cost=1.0,  # [cost] / [rad]
                    lm_damping=0,  # dampening for solver for step jumps
                    gain=0.5,
                ),
                DampingTask(
                    cost=0.5,  # [cost] * [s] / [rad]
                ),
                NullSpacePostureTask(
                    cost=0.5,
                    lm_damping=1,
                    gain=0.5,
                    controlled_frames=[
                        "ee_L",
                        "ee_R",
                    ],
                    controlled_joints=[
                        "joint1",
                        "joint2",
                        "joint3",
                        "joint4",
                        "dummy_x_joint",
                        "dummy_y_joint",
                    ],
                ),
                JointCouplingTask(
                    joint1="joint2",
                    joint2="joint3",
                    joint3="joint4",
                    cost=0.1,
                    gain=0.5,
                    lm_damping=1.0,
                )
            ],
            fixed_input_tasks=[],
        ),
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        
        # robot_joint_pos = ObsTerm(
        #     func=base_mdp.joint_pos,
        #     params={"asset_cfg": SceneEntityCfg("robot")},
        # )

        # robot_root_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("robot")})
        # robot_root_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("robot")})
        # object_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("object")})
        # object_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("object")})
        # robot_links_state = ObsTerm(func=mdp.get_all_robot_link_state)

        # left_eef_pos = ObsTerm(func=mdp.get_left_eef_pos)
        # left_eef_quat = ObsTerm(func=mdp.get_left_eef_quat)
        # right_eef_pos = ObsTerm(func=mdp.get_right_eef_pos)
        # right_eef_quat = ObsTerm(func=mdp.get_right_eef_quat)

        # hand_joint_state = ObsTerm(func=mdp.get_hand_state)
        # head_joint_state = ObsTerm(func=mdp.get_head_state)

        # object = ObsTerm(func=mdp.object_obs)

        # top_img=ObsTerm(func=mdp.get_top_cam)
        # table1_img=ObsTerm(func=mdp.get_table1_cam)
        # table2_img=ObsTerm(func=mdp.get_table2_cam)
        # # wrist_img=ObsTerm(func=mdp.get_wrist_cam)

        # actions = ObsTerm(func=mdp.last_action)
        states = ObsTerm(func=mdp.get_joint_pos)

        # right_hand_open_state=ObsTerm(func=mdp.get_right_hand_open_state)


        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


# @configclass
# class TerminationsCfg:
#     """Termination terms for the MDP."""

#     time_out = DoneTerm(func=mdp.time_out, time_out=True)

#     # object_dropping = DoneTerm(
#     #     func=mdp.root_height_below_minimum, params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("object")}
#     # )

#     success = DoneTerm(func=mdp.task_done)


# @configclass
# class EventCfg:
#     """Configuration for events."""

#     reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

#     reset_object = EventTerm(
#         func=mdp.reset_root_state_uniform,
#         mode="reset",
#         params={
#             "pose_range": {
#                 "x": [0, 0.15],#增量
#                 "y": [0, 0.4],
#             },
#             "velocity_range":{},
#             "asset_cfg": SceneEntityCfg("box"),
#         },
#     )

#     reset_container = EventTerm(
#         func=mdp.reset_root_state_uniform,
#         mode="reset",
#         params={
#             "pose_range": {
#                 "x": [0, 1.6],
#             },
#             "velocity_range":{},
#             "asset_cfg": SceneEntityCfg("container"),
#         },
#     )
    


@configclass
class TJ_MM_PINK_EnvCfg(ManagerBasedRLEnvCfg):
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
        self.decimation = 5 # decimation=5的意思是，env层动作指令的频率是sim实际仿真频率的1/5，假设sim仿真步长是0.01s，那么env层动作指令的频率就是0.05s，
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 1 / 200  # 60Hz
        self.sim.render_interval = 5

        # Convert USD to URDF and change revolute joints to fixed
        temp_urdf_output_path, temp_urdf_meshes_output_path = ControllerUtils.convert_usd_to_urdf(
            self.scene.robot.spawn.usd_path, self.temp_urdf_dir, force_conversion=True
        )

        # Set the URDF and mesh paths for the IK controller
        self.actions.upper_body_ik.controller.urdf_path = temp_urdf_output_path
        self.actions.upper_body_ik.controller.mesh_path = temp_urdf_meshes_output_path