# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import tempfile
import torch

from pink.tasks import FrameTask

import isaaclab.controllers.utils as ControllerUtils
import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
# from isaaclab.controllers.pink_ik_cfg import PinkIKControllerCfg
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
from isaaclab.utils.configclass import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp

from isaaclab_assets.robots.fourier import GR1T2_CFG  # isort: skip
from .uragv_cfg import URAGV_CFG

from isaaclab.sensors import CameraCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.envs.common import ViewerCfg
##
# Scene definition
##
@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):

    # Table
    packing_table1 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.9, -0.45], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path="/WorkSpace/LiuBang/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/uragv/Table1.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )
    # Object
    container = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Container",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.8, 0.8, 0.99257-0.45], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path="/WorkSpace/LiuBang/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/uragv/container.usd",
        ),
    )

    box = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Box",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-1.55, -0.8, 1.0413-0.4], rot=[0.7071, 0.0, 0.0, 0.7071]), #-1.55, -0.62
        spawn=UsdFileCfg(
            usd_path="/WorkSpace/LiuBang/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/uragv/box.usd",
        ),
    )

    # object = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Object",
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=[-1.55, -0.62, 1.0413-0.45], rot=[1, 0, 0, 0]),# -1.6,-0.62
    #     spawn=sim_utils.CuboidCfg(
    #         size=(0.04,0.04,0.4),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.15, 0.15), metallic=1.0),
    #         physics_material=sim_utils.RigidBodyMaterialCfg(
    #             # friction_combine_mode="max",
    #             # restitution_combine_mode="min",
    #             static_friction=0.99,
    #             dynamic_friction=0.99,
    #             restitution=0.0,
    #         ),
    #     ),
    # )

    # Humanoid robot w/ arms higher
    robot: ArticulationCfg = URAGV_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0, 0.05),
            # pos=(0, 0, 0.0),
            # rot=(0.7071, 0.7071, 0, 0),
            # rot=(0.7071, 0, 0,0.7071),
            # rot=(0.5, 0.5, 0.5, 0.5),
            joint_pos={
                # "dummy_.*":1.0,
                "ur_arm_shoulder_pan_joint":3.14,
                "ur_arm_shoulder_lift_joint":-1.57,
                "ur_arm_elbow_joint":1.5,
                "ur_arm_wrist_1_joint":-3.14,
                "ur_arm_wrist_2_joint":-1.57,
                "ur_arm_wrist_3_joint":3.14,
                "Slider_1":0.0,
                "Slider_2":0.0,
            },
            # joint_vel={".*": 0.0},
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

    camera_top = CameraCfg(
        prim_path="/World/envs/env_.*/PackingTable/camera_top",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=11.9, focus_distance=400.0, horizontal_aperture=20.955,vertical_aperture=15.2908, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(-0.568, -0.852, 3.546), rot=(0,-1,0,0), convention="ros"),
    )
    camera_1 = CameraCfg(
        prim_path="/World/envs/env_.*/PackingTable/camera_1",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.148, focus_distance=400.0, horizontal_aperture=20.955,vertical_aperture=15.2908, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(-3.337, -1.428, 2.278), rot=(0.36652,-0.6047,0.6047,-0.36652), convention="ros"),
        # 原：0.6047,0.36652,-0.36652,-0.6047，需要变换一下次序，奇怪的bug
        # (x1,x2,x3,x4) -> (x2,-x1,x4,-x3)
    )
    camera_2 = CameraCfg(
        prim_path="/World/envs/env_.*/PackingTable/camera_2",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.148, focus_distance=400.0, horizontal_aperture=20.955,vertical_aperture=15.2908, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(-0.082,1.955,2.277), rot=(0.0, 0.0, -0.85408,0.52015), convention="ros"),
    )
    camera_wrist = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/gripper1/RobotIq_Hand_E_base/ee_link/camera_wrist",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=14.5, focus_distance=400.0, horizontal_aperture=20.955,vertical_aperture=15.2908, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(-0.13688,0.0,0.09234), rot=(-0.40343,0.58072,-0.58072,0.40343), convention="ros"),
    )
##
# MDP settings
##
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    # 关节位置控制
    joint_position = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=[
            "ur_arm_.*",
            "dummy_base_revolute_.*",
            "dummy_base_prismatic_.*",
            "Slider_.*"
        ],
        use_default_offset=False # 如果是true的话，会在给定action上增加一个offset，
        # offset的值见/WorkSpace/LiuBang/IsaacLab/source/isaaclab/isaaclab/envs/mdp/actions/joint_actions.py line.156
    )
    # joint_velocity = mdp.JointVelocityActionCfg(
    #     asset_name="robot",
    #     joint_names=[
    #         "dummy_base_prismatic_.*"
    #     ],
    #     use_default_offset=False
    # )


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
        table1_img=ObsTerm(func=mdp.get_table1_cam)
        table2_img=ObsTerm(func=mdp.get_table2_cam)
        # wrist_img=ObsTerm(func=mdp.get_wrist_cam)

        actions = ObsTerm(func=mdp.last_action)
        states = ObsTerm(func=mdp.get_joint_pos_uragv)

        # right_hand_open_state=ObsTerm(func=mdp.get_right_hand_open_state)


        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("object")}
    )

    success = DoneTerm(func=mdp.task_done)


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": [0, 0.15],#增量
                "y": [0, 0.4],
            },
            "velocity_range":{},
            "asset_cfg": SceneEntityCfg("box"),
        },
    )

    reset_container = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": [0, 1.6],
            },
            "velocity_range":{},
            "asset_cfg": SceneEntityCfg("container"),
        },
    )
    


@configclass
class URAGVEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the GR1T2 environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=1, env_spacing=10, replicate_physics=True)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()
    events = EventCfg()

    viewer: ViewerCfg = ViewerCfg(eye=(2,-2,2),lookat=(0,0,1))

    # Unused managers
    commands = None
    rewards = None
    curriculum = None

    # Position of the XR anchor in the world frame
    xr: XrCfg = XrCfg(
        anchor_pos=(0.0, 0.0, 0.0),
        anchor_rot=(1.0, 0.0, 0.0, 0.0),
    )

    # Temporary directory for URDF files
    temp_urdf_dir = tempfile.gettempdir()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 1 # decimation=5的意思是，env层动作指令的频率是sim实际仿真频率的1/5，假设sim仿真步长是0.01s，那么env层动作指令的频率就是0.05s，
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 1 / 60  # 100Hz
        self.sim.render_interval = 1
