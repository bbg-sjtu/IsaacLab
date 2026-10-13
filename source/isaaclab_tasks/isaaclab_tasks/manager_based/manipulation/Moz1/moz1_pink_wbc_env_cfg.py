# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based Moz1 whole-body end-effector tracking environment."""

from __future__ import annotations

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.controllers.pink_ik import (
    DampingTaskCfg,
    FrameTaskCfg,
    NullSpacePostureTaskCfg,
    PinkIKControllerCfg,
    WaistTaskCfg,
)
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils.configclass import configclass

from .moz1_actions import Moz1PinkWholeBodyIKActionCfg
from .moz1_cfg import (
    MOZ1_BASE_REFERENCE_LINK,
    MOZ1_CFG,
    MOZ1_LEFT_EEF_LINK,
    MOZ1_POSTURE_JOINT_NAMES,
    MOZ1_RIGHT_EEF_LINK,
    MOZ1_ROBOT_MODEL_DIR,
    MOZ1_URDF_PATH,
)


@configclass
class Moz1SceneCfg(InteractiveSceneCfg):
    """A minimal scene for validating Moz1 whole-body IK."""

    robot: ArticulationCfg = MOZ1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(color=(0.15, 0.15, 0.15)),
    )

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.8, 0.8, 0.8), intensity=1200.0),
    )


@configclass
class ActionsCfg:
    """Two environment-relative xyz/xyzw poses followed by a 9-DoF posture target."""

    whole_body_ik = Moz1PinkWholeBodyIKActionCfg(
        asset_name="robot",
        enable_gravity_compensation=False,
        pink_controlled_joint_names=[
            "dummy_.*_joint",
            "LegWaist_.*",
            "LeftArm_.*",
            "RightArm_.*",
        ],
        hand_joint_names=[],
        target_eef_link_names={
            "left_eef": MOZ1_LEFT_EEF_LINK,
            "right_eef": MOZ1_RIGHT_EEF_LINK,
        },
        controller=PinkIKControllerCfg(
            articulation_name="robot",
            base_link_name=MOZ1_BASE_REFERENCE_LINK,
            urdf_path=str(MOZ1_URDF_PATH),
            # Pinocchio resolves package://spirit01_model paths from this parent.
            mesh_path=str(MOZ1_ROBOT_MODEL_DIR),
            num_posture_controlled_joints=len(MOZ1_POSTURE_JOINT_NAMES),
            show_ik_warnings=True,
            fail_on_joint_limit_violation=False,
            variable_input_tasks=[
                FrameTaskCfg(
                    frame=MOZ1_LEFT_EEF_LINK,
                    position_cost=10.0,
                    orientation_cost=2.0,
                    lm_damping=0.0,
                    gain=0.5,
                ),
                FrameTaskCfg(
                    frame=MOZ1_RIGHT_EEF_LINK,
                    position_cost=10.0,
                    orientation_cost=2.0,
                    lm_damping=0.0,
                    gain=0.5,
                ),
                DampingTaskCfg(cost=0.2),
                NullSpacePostureTaskCfg(
                    cost=0.25,
                    lm_damping=1.0,
                    gain=0.5,
                    controlled_frames=[MOZ1_LEFT_EEF_LINK, MOZ1_RIGHT_EEF_LINK],
                    controlled_joints=list(MOZ1_POSTURE_JOINT_NAMES),
                ),
            ],
            fixed_input_tasks=[
                # All waist preferences are declared here, with no hidden tasks.
                WaistTaskCfg(
                    joint1="LegWaist_1",
                    joint2="LegWaist_2",
                    joint3="LegWaist_3",
                    cost=5.0,
                    sign_cost=5.0,
                    joint_signs=(1, -1, 1),
                    gain=0.5,
                    lm_damping=0.0,
                ),
            ],
        ),
    )


@configclass
class ObservationsCfg:
    """Expose joint positions for inspection and downstream applications."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class Moz1PinkWBCEnvCfg(ManagerBasedRLEnvCfg):
    """Moz1 Pink whole-body tracking environment configuration."""

    scene: Moz1SceneCfg = Moz1SceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()

    commands = {}
    rewards = {}
    terminations = {}
    events = {"reset_scene": EventTermCfg(func=mdp.reset_scene_to_default, mode="reset")}
    curriculum = {}

    viewer: ViewerCfg = ViewerCfg(eye=(2.2, 2.2, 1.8), lookat=(0.0, 0.3, 0.9))

    def __post_init__(self):
        # 240 Hz physics / 2 = 120 Hz task-space control, matching the rate
        # recommended by the public Moz1 integration documentation.
        self.sim.dt = 1.0 / 240.0
        # Public assets lack measured inertias. This is a gravity-free IK demo.
        self.sim.gravity = (0.0, 0.0, 0.0)
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 30.0
