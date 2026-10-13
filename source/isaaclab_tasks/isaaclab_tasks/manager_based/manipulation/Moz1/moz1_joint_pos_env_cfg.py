# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Moz1 absolute joint-position control, without an IK controller."""

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass

from .moz1_cfg import MOZ1_CFG

# Explicit action order, independent of the USD articulation's internal order.
MOZ1_JOINT_ACTION_NAMES = (
    "dummy_x_joint",
    "dummy_y_joint",
    "dummy_yaw_joint",
    *(f"LegWaist_{i}" for i in range(6)),
    *(f"LeftArm_{i}" for i in range(7)),
    *(f"RightArm_{i}" for i in range(7)),
)


@configclass
class Moz1JointSceneCfg(InteractiveSceneCfg):
    """Reuse the local USD and PD actuators in a gravity/contact-free scene."""

    robot: ArticulationCfg = MOZ1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(color=(0.15, 0.15, 0.15)),
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.8, 0.8, 0.8), intensity=1200.0),
    )


@configclass
class Moz1JointActionsCfg:
    """Absolute positions [m for base x/y, rad for all other joints]."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(MOZ1_JOINT_ACTION_NAMES),
        preserve_order=True,
        scale=1.0,
        offset=0.0,
        use_default_offset=False,
    )


@configclass
class Moz1JointObservationsCfg:
    """Joint positions in the same order as the action."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        joint_pos = ObservationTermCfg(
            func=mdp.joint_pos,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=list(MOZ1_JOINT_ACTION_NAMES), preserve_order=True)
            },
        )
        enable_corruption = False
        concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class Moz1JointPositionEnvCfg(ManagerBasedRLEnvCfg):
    """Direct joint-position tracking with a 23-dimensional absolute action."""

    scene: Moz1JointSceneCfg = Moz1JointSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=True)
    actions: Moz1JointActionsCfg = Moz1JointActionsCfg()
    observations: Moz1JointObservationsCfg = Moz1JointObservationsCfg()
    commands = {}
    rewards = {}
    terminations = {}
    curriculum = {}
    events = {"reset_scene": EventTermCfg(func=mdp.reset_scene_to_default, mode="reset")}
    viewer: ViewerCfg = ViewerCfg(eye=(2.2, 2.2, 1.8), lookat=(0.0, 0.3, 0.9))

    def __post_init__(self):
        self.sim.dt = 1.0 / 240.0
        self.sim.gravity = (0.0, 0.0, 0.0)
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 30.0
