# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Moz1-specific Pink whole-body action configuration."""

from __future__ import annotations

import warnings

import numpy as np
import torch

from isaaclab.controllers.pink_ik import WaistTaskCfg
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg_NullSpacePostureControl
from isaaclab.envs.mdp.actions.pink_task_space_actions import PinkInverseKinematicsAction_NullSpacePostureControl
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import math as math_utils
from isaaclab.utils.configclass import configclass


class Moz1PinkWholeBodyIKAction(PinkInverseKinematicsAction_NullSpacePostureControl):
    """Track environment-relative goals using the fixed URDF root as Pink's world."""

    def _initialize_ik_controllers(self) -> None:
        # Deprecated aliases can override an explicitly declared task, but must
        # never create a hidden task. Normal configurations use only controller.
        overrides = {"cost": self.cfg.waist_coupling_cost, "sign_cost": self.cfg.waist_sign_cost}
        if any(value is not None for value in overrides.values()):
            warnings.warn(
                "waist_coupling_cost/waist_sign_cost are deprecated; configure WaistTaskCfg.cost/sign_cost "
                "inside controller.fixed_input_tasks instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.cfg.controller = self.cfg.controller.copy()
            tasks = [
                task
                for task in self.cfg.controller.fixed_input_tasks + self.cfg.controller.variable_input_tasks
                if isinstance(task, WaistTaskCfg)
            ]
            if len(tasks) != 1:
                raise ValueError("Deprecated waist costs require exactly one explicit WaistTaskCfg.")
            for name, value in overrides.items():
                if value is not None:
                    setattr(tasks[0], name, value)
        super()._initialize_ik_controllers()

    def _set_frame_and_posture_targets(
        self,
        poses_env: torch.Tensor,
        poses_base: tuple[torch.Tensor, torch.Tensor],
        posture_targets: np.ndarray,
    ) -> None:
        # Pinocchio's world is world_base, not the environment origin. Account
        # for the robot's spawn translation/rotation before assigning FrameTask goals.
        poses_pink = math_utils.make_pose(*poses_base)
        super()._set_frame_and_posture_targets(poses_pink, poses_base, posture_targets)


@configclass
class Moz1PinkWholeBodyIKActionCfg(PinkInverseKinematicsActionCfg_NullSpacePostureControl):
    """Configuration for :class:`Moz1PinkWholeBodyIKAction`."""

    class_type: type[ActionTerm] = Moz1PinkWholeBodyIKAction
    waist_coupling_cost: float | None = None
    """Deprecated override [cost/rad]; set :attr:`WaistTaskCfg.cost` instead."""
    waist_sign_cost: float | None = None
    """Deprecated override [cost/rad]; set :attr:`WaistTaskCfg.sign_cost` instead."""
