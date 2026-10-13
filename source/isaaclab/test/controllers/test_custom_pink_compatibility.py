# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression checks for the custom Pink actions migrated from the TianJi branch."""

from types import SimpleNamespace

from pink.tasks import DampingTask, FrameTask

from isaaclab.controllers.pink_ik import JointCouplingTask
from isaaclab.envs.mdp.actions.pink_task_space_actions import (
    PinkInverseKinematicsAction_NullSpacePostureControl,
)


def test_posture_action_dimension_before_helper_tensors():
    # The base constructor requests action_dim before creating _num_frame_tasks.
    action = object.__new__(PinkInverseKinematicsAction_NullSpacePostureControl)
    action.cfg = SimpleNamespace(controller=SimpleNamespace(num_posture_controlled_joints=9))
    action._ik_controllers = [
        SimpleNamespace(
            cfg=SimpleNamespace(
                variable_input_tasks=[
                    FrameTask("left", position_cost=1.0, orientation_cost=1.0),
                    DampingTask(cost=0.1),
                    FrameTask("right", position_cost=1.0, orientation_cost=1.0),
                ]
            )
        )
    ]
    assert action.action_dim == 23


def test_joint_coupling_preserves_keyword_constructor():
    task = JointCouplingTask(joint1="a", joint2="b", joint3="c")
    assert (task.joint1, task.joint2, task.joint3) == ("a", "b", "c")
