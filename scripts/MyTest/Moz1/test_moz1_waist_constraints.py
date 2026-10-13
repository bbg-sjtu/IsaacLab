# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU checks that waist preferences remain soft and have correct derivatives."""

from unittest.mock import patch

import numpy as np
import pinocchio as pin
import pytest
from pink import build_ik
from pink.tasks import PostureTask

from isaaclab.controllers.pink_ik import PinkIKController, WaistTask, WaistTaskCfg
from isaaclab.envs.mdp.actions.pink_task_space_actions import PinkInverseKinematicsAction_NullSpacePostureControl

from isaaclab_tasks.manager_based.manipulation.Moz1.moz1_actions import Moz1PinkWholeBodyIKAction
from isaaclab_tasks.manager_based.manipulation.Moz1.moz1_cfg import MOZ1_CFG, MOZ1_URDF_PATH
from isaaclab_tasks.manager_based.manipulation.Moz1.moz1_pink_wbc_env_cfg import ActionsCfg


def make_controller(reverse=False, cost=5.0):
    model = pin.buildModelFromUrdf(str(MOZ1_URDF_PATH))
    names = list(model.names)[1:]
    if reverse:
        names.reverse()
    cfg = ActionsCfg().whole_body_ik.controller
    for task in cfg.fixed_input_tasks:
        task.cost = cost
        task.sign_cost = cost
    cfg.joint_names = names
    cfg.all_joint_names = names
    # Use the real kinematic tree without repeatedly loading visualization meshes.
    with patch.object(pin.RobotWrapper, "BuildFromURDF", return_value=pin.RobotWrapper(model)):
        return PinkIKController(cfg, MOZ1_CFG, "cpu", list(range(model.nv)))


def waist_indices(controller):
    model = controller.pink_configuration.model
    return [model.joints[model.getJointId(name)].idx_q for name in ("LegWaist_1", "LegWaist_2", "LegWaist_3")]


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("dt", [1 / 240, 1 / 120])
def test_weights_reduce_violations_without_forcing_feasibility(reverse, dt):
    residuals = []
    for cost in (0.0, 5.0):
        controller = make_controller(reverse, cost)
        model = controller.pink_configuration.model
        indices = waist_indices(controller)
        q = controller.init_joint_positions.copy()
        q[indices] = (-0.2, -0.1, -0.1)
        task = PostureTask(cost=1.0, gain=0.5)
        task.set_target(q.copy())
        controller._variable_input_tasks = [task]
        result = controller.compute(q[controller.pink_to_isaac_lab_ordering], dt).numpy()
        next_q = result[controller.isaac_lab_to_pink_ordering]
        assert np.isfinite(next_q).all()
        assert next_q[indices[0]] < 0  # Soft: cannot and need not recover in one step.
        assert abs(next_q[indices].sum()) > 0.1
        assert np.all(np.abs(next_q - q) <= model.velocityLimit * dt + 1e-6)
        controller.pink_configuration.update(next_q)
        residuals.append(np.linalg.norm(controller._fixed_input_tasks[-1].compute_error(controller.pink_configuration)))
    assert residuals[1] < residuals[0]


def test_only_objective_changes_not_qp_constraints():
    controller = make_controller()
    cfg = controller.pink_configuration
    cfg.update(controller.init_joint_positions)
    original = build_ik(cfg, controller._variable_input_tasks, 1 / 120)
    soft = build_ik(cfg, controller._variable_input_tasks + controller._fixed_input_tasks, 1 / 120)
    assert soft.A is None and soft.b is None
    np.testing.assert_array_equal(soft.G, original.G)
    np.testing.assert_array_equal(soft.h, original.h)
    assert not np.allclose(soft.P, original.P)


def test_residual_and_jacobian():
    controller = make_controller()
    cfg = controller.pink_configuration
    q = controller.init_joint_positions.copy()
    task = controller._fixed_input_tasks[-1]
    cfg.update(q)
    np.testing.assert_allclose(task.compute_error(cfg), 0.0, atol=1e-7)
    q[waist_indices(controller)] = (-0.2, 0.01, -0.1)
    cfg.update(q)
    np.testing.assert_allclose(task.compute_error(cfg), [-0.29, 0.2, 0.01, 0.1])
    jacobian = task.compute_jacobian(cfg)
    error = task.compute_error(cfg)
    for i in range(len(q)):
        perturbed = q.copy()
        perturbed[i] += 1e-7
        cfg.update(perturbed)
        np.testing.assert_allclose((task.compute_error(cfg) - error) / 1e-7, jacobian[:, i], atol=1e-6)


def test_invalid_cost_is_rejected():
    with pytest.raises(ValueError, match="nonnegative"):
        make_controller(cost=-1.0)


def test_all_waist_preferences_are_explicit_without_hidden_tasks():
    action_cfg = ActionsCfg().whole_body_ik
    tasks = action_cfg.controller.fixed_input_tasks
    assert len(tasks) == 1
    assert isinstance(tasks[0], WaistTaskCfg)
    assert (tasks[0].joint1, tasks[0].joint2, tasks[0].joint3) == ("LegWaist_1", "LegWaist_2", "LegWaist_3")
    assert tasks[0].cost == tasks[0].sign_cost == 5.0
    assert tasks[0].joint_signs == (1, -1, 1)
    controller = make_controller()
    assert type(controller) is PinkIKController
    runtime_tasks = controller._fixed_input_tasks
    assert len(runtime_tasks) == 1
    assert isinstance(runtime_tasks[0], WaistTask)
    cfg = controller.pink_configuration
    q = controller.init_joint_positions.copy()
    indices = waist_indices(controller)
    q[indices] = (1.0, -1.5, 0.5)
    cfg.update(q)
    np.testing.assert_allclose(runtime_tasks[0].compute_error(cfg), [0.0, 0.0, 0.0, 0.0], atol=1e-12)
    q[indices[2]] += 0.1
    cfg.update(q)
    np.testing.assert_allclose(runtime_tasks[0].compute_error(cfg), [0.1, 0.0, 0.0, 0.0], atol=1e-12)


def test_removing_task_from_configuration_disables_all_waist_preferences():
    model = pin.buildModelFromUrdf(str(MOZ1_URDF_PATH))
    cfg = ActionsCfg().whole_body_ik.controller
    cfg.fixed_input_tasks = []
    cfg.joint_names = cfg.all_joint_names = list(model.names)[1:]
    with patch.object(pin.RobotWrapper, "BuildFromURDF", return_value=pin.RobotWrapper(model)):
        controller = PinkIKController(cfg, MOZ1_CFG, "cpu", list(range(model.nv)))
    assert controller._fixed_input_tasks == []


@pytest.mark.parametrize("field", ["cost", "sign_cost"])
@pytest.mark.parametrize("value", [-1.0, np.nan, np.inf])
def test_invalid_individual_weights_are_rejected(field, value):
    cfg = ActionsCfg().whole_body_ik.controller.fixed_input_tasks[0]
    setattr(cfg, field, value)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        WaistTask(cfg)


@pytest.mark.parametrize("signs", [(1, 2, -1), (1, -1), (1, np.nan, 1)])
def test_invalid_joint_signs_are_rejected(signs):
    cfg = ActionsCfg().whole_body_ik.controller.fixed_input_tasks[0]
    cfg.joint_signs = signs
    with pytest.raises(ValueError, match="joint_signs"):
        WaistTask(cfg)


def test_joint_signs_are_configurable_and_can_be_disabled_individually():
    controller = make_controller()
    cfg = ActionsCfg().whole_body_ik.controller.fixed_input_tasks[0]
    cfg.joint_signs = (-1, 1, 0)
    task = WaistTask(cfg)
    q = controller.init_joint_positions.copy()
    q[waist_indices(controller)] = (0.2, -0.1, 0.3)
    controller.pink_configuration.update(q)
    np.testing.assert_allclose(task.compute_error(controller.pink_configuration), [0.4, 0.2, 0.1, 0.0])
    np.testing.assert_allclose(task.compute_jacobian(controller.pink_configuration)[3], 0.0)


def test_action_delegates_to_standard_controller_without_injecting_tasks():
    action = object.__new__(Moz1PinkWholeBodyIKAction)
    action.cfg = ActionsCfg().whole_body_ik
    assert action.cfg.waist_coupling_cost is action.cfg.waist_sign_cost is None
    action.cfg.controller.fixed_input_tasks = []
    with patch.object(PinkInverseKinematicsAction_NullSpacePostureControl, "_initialize_ik_controllers") as initialize:
        action._initialize_ik_controllers()
    initialize.assert_called_once_with()
    assert action.cfg.controller.fixed_input_tasks == []


def test_deprecated_cost_alias_only_overrides_an_explicit_task():
    action = object.__new__(Moz1PinkWholeBodyIKAction)
    action.cfg = ActionsCfg().whole_body_ik
    action.cfg.waist_sign_cost = 2.0
    with (
        pytest.warns(DeprecationWarning, match="WaistTaskCfg"),
        patch.object(PinkInverseKinematicsAction_NullSpacePostureControl, "_initialize_ik_controllers"),
    ):
        action._initialize_ik_controllers()
    assert len(action.cfg.controller.fixed_input_tasks) == 1
    assert action.cfg.controller.fixed_input_tasks[0].sign_cost == 2.0
    action.cfg.controller.fixed_input_tasks = []
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="explicit WaistTaskCfg"):
        action._initialize_ik_controllers()
