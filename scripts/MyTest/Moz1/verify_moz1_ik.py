# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check Moz1 model loading and Pink convergence without starting Isaac Sim."""

import numpy as np
import pinocchio as pin
from pink.tasks import FrameTask

from isaaclab.controllers.pink_ik import PinkIKController

from isaaclab_tasks.manager_based.manipulation.Moz1.moz1_cfg import MOZ1_CFG, MOZ1_URDF_PATH
from isaaclab_tasks.manager_based.manipulation.Moz1.moz1_pink_wbc_env_cfg import ActionsCfg


def main() -> None:
    """Track a reachable, stationary dual-wrist target with the 23-DoF model."""
    model = pin.buildModelFromUrdf(str(MOZ1_URDF_PATH))
    assert model.nq == model.nv == 23
    names = list(model.names)[1:]
    action_cfg = ActionsCfg().whole_body_ik
    cfg = action_cfg.controller
    cfg.joint_names = names
    cfg.all_joint_names = names
    cfg.fail_on_joint_limit_violation = True
    controller = PinkIKController(
        cfg,
        MOZ1_CFG,
        "cpu",
        list(range(model.nv)),
    )
    q = controller.init_joint_positions.copy()
    controller.pink_configuration.update(q)
    frame_tasks = [task for task in controller.cfg.variable_input_tasks if isinstance(task, FrameTask)]
    assert len(frame_tasks) == 2
    for task in frame_tasks:
        target = controller.pink_configuration.get_transform_frame_to_world(task.frame)
        target.translation[1] += 0.03
        task.set_target(target)

    def position_error() -> float:
        return max(
            np.linalg.norm(
                controller.pink_configuration.get_transform_frame_to_world(task.frame).translation
                - task.transform_target_to_world.translation
            )
            for task in frame_tasks
        )

    initial_error = position_error()
    for _ in range(240):
        q = controller.compute(q, 1.0 / 120.0).numpy()
        assert np.isfinite(q).all()
        controller.pink_configuration.update(q)
    final_error = position_error()
    assert final_error < 1e-3, f"Pink failed to converge: {final_error:.6f} m"
    print(f"PASS: nq=nv=23, dual-wrist error {initial_error:.6f} -> {final_error:.6f} m")
    print("This checks kinematics and the QP solver only; Isaac Sim/USD validation is separate.")


if __name__ == "__main__":
    main()
