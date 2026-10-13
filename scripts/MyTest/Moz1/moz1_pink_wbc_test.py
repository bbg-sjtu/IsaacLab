# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Track two end-effector targets with Moz1 whole-body Pink IK."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="Isaac-Moz1-Pink-WBC-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=1200, help="Number of environment/control steps.")
parser.add_argument("--period", type=float, default=5.0, help="Target trajectory period in seconds.")
parser.add_argument("--radius", type=float, default=0.20, help="End-effector circle radius in metres.")
parser.add_argument("--max_position_error", type=float, default=None, help="Fail above this tracking error [m].")
parser.add_argument("--diagnostics", action="store_true", help="Print initial FK and tracking diagnostics.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.steps <= 0 or args_cli.period <= 0 or args_cli.num_envs <= 0 or args_cli.radius < 0:
    parser.error("steps, period and num_envs must be positive; radius must be nonnegative")

# Pinocchio must be imported before AppLauncher so Isaac Lab's compatible
# version wins over the copy bundled with Isaac Sim.
import pinocchio  # noqa: E402, F401

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import math  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.markers import VisualizationMarkers  # noqa: E402
from isaaclab.markers.config import FRAME_MARKER_CFG  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.manager_based.manipulation.Moz1.moz1_cfg import (  # noqa: E402
    MOZ1_DEFAULT_POSTURE,
    MOZ1_LEFT_EEF_LINK,
    MOZ1_RIGHT_EEF_LINK,
)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    try:
        env.reset()

        eef_cfg = SceneEntityCfg("robot", body_names=[MOZ1_LEFT_EEF_LINK, MOZ1_RIGHT_EEF_LINK])
        eef_cfg.resolve(env.scene)

        marker_cfg = FRAME_MARKER_CFG.copy()
        left_goal_marker = VisualizationMarkers(marker_cfg.replace(prim_path="/Visuals/Moz1/left_goal"))
        right_goal_marker = VisualizationMarkers(marker_cfg.replace(prim_path="/Visuals/Moz1/right_goal"))
        left_current_marker = VisualizationMarkers(marker_cfg.replace(prim_path="/Visuals/Moz1/left_current"))
        right_current_marker = VisualizationMarkers(marker_cfg.replace(prim_path="/Visuals/Moz1/right_current"))

        robot = env.scene["robot"]
        if args_cli.diagnostics:
            term = env.action_manager.get_term("whole_body_ik")
            controller = term._ik_controllers[0]
            q = robot.data.joint_pos.torch[0].cpu().numpy()
            controller.pink_configuration.update(q[controller.isaac_lab_to_pink_ordering])
            print("MOZ1_INITIAL_Q", q, flush=True)
            for body_id, name in zip(eef_cfg.body_ids, (MOZ1_LEFT_EEF_LINK, MOZ1_RIGHT_EEF_LINK)):
                print(
                    "MOZ1_FK",
                    name,
                    "sim",
                    robot.data.body_link_pose_w.torch[0, body_id].tolist(),
                    "pin",
                    controller.pink_configuration.get_transform_frame_to_world(name).translation,
                    flush=True,
                )
        left_home = robot.data.body_link_pose_w.torch[:, eef_cfg.body_ids[0]].clone()
        right_home = robot.data.body_link_pose_w.torch[:, eef_cfg.body_ids[1]].clone()
        posture_target = torch.tensor(MOZ1_DEFAULT_POSTURE, dtype=torch.float32, device=env.device).repeat(
            env.num_envs, 1
        )

        control_dt = env_cfg.sim.dt * env_cfg.decimation
        period_steps = max(1, round(args_cli.period / control_dt))
        position_errors = []

        for step in range(args_cli.steps):
            phase = 2.0 * math.pi * step / period_steps
            forward_offset = args_cli.radius * (1.0 - math.cos(phase))
            vertical_offset = args_cli.radius * math.sin(phase)

            left_goal = left_home.clone()
            right_goal = right_home.clone()
            # Moz1's public model uses +Y as the forward direction. Both hands
            # move together so the virtual base and waist participate naturally.
            left_goal[:, 1] += forward_offset
            right_goal[:, 1] += forward_offset
            left_goal[:, 2] += vertical_offset
            right_goal[:, 2] -= vertical_offset

            left_goal_env = left_goal.clone()
            right_goal_env = right_goal.clone()
            left_goal_env[:, :3] -= env.scene.env_origins
            right_goal_env[:, :3] -= env.scene.env_origins
            action = torch.cat((left_goal_env, right_goal_env, posture_target), dim=-1)
            left_goal_marker.visualize(left_goal[:, :3], left_goal[:, 3:7])
            right_goal_marker.visualize(right_goal[:, :3], right_goal[:, 3:7])

            env.step(action)

            left_current = robot.data.body_link_pose_w.torch[:, eef_cfg.body_ids[0]]
            right_current = robot.data.body_link_pose_w.torch[:, eef_cfg.body_ids[1]]
            if not torch.isfinite(robot.data.joint_pos.torch).all():
                raise RuntimeError(f"Non-finite joint state at step {step}")
            errors = torch.stack(
                (
                    torch.linalg.vector_norm(left_current[:, :3] - left_goal[:, :3], dim=-1),
                    torch.linalg.vector_norm(right_current[:, :3] - right_goal[:, :3], dim=-1),
                )
            )
            if not torch.isfinite(errors).all():
                raise RuntimeError(f"Non-finite end-effector state at step {step}")
            position_errors.append(errors)
            if args_cli.diagnostics and step in (0, 1, 59, args_cli.steps - 1):
                print(
                    "MOZ1_STEP",
                    step,
                    "errors",
                    errors.tolist(),
                    "q",
                    robot.data.joint_pos.torch[0].tolist(),
                    flush=True,
                )
                print(
                    "MOZ1_TARGET",
                    term.processed_actions[0].tolist(),
                    "base",
                    term.base_link_frame_in_world_rf[0].tolist(),
                    flush=True,
                )
                print(
                    "MOZ1_TASK_ERRORS",
                    [
                        task.compute_error(controller.pink_configuration).tolist()
                        for task in controller._variable_input_tasks
                        if hasattr(task, "frame")
                    ],
                    flush=True,
                )
            left_current_marker.visualize(left_current[:, :3], left_current[:, 3:7])
            right_current_marker.visualize(right_current[:, :3], right_current[:, 3:7])
        errors = torch.stack(position_errors)
        maximum_error = errors.max().item()
        print(
            f"MOZ1_VALIDATION steps={args_cli.steps} envs={env.num_envs} "
            f"rms_position_error_m={errors.square().mean().sqrt().item():.6f} "
            f"max_position_error_m={maximum_error:.6f}"
        )
        if args_cli.max_position_error is not None and maximum_error > args_cli.max_position_error:
            raise RuntimeError(f"Tracking error {maximum_error:.6f} m exceeds {args_cli.max_position_error} m")
    finally:
        env.close()


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except BaseException:
        # Kit shutdown can terminate the process before Python prints a pending
        # exception. Print it first so failed validations cannot appear to pass.
        import traceback

        exit_code = 1
        traceback.print_exc()
        raise
    finally:
        simulation_app.close(exit_code=exit_code)
