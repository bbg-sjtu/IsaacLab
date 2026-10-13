# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command absolute Moz1 joint positions; no Pink/Pinocchio solver is used."""

import argparse
import traceback

from isaaclab.app import AppLauncher

# 在这里直接修改 action：绝对关节目标，底盘 x/y 单位为 m，其余为 rad。
# 多环境运行时，每个机器人使用同一组目标。以下值与 reset 默认关节位置一致。
JOINT_TARGETS = [
    0.0,  # 0: dummy_x_joint
    0.0,  # 1: dummy_y_joint
    0.0,  # 2: dummy_yaw_joint
    0.0,  # 3: LegWaist_0
    1.0,  # 4: LegWaist_1
    -1.5,  # 5: LegWaist_2
    0.5,  # 6: LegWaist_3
    0.0,  # 7: LegWaist_4
    0.0,  # 8: LegWaist_5
    0.0,  # 9: LeftArm_0
    -0.4,  # 10: LeftArm_1
    0.0,  # 11: LeftArm_2
    -1.5,  # 12: LeftArm_3
    -1.2,  # 13: LeftArm_4
    0.0,  # 14: LeftArm_5
    0.0,  # 15: LeftArm_6
    0.0,  # 16: RightArm_0
    -0.4,  # 17: RightArm_1
    0.0,  # 18: RightArm_2
    1.5,  # 19: RightArm_3
    1.2,  # 20: RightArm_4
    0.0,  # 21: RightArm_5
    0.0,  # 22: RightArm_6
]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=200)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.num_envs <= 0 or args_cli.steps <= 0:
    parser.error("num_envs and steps must be positive")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.manager_based.manipulation.Moz1.moz1_joint_pos_env_cfg import (  # noqa: E402
    MOZ1_JOINT_ACTION_NAMES,
)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


def main() -> None:
    """Send joint targets in [m or rad, depending on joint type] at 120 Hz."""
    task = "Isaac-Moz1-Joint-Position-v0"
    cfg = parse_env_cfg(task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(task, cfg=cfg).unwrapped
    try:
        env.reset()
        robot = env.scene["robot"]
        joint_ids, joint_names = robot.find_joints(list(MOZ1_JOINT_ACTION_NAMES), preserve_order=True)
        if tuple(joint_names) != MOZ1_JOINT_ACTION_NAMES or env.action_manager.total_action_dim != 23:
            raise RuntimeError("Unexpected joint action mapping")
        print("Action order:", list(enumerate(joint_names)), flush=True)
        if len(JOINT_TARGETS) != len(joint_names):
            raise ValueError("JOINT_TARGETS must contain exactly 23 joint positions")
        action = torch.tensor(JOINT_TARGETS, device=env.device, dtype=torch.float32).repeat(env.num_envs, 1)
        limits = robot.data.joint_pos_limits.torch[:, joint_ids]
        if not torch.isfinite(action).all() or ((action < limits[..., 0]) | (action > limits[..., 1])).any():
            raise ValueError("JOINT_TARGETS must be finite and within URDF joint position limits")

        max_error = torch.zeros(23, device=env.device)
        for step in range(args_cli.steps):
            # 直接发送指定目标，不做正弦运动或插值；由关节 PD 驱动跟踪。
            env.step(action)
            # Verify the manager introduces neither scale nor default-pose offsets.
            torch.testing.assert_close(env.action_manager.get_term("joint_pos").processed_actions, action)
            actual = robot.data.joint_pos.torch[:, joint_ids]
            if not torch.isfinite(actual).all():
                raise RuntimeError(f"Non-finite joint state at step {step}")
            max_error = torch.maximum(max_error, (actual - action).abs().amax(dim=0))
        print(
            f"MOZ1_JOINT_VALIDATION steps={args_cli.steps} envs={env.num_envs} "
            f"max_base_xy_error_m={max_error[:2].max().item():.6f} "
            f"max_angular_joint_error_rad={max_error[2:].max().item():.6f}",
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except BaseException:
        exit_code = 1
        traceback.print_exc()
        raise
    finally:
        simulation_app.close(exit_code=exit_code)
