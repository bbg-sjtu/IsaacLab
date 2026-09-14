
import argparse

from isaaclab.app import AppLauncher
# add argparse arguments
parser = argparse.ArgumentParser(description="Replay demonstrations in Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to replay episodes.")
# parser.add_argument("--task", type=str, default=None, help="Force to use the specified task.")
parser.add_argument("--task", type=str, default="Isaac-TJ-MM-v1", help="Force to use the specified task.")

parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=True,
    help="Enable Pinocchio.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# args_cli.headless = True

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and not the one installed by Isaac Sim
    # pinocchio is required by the Pink IK controllers and the GR1T2 retargeter
    import pinocchio  # noqa: F401

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import gymnasium as gym
import os
import torch

from isaaclab.devices import Se3Keyboard
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
import math
def main():

    if args_cli.task is not None:
        env_name = args_cli.task
    if env_name is None:
        raise ValueError("Task/env name was not specified nor found in the dataset.")

    num_envs = args_cli.num_envs

    env_cfg = parse_env_cfg(env_name, device=args_cli.device, num_envs=num_envs)

    env_cfg.terminations = {}

    # create environment from loaded config
    env = gym.make(env_name, cfg=env_cfg).unwrapped

    env.reset()

    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    goal_marker_L = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal_L"))
    goal_marker_R = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal_R"))
    ee_marker_L = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current_L"))
    ee_marker_R = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current_R"))

    robot_entity_cfg = SceneEntityCfg("robot", 
                        body_names=[
                            "ee_L",
                            "ee_R"
                            ])
    robot_entity_cfg.resolve(env.scene)

    # # test1
    # action = torch.tensor([[0.4, -0.2,  0.8, -0.0222, -0.7344, -0.0244, -0.6779, 
    #                         0.4,  0.5,  0.8,  0.0219, -0.7346,  0.0244, -0.6777,
    #                         0.0, -0.7,1.2,-0.5]])
    # goal_marker_L.visualize(action[:, 0:3], action[:, 3:7])
    # goal_marker_R.visualize(action[:, 7:10], action[:, 10:14])

    # for i in range(150):
    #     env.step(action)
    #     ee_L_pose_w = env.scene["robot"].data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
    #     ee_R_pose_w = env.scene["robot"].data.body_state_w[:, robot_entity_cfg.body_ids[1], 0:7]
    #     ee_marker_L.visualize(ee_L_pose_w[:, 0:3], ee_L_pose_w[:, 3:7])
    #     ee_marker_R.visualize(ee_R_pose_w[:, 0:3], ee_R_pose_w[:, 3:7])

    # print('yes')

    # test2
    num_steps = 200

    # 建议直接用仿真环境的 device，避免 CPU/GPU 不一致
    device = args_cli.device if hasattr(env, "device") else "cpu"

    # 左右末端圆心
    center_L = torch.tensor([0.4320, -0.34, 0.8283], dtype=torch.float32, device=device)
    center_R = torch.tensor([0.4320, 0.34, 0.8283], dtype=torch.float32, device=device)

    # 半径
    radius_L = 0.2
    radius_R = 0.15

    # 姿态保持不变，注意这里沿用你原来的四元数顺序
    quat_L = torch.tensor([-0.0222, -0.7344, -0.0244, -0.6779], dtype=torch.float32, device=device)
    quat_R = torch.tensor([ 0.0222, -0.7344,  0.0244, -0.6779], dtype=torch.float32, device=device)

    # 最后4个指令保持不变
    joint_cmd = torch.tensor([0.0, -0.7, 1.2, -0.5, 0.0, 0.0], dtype=torch.float32, device=device)

    for i in range(num_steps*2):
        # 一个周期
        theta = 2.0 * math.pi * i / (num_steps - 1)

        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        # 左末端圆路径：x-y 平面，z 不变
        pos_L = center_L + torch.tensor(
            [0.0, radius_L * cos_t-0.15, radius_L * sin_t],
            dtype=torch.float32,
            device=device,
        )

        # 右末端圆路径：x-y 平面，z 不变
        pos_R = center_R + torch.tensor(
            [radius_R * cos_t-0.15, 0.0,radius_R * sin_t],
            dtype=torch.float32,
            device=device,
        )

        # 拼接 action: 7 + 7 + 4
        action = torch.cat(
            [
                pos_L,
                quat_L,
                pos_R,
                quat_R,
                joint_cmd,
            ],
            dim=0,
        ).unsqueeze(0)

        # 可视化目标
        goal_marker_L.visualize(action[:, 0:3], action[:, 3:7])
        goal_marker_R.visualize(action[:, 7:10], action[:, 10:14])

        # 仿真步进
        env.step(action)

        # 可视化实际末端
        ee_L_pose_w = env.scene["robot"].data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        ee_R_pose_w = env.scene["robot"].data.body_state_w[:, robot_entity_cfg.body_ids[1], 0:7]

        ee_marker_L.visualize(ee_L_pose_w[:, 0:3], ee_L_pose_w[:, 3:7])
        ee_marker_R.visualize(ee_R_pose_w[:, 0:3], ee_R_pose_w[:, 3:7])

    print('yes')

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
