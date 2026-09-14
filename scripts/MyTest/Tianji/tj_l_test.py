# 双臂操作数据太少，先用单臂抓取放置测试一下
# 数据可通过随机化物体位置，获取仿真环境中物体的位置信息，然后给出机械手末端目标位姿，然后控制抓取从而产生数据
# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to replay demonstrations with Isaac Lab environments."""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

import pandas as pd

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay demonstrations in Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to replay episodes.")
# parser.add_argument("--task", type=str, default=None, help="Force to use the specified task.")
parser.add_argument("--task", type=str, default="Isaac-TJ-L-v0", help="Force to use the specified task.")

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
from isaaclab.utils.math import subtract_frame_transforms
def main():

    if args_cli.task is not None:
        env_name = args_cli.task
    if env_name is None:
        raise ValueError("Task/env name was not specified nor found in the dataset.")

    num_envs = args_cli.num_envs

    env_cfg = parse_env_cfg(env_name, device=args_cli.device, num_envs=num_envs)

    # Disable all recorders and terminations
    # env_cfg.recorders = {}
    env_cfg.terminations = {}

    # create environment from loaded config
    env = gym.make(env_name, cfg=env_cfg).unwrapped

    env.reset()
    actions=torch.tensor([[0,1.57,1.57,-1.57,0,0,0]])
    for i in range(30):
        env.step(actions)
    state=env.obs_buf['policy']['states'].cpu().numpy()

    actions=torch.tensor([[0,1.57,1.57,-0.5,0,0,0]])
    for i in range(30):
        env.step(actions)
    state=env.obs_buf['policy']['states'].cpu().numpy()

    actions=torch.tensor([[0,1.57,1.57,-0.4,0,0,0]])
    for i in range(30):
        env.step(actions)
    state=env.obs_buf['policy']['states'].cpu().numpy()
    
    while True:
        env.step(actions)
    # actions=torch.tensor([[2.5548322, 1.4813887, -1.0005894, 0.3611008, 0.36108363, -1.2088392, 0.0, 0.0, 0.0]])
    # for i in range(20):
    #     env.step(actions)
    print("done")

# tensor([[ 1.7493e-01, -6.8800e-01,  8.9427e-01,  7.0350e-01,  7.1070e-01,
#           6.1800e-04, -1.8703e-04]], device='cuda:0')
# tensor([[ 1.7493e-01, -6.8800e-01,  8.9427e-01,  7.0350e-01,  7.1070e-01,
#           6.1805e-04, -1.8700e-04]], device='cuda:0')
if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
