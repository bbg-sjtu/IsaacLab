
import argparse

from isaaclab.app import AppLauncher

import pandas as pd
import time
import h5py
# add argparse arguments
parser = argparse.ArgumentParser(description="Replay demonstrations in Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to replay episodes.")
# parser.add_argument("--task", type=str, default=None, help="Force to use the specified task.")
parser.add_argument("--task", type=str, default="Isaac-TJ-LR-v0", help="Force to use the specified task.")

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

    action = torch.tensor([[0.4, -0.2,  0.5, -0.0222, -0.7344, -0.0244, -0.6779, 
                            0.4,  0.5,  0.5,  0.0219, -0.7346,  0.0244, -0.6777]])
    goal_marker_L.visualize(action[:, 0:3], action[:, 3:7])
    goal_marker_R.visualize(action[:, 7:10], action[:, 10:14])

    for i in range(150):
        env.step(action)
        ee_L_pose_w = env.scene["robot"].data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        ee_R_pose_w = env.scene["robot"].data.body_state_w[:, robot_entity_cfg.body_ids[1], 0:7]
        ee_marker_L.visualize(ee_L_pose_w[:, 0:3], ee_L_pose_w[:, 3:7])
        ee_marker_R.visualize(ee_R_pose_w[:, 0:3], ee_R_pose_w[:, 3:7])

    print('yes')

    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
