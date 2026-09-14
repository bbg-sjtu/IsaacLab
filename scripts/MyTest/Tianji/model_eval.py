# 双臂操作数据太少，先用单臂抓取放置测试一下
# 数据可通过随机化物体位置，获取仿真环境中物体的位置信息，然后给出机械手末端目标位姿，然后控制抓取从而产生数据
# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to replay demonstrations with Isaac Lab environments."""

"""Launch Isaac Sim Simulator first."""


import argparse
import time
import h5py
from isaaclab.app import AppLauncher

# from VideoRecorder import singlearmVideoRecorderWrapper
import pandas as pd

from pickplace_action import getPickplaceAction

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay demonstrations in Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to replay episodes.")
# parser.add_argument("--task", type=str, default=None, help="Force to use the specified task.")
parser.add_argument("--task", type=str, default="Isaac-UR-AGV-v0", help="Force to use the specified task.")

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
import jsonlines


import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data # data functions
from utils import sample_box_pose, sample_insertion_pose # robot functions
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy, CNNMLPPolicy
from pickplace_action import hand_action

import IPython
e = IPython.embed

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

    set_seed(1)
    # command line parameters
    ckpt_dir = '/WorkSpace/LiuBang/act/checkpoints'
    policy_class = 'ACT'
    task_name = 'sim_uragv'
    batch_size_train = 8
    batch_size_val = 8
    num_epochs = 2000

    # get task parameters
    is_sim = task_name[:4] == 'sim_'
    if is_sim:
        from constants import SIM_TASK_CONFIGS
        task_config = SIM_TASK_CONFIGS[task_name]
    else:
        from aloha_scripts.constants import TASK_CONFIGS
        task_config = TASK_CONFIGS[task_name]
    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']

    # fixed parameters
    state_dim = 11
    lr_backbone = 1e-5
    backbone = 'resnet18'

    enc_layers = 4
    dec_layers = 7
    nheads = 8
    policy_config = {'lr': 1e-5,
                        'num_queries': 50,
                        'kl_weight': 10,
                        'hidden_dim': 512,
                        'dim_feedforward': 3200,
                        'lr_backbone': lr_backbone,
                        'backbone': backbone,
                        'enc_layers': enc_layers,
                        'dec_layers': dec_layers,
                        'nheads': nheads,
                        'camera_names': camera_names,
                        }
    
    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': 1e-5,
        'policy_class': policy_class,
        'onscreen_render': True,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': 10,
        'temporal_agg': True, # 时间集成
        'camera_names': camera_names,
        'real_robot': not is_sim
    }


    ckpt_name = 'policy_best.ckpt'

    eval_bc(config, ckpt_name, env,save_episode=True)
    print('done')
    env.close()


def eval_bc(config, ckpt_name, env, save_episode=True):
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']

    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']


    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = ACTPolicy(policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # load environment
    # if real_robot:
    #     from aloha_scripts.robot_utils import move_grippers # requires aloha
    #     from aloha_scripts.real_env import make_real_env # requires aloha
    #     env = make_real_env(init_node=True)
    #     env_max_reward = 0
    # else:
    #     from sim_env import make_sim_env
    #     env = make_sim_env(task_name)
    #     env_max_reward = env.task.max_reward

    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

    num_rollouts = 100
    num_success=0
    for rollout_id in range(num_rollouts):
        # ### set task
        # if 'sim_transfer_cube' in task_name:
        #     BOX_POSE[0] = sample_box_pose() # used in sim reset
        # elif 'sim_insertion' in task_name:
        #     BOX_POSE[0] = np.concatenate(sample_insertion_pose()) # used in sim reset

        # ts = env.reset()
        env.reset()
        states, curr_image=getObs(env)

        ### onscreen render
        # if onscreen_render:
        #     ax = plt.subplot()
        #     plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
        #     plt.ion()

        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()

        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        # image_list = [] # for visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []
        # with torch.inference_mode():
        with torch.no_grad():
            for t in range(max_timesteps):
                ### update onscreen render and wait for DT
                # if onscreen_render:
                #     image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                #     plt_img.set_data(image)
                #     plt.pause(DT)

                ### process previous timestep to get qpos and image_list
                # obs = ts.observation
                # if 'images' in obs:
                #     image_list.append(obs['images'])
                # else:
                #     image_list.append({'main': obs['image']})
                
                # qpos_numpy = np.array(obs['qpos'])
                qpos_numpy = states
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                qpos_history[:, t] = qpos

                # curr_image = get_image(ts, camera_names)

                ### query policy
                if t % query_frequency == 0:
                    all_actions = policy(qpos, curr_image)
                if temporal_agg:
                    all_time_actions[[t], t:t+num_queries] = all_actions
                    actions_for_curr_step = all_time_actions[:, t]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                else:
                    raw_action = all_actions[:, t % query_frequency]


                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                openaction=hand_action(True)
                closeaction=hand_action(False)
                if action[-1]>0.015:
                    handaction=openaction
                else:
                    handaction=closeaction
                actiontensor=torch.from_numpy(action[:9]).unsqueeze(0)
                target_qpos = torch.cat([actiontensor, handaction], dim=1)

                ### step the environment
                env.step_inference_GR00T(target_qpos)

                # ### for visualization
                # qpos_list.append(qpos_numpy)
                # target_qpos_list.append(target_qpos)
                # rewards.append(ts.reward)
                states, curr_image=getObs(env)
            print('rollout: ', rollout_id)
        # rigid_obj_pos=env.scene.state["rigid_object"]["object"]["root_pose"][:,:3]
        # if rigid_obj_pos[0][2].item()>1.2:
        #     num_success+=1
        #     print('success')
        # else:
        #     print('failed')
        # if save_episode:
    print('success rate: ', num_success/num_rollouts)

def get_image(camera_img):
    curr_images = []
    for img in camera_img.values():
        curr_image = rearrange(img, 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image

def getObs(env):
    obs_buf = env.observation_manager.compute()
    camera_img={'table1_img':obs_buf["policy"]["table1_img"].cpu().numpy(),
         'table2_img':obs_buf["policy"]["table2_img"].cpu().numpy()}
    states=obs_buf["policy"]["states"].cpu().numpy().squeeze(0)

    img=get_image(camera_img)

    return states, img


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
