
import argparse

from isaaclab.app import AppLauncher

import pandas as pd
import time
import h5py
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
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms
from IKCtrl import URAGV_IK
from pickplace_action import getPickplaceAction
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
    dataset_dir='/WorkSpace/LiuBang/IsaacLab/videoOutput/data_for_uragv'
    # camera_names=['top_img', 'table1_img', 'table2_img', 'wrist_img']
    camera_names=['table1_img', 'table2_img']
    num_episodes=50
    i=0
    while i<num_episodes:
        i+=1
        data_dict = {
            '/observations/qpos': [],
            '/action': [],
            # '/observations/images/top_img': [],
            '/observations/images/table1_img': [],
            '/observations/images/table2_img': [],
            # '/observations/images/wrist_img': [],
        }
        env.reset()
        actions=torch.tensor([[0.0,0.0,0.0,3.14,-1.57,1.57,-3.14,-1.57,3.14,0.03,0.03]])
        for k in range(10):
            env.step(actions)

        uragv_ik=URAGV_IK(env)
        rigid_obj_pos=env.scene.state["rigid_object"]["box"]["root_pose"][:,:3]
        container_pos=env.scene.state["rigid_object"]["container"]["root_pose"][:,:3]
        raw_actions=getPickplaceAction(rigid_obj_pos,container_pos)
        index=0
        should_break=False
        print(f'episode {i-1} start!')
        for k in range(8):
            action=raw_actions[k]
            if k==0 or k==6:
                for j in range(130):
                    data_dict,index=env.step_record_URAGV(action,data_dict,index)
            elif k==1 or k==2 or k==3 or k==4:
                uragv_ik.set_goal(action)
                for j in range(20):
                    actions=uragv_ik.action_compute()
                    data_dict,index=env.step_record_URAGV(actions,data_dict,index)
            elif k==5:
                tmp_action[:,6]=-2.5
                for j in range(50):
                    data_dict,index=env.step_record_URAGV(tmp_action,data_dict,index)
            elif k==7:
                for j in range(20):
                    data_dict,index=env.step_record_URAGV(action,data_dict,index)
            if k==4:
                tmp_action=actions.clone()
            # 判断是否失败
            if k==5 or k==6:
                # 检查物体位置是否被拿起，否则丢弃本次记录
                rigid_obj_pos=env.scene.state["rigid_object"]["box"]["root_pose"][:,:3]
                if rigid_obj_pos[0][2].item()<0.8:
                    print(f'episode {i-1} failed at step {k}, discard this episode!')
                    i-=1
                    should_break=True
                    break
        if should_break:
            continue
        else:
            # 保存HDF5
            max_timesteps=index
            t0 = time.time()
            dataset_path = os.path.join(dataset_dir, f'episode_{i-1}')
            with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
                root.attrs['sim'] = True
                obs = root.create_group('observations')
                image = obs.create_group('images')
                for cam_name in camera_names:
                    _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                            chunks=(1, 480, 640, 3), )
                qpos = obs.create_dataset('qpos', (max_timesteps, 11))
                action = root.create_dataset('action', (max_timesteps, 11))

                for name, array in data_dict.items():
                    root[name][...] = array
            print(f'episode {i-1} success, saving: {time.time() - t0:.1f} secs\n')
    
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
