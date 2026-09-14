# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import ActionManager


def object_obs(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """
    Object observations (in world frame):
        object pos,
        object quat,
        left_eef to object,
        right_eef_to object,
    """

    body_pos_w = env.scene["robot"].data.body_pos_w
    left_eef_idx = env.scene["robot"].data.body_names.index("left_hand_roll_link")
    right_eef_idx = env.scene["robot"].data.body_names.index("right_hand_roll_link")
    left_eef_pos = body_pos_w[:, left_eef_idx] - env.scene.env_origins
    right_eef_pos = body_pos_w[:, right_eef_idx] - env.scene.env_origins

    object_pos = env.scene["object"].data.root_pos_w - env.scene.env_origins
    object_quat = env.scene["object"].data.root_quat_w

    left_eef_to_object = object_pos - left_eef_pos
    right_eef_to_object = object_pos - right_eef_pos

    return torch.cat(
        (
            object_pos,
            object_quat,
            left_eef_to_object,
            right_eef_to_object,
        ),
        dim=1,
    )


def get_left_eef_pos(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    body_pos_w = env.scene["robot"].data.body_pos_w
    left_eef_idx = env.scene["robot"].data.body_names.index("left_hand_roll_link")
    left_eef_pos = body_pos_w[:, left_eef_idx] - env.scene.env_origins

    return left_eef_pos


def get_left_eef_quat(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    body_quat_w = env.scene["robot"].data.body_quat_w
    left_eef_idx = env.scene["robot"].data.body_names.index("left_hand_roll_link")
    left_eef_quat = body_quat_w[:, left_eef_idx]

    return left_eef_quat


def get_right_eef_pos(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    body_pos_w = env.scene["robot"].data.body_pos_w
    right_eef_idx = env.scene["robot"].data.body_names.index("right_hand_roll_link")
    right_eef_pos = body_pos_w[:, right_eef_idx] - env.scene.env_origins

    return right_eef_pos


def get_right_eef_quat(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    body_quat_w = env.scene["robot"].data.body_quat_w
    right_eef_idx = env.scene["robot"].data.body_names.index("right_hand_roll_link")
    right_eef_quat = body_quat_w[:, right_eef_idx]

    return right_eef_quat


def get_hand_state(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    hand_joint_states = env.scene["robot"].data.joint_pos[:, -22:]  # Hand joints are last 22 entries of joint state

    return hand_joint_states


def get_head_state(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    robot_joint_names = env.scene["robot"].data.joint_names
    head_joint_names = ["head_pitch_joint", "head_roll_joint", "head_yaw_joint"]
    indexes = torch.tensor([robot_joint_names.index(name) for name in head_joint_names], dtype=torch.long)
    head_joint_states = env.scene["robot"].data.joint_pos[:, indexes]

    return head_joint_states


def get_all_robot_link_state(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    body_pos_w = env.scene["robot"].data.body_link_state_w[:, :, :]
    all_robot_link_pos = body_pos_w

    return all_robot_link_pos

def get_top_cam(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    cam = env.scene["camera_top"].data.output["rgb"].squeeze(0)
    return cam

def get_table1_cam(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    cam = env.scene["camera_1"].data.output["rgb"].squeeze(0)
    return cam

def get_table2_cam(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    cam = env.scene["camera_2"].data.output["rgb"].squeeze(0)
    return cam

def get_wrist_cam(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    cam = env.scene["camera_wrist"].data.output["rgb"].squeeze(0)
    return cam

def get_joint_pos_uragv(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    joint_pos = env.scene["robot"].data.joint_pos
    return joint_pos

def get_cam_head_rgb(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """
    Get the RGB image from the camera mounted on the head of the robot.
    """
    cam_head_rgb = env.scene["camera_head"].data.output["rgb"].squeeze(0)
    return cam_head_rgb

def get_cam_right_wrist_rgb(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """
    Get the RGB image from the camera mounted on the right wrist of the robot.
    """
    cam_right_wrist_rgb = env.scene["camera_right_wrist"].data.output["rgb"].squeeze(0)
    return cam_right_wrist_rgb

def get_cam_left_wrist_rgb(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """
    Get the RGB image from the camera mounted on the left wrist of the robot.
    """
    cam_left_wrist_rgb = env.scene["camera_left_wrist"].data.output["rgb"].squeeze(0)
    return cam_left_wrist_rgb

def get_right_hand_open_state(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    states1 = env.scene["robot"].data.joint_pos[:, -17:-12]  # Hand joints are last 22 entries of joint state
    states2 = env.scene["robot"].data.joint_pos[:, -7:-2]
    states3 = env.scene["robot"].data.joint_pos[:, -1:]
    right_hand_open_states = torch.cat([states1, states2, states3],dim=1)
    if (states1[:,0:4]<-0.1).all().item() and (states2[:,0:4]<-0.1).all().item():
        return torch.tensor([[0]]).to(env.device)
    else:
        return torch.tensor([[1]]).to(env.device)

def get_pink_state(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    right_hand_pos=get_right_eef_pos(env)
    right_hand_quat=get_right_eef_quat(env)
    right_hand_open_state=get_right_hand_open_state(env)
    pink_state=torch.cat([right_hand_pos, right_hand_quat, right_hand_open_state],dim=1)
    return pink_state.squeeze(0)