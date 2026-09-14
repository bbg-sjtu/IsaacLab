import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms
import torch


class TJ_IK():
    def __init__(self,env):
        self.env=env
        self.robot = env.scene["robot"]
        diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="pinv")
        self.diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=1, device=env.device)
        frame_marker_cfg = FRAME_MARKER_CFG.copy()
        frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        self.ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
        self.goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))
        self.ik_commands = torch.zeros(1, self.diff_ik_controller.action_dim, device=env.device)
        self.parse_entity(self.env)


    def parse_entity(self,env):
        self.robot_entity_cfg = SceneEntityCfg("robot", 
                                    joint_names=[
                                        "Joint.*",
                                    ], 
                                    body_names=[
                                        "ee_link"
                                        ])
        self.robot_entity_cfg.resolve(env.scene)
        
        if self.robot.is_fixed_base:
            self.ee_jacobi_idx = self.robot_entity_cfg.body_ids[0]-1
        else:
            self.ee_jacobi_idx = self.robot_entity_cfg.body_ids[0]

    def set_goal(self,ee_goal):

        self.ik_commands[:] = ee_goal
        self.diff_ik_controller.reset()
        self.diff_ik_controller.set_command(self.ik_commands)        
        self.goal_marker.visualize(self.ik_commands[:, 0:3] + self.env.scene.env_origins, self.ik_commands[:, 3:7])

    def action_compute(self,):
        jacobian_all=self.robot.root_physx_view.get_jacobians()
        jacobian = jacobian_all[:, self.ee_jacobi_idx, :, self.robot_entity_cfg.joint_ids]
        ee_pose_w = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]

        joint_pos = self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]
        # compute the joint commands
        joint_pos_des = self.diff_ik_controller.compute(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7], jacobian, joint_pos)
        return joint_pos_des