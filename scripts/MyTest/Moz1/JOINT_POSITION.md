# Moz1 关节位置控制

任务：`Isaac-Moz1-Joint-Position-v0`。
配置：`moz1_joint_pos_env_cfg.py`；主程序：`moz1_joint_pos_test.py`。
复用现有 Moz1 USD、虚拟底盘和 PD 执行器，不调用 Pink IK。
仍是无重力、无机器人碰撞的演示场景，不是实机控制器。

```bash
conda activate env_isaaclab
./isaaclab.sh -p scripts/MyTest/Moz1/moz1_joint_pos_test.py --viz kit
```

直接编辑主程序顶部的 `JOINT_TARGETS` 列表，每个值旁都注明了对应关节。
当前 23 个关节目标均与 reset 默认位置一致，用于保持初始姿态。
每一步直接发送相同 action，不做正弦运动或插值。
省略 `--viz kit` 时无窗口运行。

## action 定义

形状为 `(num_envs, 23)`，各行控制一个机器人。
`scale=1`、`offset=0`、`use_default_offset=False`，所以
**action 就是绝对关节目标值**，不是增量、速度或归一化动作。
关节 PD 负责跟踪，机器人不会瞬间到达目标。

| action 索引 | 关节 | 单位 |
| --- | --- | --- |
| 0、1 | dummy_x_joint、dummy_y_joint | m |
| 2 | dummy_yaw_joint | rad |
| 3–8 | LegWaist_0 … LegWaist_5 | rad |
| 9–15 | LeftArm_0 … LeftArm_6 | rad |
| 16–22 | RightArm_0 … RightArm_6 | rad |

配置用 `preserve_order=True` 固定上述顺序，不依赖 USD 内部交错的左右臂顺序。
观测 `policy.joint_pos` 也采用相同顺序。

在主程序中直接设置即可：

```python
action = torch.tensor(JOINT_TARGETS, device=env.device).repeat(env.num_envs, 1)
action[:, 1] = 0.1    # 底盘 y = 0.1 m
action[:, 9] = 1.2    # 左臂第 0 关节 = 1.2 rad
env.step(action)
```

不再使用 `--joint_targets`、`--ramp_time`、`--period`、`--amplitude` 参数。
示例会拒绝非有限值和超出关节位置限位的目标。直接指定目标会产生目标阶跃，
适用于此仿真演示；不能直接作为实机的安全轨迹控制方案。
