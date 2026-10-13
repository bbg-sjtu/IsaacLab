# Moz1 Pink whole-body IK demo

This demo controls both official Moz1 wrist/end links (`left07`, `right07`)
while Pink jointly solves:

- virtual planar base: `dummy_x_joint`, `dummy_y_joint`, `dummy_yaw_joint`;
- six `LegWaist_*` joints;
- fourteen `LeftArm_*` / `RightArm_*` joints.

The action has 23 values per environment:

```text
[left position xyz, left quaternion xyzw,
 right position xyz, right quaternion xyzw,
 base/waist posture target (9)]
```

The four wheel joints are fixed. The demo intentionally does not model wheel
kinematics, wheel dynamics, or wheel-ground force transmission.

## Run

Activate the Isaac Lab environment first, then run:

```bash
conda activate env_isaaclab
./isaaclab.sh -p scripts/MyTest/Moz1/moz1_pink_wbc_test.py --viz kit
```

Without `--viz kit`, the demo runs headless. Useful options include `--steps`,
`--period`, `--radius`, `--num_envs`, and `--max_position_error` [m].
Successful runs print `MOZ1_VALIDATION` and measured RMS/maximum wrist errors.

## Regenerate the USD

The local USD is generated from `moz1_wbc.urdf`. Set
`ROS_PACKAGE_PATH` so the importer can resolve the vendor's
`package://spirit01_model/meshes/...` URIs:

```bash
export ROS_PACKAGE_PATH="$PWD/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/Moz1/robot_model"
./isaaclab.sh -p scripts/tools/convert_urdf.py \
  source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/Moz1/robot_model/spirit01_model/urdf/moz1_wbc.urdf \
  source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/Moz1/robot_model/spirit01_model/usd \
  --fix-base --joint-stiffness 4000 --joint-damping 100 --headless
```

See `robot_model/SOURCE.md` for source hashes, adaptations, and the vendor
package's licensing caveat.

The new importer accepts an output **directory**, creating
`usd/moz1_wbc/moz1_wbc.usda` and its dependent layers. Keep the whole directory;
USD artifacts are ignored by the repository's existing Git rules.

## 与 TianJi 方案的对应关系

### Moz1 腿腰软约束

环境显式配置的 `WaistTaskCfg` 通过通用 Pink 控制器尽量满足：
`LegWaist_1 + LegWaist_2 + LegWaist_3 = 0`，且
`LegWaist_1 >= 0`、`LegWaist_2 <= 0`、`LegWaist_3 >= 0`。
三个关节的转轴均为局部 X 轴，该等式抵消这段串联机构的累计弯曲角。
它不限制其他腰部转轴，因而不等价于对上半身世界坐标姿态的完整锁定。

耦合残差为 `q1+q2+q3`；符号残差为
`max(-q1,0)`、`max(q2,0)`、`max(-q3,0)`，只惩罚不符合偏好的方向。
它们与末端和零空间姿态目标一起加权优化，不添加任何 QP 等式或不等式。
原 URDF 位置限位及速度限制仍保留。
不修改公共 URDF、USD 或关节位置控制任务。当前 reset 姿态满足约束。
因此允许为末端跟踪作出折中，也不会仅因无法在一步内恢复这些偏好而报不可行。
所有任务统一在 `moz1_pink_wbc_env_cfg.py` 的 `ActionsCfg.whole_body_ik.controller`
中声明，腰部任务位于 `fixed_input_tasks`：

```python
WaistTaskCfg(
    joint1="LegWaist_1",
    joint2="LegWaist_2",
    joint3="LegWaist_3",
    cost=5.0,                 # 三个角度之和接近零的权重
    sign_cost=5.0,            # 符号违反的权重
    joint_signs=(1, -1, 1),   # 非负、非正、非负；某项为 0 则不限制该关节符号
    gain=0.5,
    lm_damping=0.0,
)
```

`cost` 或 `sign_cost` 设为 0 可分别关闭耦合或符号偏好；删除该配置即可关闭
全部腰部偏好，不再有内部自动追加的任务。例如 `(1.0, -1.5, 0.5)` 的全部误差为零。
这是软目标，不保证任意时刻严格满足等式，也不会把三个角度固定为上述数值。

实现位于公共 Pink 模块的 `waist_task.py`，配置类位于 `pink_task_cfg.py`。
旧 `JointCouplingTaskCfg` 保持原行为，供 TianJi 等仅需角度和约束的应用使用。
旧 action 字段 `waist_coupling_cost` / `waist_sign_cost` 已弃用；兼容入口仅能覆盖
已有显式任务的权重并发出警告，不会创建任务。新代码请仅使用上述配置。

这些是初始权重，不保证严格直立或角度符号。CPU 验证：

```bash
./isaaclab.sh -p -m pytest scripts/MyTest/Moz1/test_moz1_waist_constraints.py -q
```

测试检查软目标允许偏差、增大权重能减小违反量、雅可比正确，
并确认 QP 仅改变目标函数、没有新增硬约束。此前硬约束版本的仿真结果
不作为当前软约束权重的跟踪性能保证。

TianJi 主程序构造双末端位姿与姿态目标，环境的 action term 将它们交给
Pink。Pinocchio 使用 URDF 计算正运动学和雅可比；Pink 在关节限位、速度
限制下求解速度级 QP，再积分成关节位置目标，由仿真的关节 PD 驱动执行。
FrameTask 负责末端跟踪，DampingTask 抑制过大的关节速度，
NullSpacePostureTask 在末端任务的零空间中偏向指定姿态。
这属于全身运动学 IK 加位置伺服，不是包含接触力约束的动力学 WBC。

Moz1 沿用这条控制链，采用 3 个底盘虚拟关节、6 个腿腰关节和 14 个臂关节。
当前 Lab 3 的输入四元数顺序为 **xyzw**（旧 Lab 2 为 wxyz）；位置相对各环境原点，不是相对当前底盘。
适配 action 会进一步转换到固定 URDF 根 `world_base`，处理机器人出生高度
和多环境原点偏移。手腕 `left07` / `right07` 是公开模型的末端，不含额外工具 TCP。

公开 URDF 缺失惯量和有效力矩限制，因此质量由导入器近似计算，力矩上限及
PD 增益仅供仿真；关闭重力和机器人碰撞，不宣称硬件安全或碰撞可行性。
底盘轮子固定，不求解轮速和轮地接触传动。

安装与版本说明见 `VERSION_ALIGNMENT.md`。不要在另一 Git 分支上直接复用
此分支的 editable 安装；切换旧版 Lab 时应使用独立的对应环境。
