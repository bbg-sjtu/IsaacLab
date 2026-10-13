# 本地版本对齐记录（2026-09-14）

状态：依赖检查、CPU IK、双环境 GPU 跟踪均已通过。
这是本机验证的组合，不代表全部旧应用已完成回归。

## Git 与环境

- 当前分支：`codex/isaacsim-6.0-aligned`。
- 基线：`v3.0.0-beta2`，`28a37cecdd433c22d9eabd6a5954add9f13a8951`。
- TianJi/URAGV 迁移提交：`5733bef10`；后续修复与 Moz1 文件尚未提交或推送。
- 原分支 `feature/bbg-isaaclab` 保留在 `c5428e140d235b571431d153f8b147536be4d778`。
- Python：`/home/bbgsjtu/anaconda3/envs/env_isaaclab/bin/python`，3.12.14。

| 组件 | 对齐版本 |
| --- | --- |
| Isaac Sim | 6.0.0.1；实际二进制为 6.0.0-rc.59 |
| Isaac Lab editable | 6.1.11（此 Git 基线的内部包版本） |
| PyTorch / torchvision / torchaudio | 2.11.0 / 0.26.0 / 2.11.0，cu128 |
| NumPy | 2.3.1 |
| usd-exchange / OpenUSD | 3.0.0 / 26.8；不再同时安装 usd-core |
| Pinocchio / Pink / DAQP | 4.1.0 / 3.1.0 / 0.8.5 |

## 修复内容

- 安装器遵循已安装 Sim wheel 的 PyTorch 依赖，不再降级至 2.10。
- 移除重叠的 usd-core 25.11，重装 usd-exchange，统一 pxr/native 库。
- 放宽 Lab 的 coverage、packaging 限定以兼容 Sim 固定版本。
- 移除旧 hpp-fcl 2.4.4；当前 Pinocchio 使用 coal，旧包与新版 cmeel 冲突。
- 对齐 NumPy、llvmlite、click、psutil，以及相关的 dex-retargeting、
  moviepy、IPython、Numba、wandb；具体版本见约束文件。
- 适配新版 task 配置、ProxyArray 的 `.torch` 访问及 xyzw 四元数。
- Moz1 管理器使用空配置而非 `None`，并添加场景重置事件。
- 关闭机器人碰撞及场景重力，取消轮地接触对虚拟底盘关节的约束。
- 在关闭 Sim 前输出异常并保留失败退出码，避免测试假成功。

## 验证

- `pip check`：`No broken requirements found.`
- 90 项相关 CPU 回归测试通过；新增安装器测试在移除修复后确认失败。
- 官方 URDF 导入成功，生成约 19 MB USD 层文件，23 个活动关节。
- CPU 双手静态目标：3 cm 位移收敛至小于 1 mm。
- GPU 双环境、600 步（5 s）、半径 0.1 m 轨迹：位置 RMS 误差
  **0.013843 m**，最大误差 **0.019978 m**，关节状态有限。
- 新增 Moz1 文件 Ruff 检查及 `git diff --check` 通过。
- `./isaaclab.sh -d` 已运行至源码索引阶段；为限制收尾开销已中止全量构建，
  不将生成中的文档视为构建通过。详见 `logs/moz1_alignment/docs.log`。
- 全仓 `./isaaclab.sh -f` 已运行，但历史 TianJi 等文件仍有静态检查问题；
  未称其为通过，并撤回无关文件的自动格式改动。
- TianJi 原主文件已转换保存的 wxyz 目标和旧张量访问；未进行完整 TianJi 仿真回归。

日志及修复前后包清单位于 `logs/moz1_alignment/`，不是完整环境备份。
切回旧 Git 分支时应使用另一兼容环境，不要复用当前 editable 安装。

## 复测

```bash
conda activate env_isaaclab
./isaaclab.sh -p -m pip check
./isaaclab.sh -p scripts/MyTest/Moz1/verify_moz1_ik.py
./isaaclab.sh -p scripts/MyTest/Moz1/moz1_pink_wbc_test.py \
  --steps 600 --num_envs 2 --max_position_error 0.05
```

## 重装本地组合

只用于本分支；约束文件防止版本漂移，但不是跨平台完整锁文件。
不要另行安装 usd-core；若已存在，先卸载，再强制重装 usd-exchange。
此命令假定其他 Lab 子包已按当前源码 editable 安装。

```bash
./isaaclab.sh -p -m pip install \
  -e source/isaaclab -e source/isaaclab_rl \
  'isaacsim[all,extscache]==6.0.0.1' dex-retargeting ipython numba wandb \
  -c scripts/MyTest/Moz1/constraints-sim6.txt \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.nvidia.com
```

本次以 wheel 元数据、`pip check` 和实际仿真交叉验证兼容性。
