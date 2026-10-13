# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

gym.register(
    id="Isaac-TJ-L-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tj_l_env_cfg:TJ_L_EnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-TJ-LR-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tj_lr_pink_env_cfg:TJ_LR_EnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-TJ-LR-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tj_lr_pink_NullSpaceDance_env_cfg:TJ_LR_NullSpaceDance_EnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-TJ-MM-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tj_MM_env_cfg:TJ_MM_EnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-TJ-MM-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tj_MM_pink_env_cfg:TJ_MM_PINK_EnvCfg",
    },
    disable_env_checker=True,
)
