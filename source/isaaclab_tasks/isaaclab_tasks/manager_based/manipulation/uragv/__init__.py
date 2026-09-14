# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
import os

from . import uragv_env_cfg

gym.register(
    id="Isaac-UR-AGV-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": uragv_env_cfg.URAGVEnvCfg,
    },
    disable_env_checker=True,
)