# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configurable soft waist coupling and joint-direction preferences for Pink."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pink.configuration import Configuration

from .joint_coupling_task import JointCouplingTask

if TYPE_CHECKING:
    from .pink_task_cfg import WaistTaskCfg


class WaistTask(JointCouplingTask):
    """Penalize the joint sum and violations of configured joint signs [rad].

    For preferred signs ``s``, the residual is
    ``[q1 + q2 + q3, max(-s1*q1, 0), max(-s2*q2, 0), max(-s3*q3, 0)]``.
    The Jacobian uses a zero subgradient at the sign boundary. This is a soft
    objective and does not change the QP equality or inequality constraints.
    """

    def __init__(self, cfg: WaistTaskCfg) -> None:
        """Initialize the task exclusively from its explicit configuration.

        Args:
            cfg: Joint names, direction preferences, weights [cost/rad], and gains.
        """
        costs = np.array([cfg.cost, cfg.sign_cost, cfg.sign_cost, cfg.sign_cost], dtype=float)
        if not np.isfinite(costs).all() or np.any(costs < 0):
            raise ValueError("Waist costs must be finite and nonnegative.")
        signs = np.asarray(cfg.joint_signs, dtype=float)
        if signs.shape != (3,) or not np.isin(signs, (-1, 0, 1)).all():
            raise ValueError("joint_signs must contain three values from {-1, 0, 1}.")
        if len({cfg.joint1, cfg.joint2, cfg.joint3}) != 3:
            raise ValueError("WaistTask requires three distinct joint names.")
        if not np.isfinite(cfg.gain) or not 0 <= cfg.gain <= 1:
            raise ValueError("Waist gain must be finite and between zero and one.")
        if not np.isfinite(cfg.lm_damping) or cfg.lm_damping < 0:
            raise ValueError("Waist lm_damping must be finite and nonnegative.")
        super().__init__(cfg)
        self.cost = costs
        self._signs = signs

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        """Return the sum and three sign-violation residuals [rad]."""
        coupling_error = super().compute_error(configuration)
        waist = configuration.q[list(self._q_indices)]
        return np.concatenate((coupling_error, np.maximum(-self._signs * waist, 0.0)))

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        """Return the residual Jacobian [rad/rad] in model velocity order."""
        coupling_jacobian = super().compute_jacobian(configuration)
        waist = configuration.q[list(self._q_indices)]
        active = -self._signs * waist > 0.0
        selection = np.eye(configuration.model.nv)[list(self._v_indices)]
        return np.vstack((coupling_jacobian, (-self._signs * active)[:, None] * selection))
