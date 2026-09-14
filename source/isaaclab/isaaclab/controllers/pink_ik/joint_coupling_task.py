# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Joint-coupling task for Pink inverse kinematics."""

from __future__ import annotations

import numpy as np
from pink.configuration import Configuration
from pink.tasks import Task


class JointCouplingTask(Task):
    r"""Softly enforce :math:`q_1 + q_2 + q_3 = 0` for scalar joints."""

    def __init__(
        self,
        cfg_or_joint1,
        joint2: str | None = None,
        joint3: str | None = None,
        cost: float = 1.0,
        gain: float = 0.5,
        lm_damping: float = 1.0,
    ) -> None:
        """Initialize from a deferred task config or explicit joint names.

        Args:
            cfg_or_joint1: A joint-coupling task config, or the first joint name.
            joint2: Second scalar joint name when using explicit arguments.
            joint3: Third scalar joint name when using explicit arguments.
            cost: Task cost.
            gain: Task gain.
            lm_damping: Levenberg-Marquardt damping.
        """
        if isinstance(cfg_or_joint1, str):
            joint1 = cfg_or_joint1
        else:
            cfg = cfg_or_joint1
            joint1 = cfg.joint1
            joint2 = cfg.joint2
            joint3 = cfg.joint3
            cost = cfg.cost
            gain = cfg.gain
            lm_damping = cfg.lm_damping

        if joint2 is None or joint3 is None:
            raise ValueError("joint2 and joint3 must be provided")

        super().__init__(cost=float(cost), gain=float(gain), lm_damping=float(lm_damping))
        self.joint1 = joint1
        self.joint2 = joint2
        self.joint3 = joint3
        self._q_indices: tuple[int, int, int] | None = None
        self._v_indices: tuple[int, int, int] | None = None

    def _build_joint_indices(self, configuration: Configuration) -> None:
        """Resolve and cache scalar configuration and velocity indices."""
        model = configuration.model
        joint_ids = tuple(model.getJointId(name) for name in (self.joint1, self.joint2, self.joint3))
        for name, joint_id in zip((self.joint1, self.joint2, self.joint3), joint_ids):
            if joint_id == 0:
                raise ValueError(f"Joint '{name}' not found in Pinocchio model.")
            if model.nqs[joint_id] != 1 or model.nvs[joint_id] != 1:
                raise ValueError(f"JointCouplingTask only supports scalar 1-DoF joints; '{name}' is not scalar.")

        self._q_indices = tuple(model.idx_qs[joint_id] for joint_id in joint_ids)
        self._v_indices = tuple(model.idx_vs[joint_id] for joint_id in joint_ids)

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        """Return the scalar coupling residual."""
        if self._q_indices is None:
            self._build_joint_indices(configuration)
        assert self._q_indices is not None
        return np.array([sum(configuration.q[index] for index in self._q_indices)], dtype=float)

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        """Return the residual Jacobian in Pinocchio velocity order."""
        if self._v_indices is None:
            self._build_joint_indices(configuration)
        assert self._v_indices is not None
        jacobian = np.zeros((1, configuration.model.nv))
        jacobian[0, list(self._v_indices)] = 1.0
        return jacobian

    def set_target_from_configuration(self, configuration: Configuration) -> None:
        """Keep the fixed zero-residual target; no state-dependent target is required."""
        return None
