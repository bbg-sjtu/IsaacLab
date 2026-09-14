import numpy as np
from pink.tasks import Task


class JointCouplingTask(Task):
    """
    Soft joint coupling task for Pink IK.

    Desired constraint:
        joint2 = -joint1 - joint3

    Equivalent residual:
        e(q) = q1 + q2 + q3

    The IK solver will try to drive:
        e(q) -> 0
    """

    def __init__(
        self,
        joint1: str,
        joint2: str,
        joint3: str,
        cost: float = 1.0,
        gain: float = 0.5,
        lm_damping: float = 1.0,
    ):
        super().__init__(
            cost=cost,
            gain=gain,
            lm_damping=lm_damping,
        )

        self.joint1 = joint1
        self.joint2 = joint2
        self.joint3 = joint3

        self._idx1 = None
        self._idx2 = None
        self._idx3 = None

    def _build_joint_indices(self, configuration):
        """
        Build q / v indices for scalar 1-DoF joints.

        This assumes joint1, joint2, joint3 are ordinary scalar joints,
        e.g. revolute or prismatic joints. For this case, nq index and nv
        index are both one-dimensional and aligned through Pinocchio's
        model.idx_qs / model.idx_vs.
        """
        model = configuration.model

        joint_id1 = model.getJointId(self.joint1)
        joint_id2 = model.getJointId(self.joint2)
        joint_id3 = model.getJointId(self.joint3)

        if joint_id1 == 0:
            raise ValueError(f"Joint '{self.joint1}' not found in Pinocchio model.")
        if joint_id2 == 0:
            raise ValueError(f"Joint '{self.joint2}' not found in Pinocchio model.")
        if joint_id3 == 0:
            raise ValueError(f"Joint '{self.joint3}' not found in Pinocchio model.")

        nq1 = model.nqs[joint_id1]
        nq2 = model.nqs[joint_id2]
        nq3 = model.nqs[joint_id3]

        nv1 = model.nvs[joint_id1]
        nv2 = model.nvs[joint_id2]
        nv3 = model.nvs[joint_id3]

        if not (nq1 == nq2 == nq3 == 1 and nv1 == nv2 == nv3 == 1):
            raise ValueError(
                "JointCouplingTask currently only supports scalar 1-DoF joints. "
                f"Got nq=({nq1}, {nq2}, {nq3}), nv=({nv1}, {nv2}, {nv3})."
            )

        self._q_idx1 = model.idx_qs[joint_id1]
        self._q_idx2 = model.idx_qs[joint_id2]
        self._q_idx3 = model.idx_qs[joint_id3]

        self._v_idx1 = model.idx_vs[joint_id1]
        self._v_idx2 = model.idx_vs[joint_id2]
        self._v_idx3 = model.idx_vs[joint_id3]

    def compute_error(self, configuration):
        """
        Error:
            e(q) = q1 + q2 + q3

        Desired:
            e(q) = 0
        which means:
            q2 = -q1 - q3
        """
        if self._idx_not_ready():
            self._build_joint_indices(configuration)

        q = configuration.q

        q1 = q[self._q_idx1]
        q2 = q[self._q_idx2]
        q3 = q[self._q_idx3]

        error = q1 + q2 + q3

        return np.array([error], dtype=float)

    def compute_jacobian(self, configuration):
        """
        Jacobian of:
            e(q) = q1 + q2 + q3

        Therefore:
            de/dq1 = 1
            de/dq2 = 1
            de/dq3 = 1
        """
        if self._idx_not_ready():
            self._build_joint_indices(configuration)

        J = np.zeros((1, configuration.model.nv))

        J[0, self._v_idx1] = 1.0
        J[0, self._v_idx2] = 1.0
        J[0, self._v_idx3] = 1.0

        return J

    def _idx_not_ready(self):
        return not hasattr(self, "_q_idx1")
    
    def set_target_from_configuration(self, configuration) -> None:
        pass