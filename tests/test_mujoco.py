"""Model loading and bounded dynamics checks, not policy or hardware approval."""
import os
import unittest

import mujoco
import numpy as np

from sim_validation.model import DEFAULT_URDF, contact_pairs, load_robot
from test_urdf import EXPECTED_LIMITS_DEG


class MujocoModelTests(unittest.TestCase):
    def setUp(self):
        self.model = load_robot(os.environ.get("ROBOT_URDF", DEFAULT_URDF))
        self.data = mujoco.MjData(self.model)

    def test_compiled_joints_and_limits(self):
        m = self.model
        self.assertEqual(m.nq, 14)
        self.assertEqual(m.nv, 14)
        names = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)}
        self.assertEqual(names, set(EXPECTED_LIMITS_DEG))
        for name, limits in EXPECTED_LIMITS_DEG.items():
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.assertTrue(m.jnt_limited[j])
            np.testing.assert_allclose(m.jnt_range[j], np.deg2rad(limits), atol=1e-6)

    def test_zero_pose_has_no_deep_self_penetration(self):
        mujoco.mj_forward(self.model, self.data)
        deep = [p for p in contact_pairs(self.model, self.data) if p[2] < -0.001]
        self.assertFalse(deep, f"Zero-pose penetration >1 mm: {deep}")

    def test_each_joint_small_motion_without_contact(self):
        # Isolate axes/dynamics from contact; contact is checked separately above.
        m, d = self.model, self.data
        m.opt.gravity[:] = 0
        m.geom_contype[:] = 0
        m.geom_conaffinity[:] = 0
        for j in range(m.njnt):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
            with self.subTest(joint=name):
                mujoco.mj_resetData(m, d)
                lo, hi = m.jnt_range[j]
                d.qpos[:] = np.clip(np.zeros(m.nq), m.jnt_range[:, 0], m.jnt_range[:, 1])
                start = d.qpos.copy()
                target = start.copy()
                target[j] = np.clip(start[j] + np.deg2rad(2), lo, hi)
                for _ in range(240):
                    mujoco.mj_forward(m, d)
                    # Computed-torque test driver, not a model of the real servo.
                    acceleration = 100 * (target - d.qpos) - 20 * d.qvel
                    mass = np.zeros((m.nv, m.nv))
                    mujoco.mj_fullM(m, d, mass)
                    d.qfrc_applied[:] = mass @ acceleration + d.qfrc_bias
                    mujoco.mj_step(m, d)
                    self.assertTrue(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all())
                    self.assertLess(float(np.max(np.abs(d.qvel))), 1.0)
                    self.assertTrue(np.all(d.qpos >= m.jnt_range[:, 0] - 0.001))
                    self.assertTrue(np.all(d.qpos <= m.jnt_range[:, 1] + 0.001))
                np.testing.assert_allclose(d.qpos, target, atol=np.deg2rad(0.1))


if __name__ == "__main__":
    unittest.main()
