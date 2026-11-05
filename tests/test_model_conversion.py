"""Independent URDF FK comparison and visual/contact invariance checks."""
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from sim_validation.model import DEFAULT_URDF, load_robot, contact_pairs
from sim_validation.inspect_model import POSES, set_pose


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle)*skew + (1-np.cos(angle))*(skew @ skew)


def urdf_fk(root, angles):
    joints = root.findall("joint")
    children = {j.find("child").get("link") for j in joints}
    transforms = {link.get("name"): np.eye(4) for link in root.findall("link")
                  if link.get("name") not in children}
    pending = joints.copy()
    while pending:
        progressed = False
        for joint in pending.copy():
            parent = joint.find("parent").get("link")
            if parent not in transforms:
                continue
            origin = joint.find("origin")
            xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
            rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
            r, p, y = rpy
            frame = np.eye(4)
            frame[:3, :3] = rotation([0, 0, 1], y) @ rotation([0, 1, 0], p) @ rotation([1, 0, 0], r)
            frame[:3, 3] = xyz
            if joint.get("type") != "fixed":
                axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                frame[:3, :3] = frame[:3, :3] @ rotation(axis, angles[joint.get("name")])
            transforms[joint.find("child").get("link")] = transforms[parent] @ frame
            pending.remove(joint)
            progressed = True
        if not progressed:
            raise ValueError("Disconnected URDF tree")
    return transforms


class ConversionTests(unittest.TestCase):
    def test_link_frames_match_urdf_at_three_poses(self):
        model = load_robot(visuals=True)
        data = mujoco.MjData(model)
        root = ET.parse(DEFAULT_URDF).getroot()
        for pose in POSES:
            set_pose(model, data, pose)
            angles = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j):
                      data.qpos[model.jnt_qposadr[j]] for j in range(model.njnt)}
            for name, transform in urdf_fk(root, angles).items():
                with self.subTest(pose=pose, link=name):
                    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                    self.assertGreaterEqual(body, 0)
                    np.testing.assert_allclose(data.xpos[body], transform[:3, 3], atol=1e-8)
                    np.testing.assert_allclose(data.xmat[body].reshape(3, 3), transform[:3, :3], atol=1e-8)

    def test_visuals_do_not_change_collision_or_mass(self):
        models = [load_robot(visuals=v) for v in (False, True)]
        snapshots = []
        for model in models:
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            snapshots.append(sorted(contact_pairs(model, data)))
        self.assertEqual(snapshots[0], snapshots[1])
        np.testing.assert_allclose(models[0].body_mass, models[1].body_mass)
        visual_model = models[1]
        visual_geoms = visual_model.geom_group == 1
        self.assertEqual(np.count_nonzero(visual_geoms), 16)
        self.assertTrue(np.all(visual_model.geom_contype[visual_geoms] == 0))
        self.assertTrue(np.all(visual_model.geom_conaffinity[visual_geoms] == 0))


if __name__ == "__main__":
    unittest.main()
