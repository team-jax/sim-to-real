"""Offline checks of the deployment model contract; never sends motor commands."""

import math
import os
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET


EXPECTED_LIMITS_DEG = {
    "shoulder_pitch_to_shoulder_roll": (-45, 45),
    "shoulder_roll_to_shoulder_yaw": (0, 45),
    "shoulder_yaw_to_upper_arm": (-90, 240),
    "upper_arm_to_under_arm": (-105, 105),
    "under_arm_to_wrist_pitch": (-90, 90),
    "wrist_pitch_to_wrist_yaw": (-90, 90),
    "wrist_yaw_to_hand_base": (-120, 120),
    "hand_base_to_finger_1_1": (-90, 90),
    "hand_base_to_finger_2_1": (-90, 90),
    "hand_base_to_rotation_base": (-180, 0),
    "finger_1_1_to_finger_1_2": (0, 120),
    "finger_1_2_to_finger_2_2": (0, 120),
    "rotation_base_to_finger_3_1": (-90, 90),
    "finger_3_1_to_finger_3_2": (0, 120),
}


class RobotContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configured = os.environ.get("ROBOT_URDF", str(
            Path(__file__).resolve().parents[1] / "models" /
            "JAXV2_Assembly_description/urdf/JAXV2_Assembly.urdf"))
        cls.path = Path(configured).expanduser().resolve()
        cls.robot = ET.parse(cls.path).getroot()

    def test_joint_names_and_limits(self):
        joints = {j.attrib["name"]: j for j in self.robot.findall("joint")
                  if j.get("type") != "fixed"}
        self.assertEqual(set(joints), set(EXPECTED_LIMITS_DEG))
        for name, expected in EXPECTED_LIMITS_DEG.items():
            with self.subTest(joint=name):
                joint = joints[name]
                self.assertEqual(joint.get("type"), "revolute")
                limit = joint.find("limit")
                self.assertIsNotNone(limit)
                for attr, degrees in zip(("lower", "upper"), expected):
                    self.assertAlmostEqual(float(limit.get(attr)), math.radians(degrees), places=5)
                for attr in ("effort", "velocity"):
                    value = float(limit.get(attr))
                    self.assertTrue(math.isfinite(value) and value > 0)

    def test_mass_and_inertia_positive_definite(self):
        for link in self.robot.findall("link"):
            with self.subTest(link=link.get("name")):
                inertial = link.find("inertial")
                self.assertIsNotNone(inertial)
                mass = float(inertial.find("mass").get("value"))
                self.assertTrue(math.isfinite(mass) and mass > 0)
                inertia = inertial.find("inertia")
                a, b, c, d, e, f = [float(inertia.get(k)) for k in
                                    ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")]
                self.assertTrue(all(math.isfinite(v) for v in (a, b, c, d, e, f)))
                self.assertGreater(a, 0)
                self.assertGreater(a*d - b*b, 0)
                self.assertGreater(a*d*f + 2*b*c*e - a*e*e - d*c*c - f*b*b, 0)

    def test_mesh_assets_exist(self):
        missing = []
        for mesh in self.robot.findall(".//mesh"):
            uri = mesh.attrib["filename"]
            relative = uri.removeprefix("package://").removeprefix("file://")
            candidates = [self.path.parent / relative,
                          self.path.parent.parent.parent / relative,
                          self.path.parent / Path(relative).name,
                          self.path.parent / "meshes" / "JAXV2_Assembly" / Path(relative).name]
            if not any(p.is_file() for p in candidates):
                missing.append(uri)
        self.assertFalse(missing, "Missing mesh assets:\n" + "\n".join(sorted(set(missing))))


if __name__ == "__main__":
    unittest.main()
