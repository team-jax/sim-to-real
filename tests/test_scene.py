import unittest
import numpy as np

from sim_validation.play_scene import SceneController
from sim_validation.model import DEFAULT_URDF
from sim_validation.scene import LAYOUTS
import trimesh


class SceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = SceneController(dx=-0.03, dy=0.02)

    def setUp(self):
        self.scene.reset()

    def test_placement_and_base_orientation(self):
        data = self.scene.data
        np.testing.assert_allclose(data.body("cube").xpos, [-0.315, 0.28, 0.760])
        np.testing.assert_allclose(data.body("base_link").xpos, [0, 0, 0.305])
        rotation = data.body("base_link").xmat.reshape(3, 3)
        np.testing.assert_allclose(rotation @ [1, 0, 0], [0, -1, 0], atol=1e-8)

    def test_joint_command_validation(self):
        with self.assertRaises(ValueError):
            self.scene.command_joint("unknown", 0)
        with self.assertRaises(ValueError):
            self.scene.command_joint("upper_arm_to_under_arm", 999)
        with self.assertRaises(ValueError):
            self.scene.command_joint("upper_arm_to_under_arm", float("nan"))

    def test_cube_rests_and_named_joint_moves(self):
        name = "shoulder_yaw_to_upper_arm"
        self.scene.command_joint(name, 5)
        for _ in range(120):
            self.scene.step()
        np.testing.assert_allclose(self.scene.data.body("cube").xpos,
                                   [-0.315, 0.28, 0.760], atol=0.001)
        self.assertGreater(self.scene.data.joint(name).qpos[0], np.deg2rad(1))
        self.assertTrue(np.isfinite(self.scene.data.qpos).all())

    def test_measured_layouts(self):
        mesh = DEFAULT_URDF.parent.parent / "meshes/JAXV2_Assembly/base_link.dae"
        base = trimesh.load(mesh, force="scene", process=False).to_geometry()
        # yaw -90: local y becomes world x; this is the modeled pedestal edge.
        base_edge = base.bounds[0, 1]
        self.assertAlmostEqual(base_edge - LAYOUTS["edge10"][1], 0.1, places=6)
        self.assertAlmostEqual(-LAYOUTS["center10"][1], 0.1)
        for layout in ("center10", "edge10"):
            with self.subTest(layout=layout):
                scene = SceneController(layout=layout)
                table_id = scene.model.geom("table").id
                top = scene.model.geom_pos[table_id, 2] + scene.model.geom_size[table_id, 2]
                self.assertAlmostEqual(top, 0.735)
                for _ in range(60):
                    scene.step()
                cube = scene.data.body("cube").xpos
                self.assertAlmostEqual(cube[0], LAYOUTS[layout][1] - 0.06, places=5)
                self.assertAlmostEqual(cube[2], 0.735 + 0.020, delta=0.001)

    def test_requested_cube_properties(self):
        model = self.scene.model
        geom = model.geom("cube_geom").id
        np.testing.assert_allclose(model.geom_size[geom], [0.02]*3)
        self.assertAlmostEqual(model.body("cube").mass[0], 0.05)
        self.assertAlmostEqual(model.geom_friction[geom, 0], 0.35)

    def test_runtime_table_contact_matches_physx_dynamic_mean(self):
        self.scene.step()
        model, data = self.scene.model, self.scene.data
        table, cube = model.geom("table").id, model.geom("cube_geom").id
        contacts = [c for c in data.contact if set(c.geom)=={table,cube}]
        self.assertTrue(contacts)
        for contact in contacts:
            np.testing.assert_allclose(contact.friction[:2],[0.675]*2)


if __name__ == "__main__":
    unittest.main()
