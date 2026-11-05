"""Real artifact checks and exported policy/backend parity."""
import unittest
import numpy as np
import torch

from deploy.policies import Policy, DEFAULT_ARTIFACTS
from deploy.trajectory import Grid
from deploy.observations import adc, pad_measurements
from deploy.controller import GraspController


class PolicyTests(unittest.TestCase):
    def test_onnx_matches_torchscript(self):
        rng = np.random.default_rng(19)
        for name, width in [("raise_v25_1197",54),("grasp_lift_13075",43)]:
            with self.subTest(policy=name):
                policy = Policy(DEFAULT_ARTIFACTS/(name+".onnx"),width)
                reference = torch.jit.load(str(DEFAULT_ARTIFACTS/(name+".pt")),map_location="cpu").eval()
                samples = np.vstack([np.zeros((1,width)),rng.uniform(-1,1,(8,width))]).astype(np.float32)
                with torch.inference_mode():
                    expected = reference(torch.from_numpy(samples)).numpy()
                actual = policy.session.run(None,{policy.input:samples})[0]
                np.testing.assert_allclose(actual,expected,atol=1e-5,rtol=1e-4)
                first = policy(samples[0])
                np.testing.assert_allclose(first,0.25*np.clip(actual[0],-1,1),atol=1e-6)
                policy.reset()
                np.testing.assert_array_equal(policy.filtered,np.zeros(11))

    def test_grid_matches_published_endpoint(self):
        grid = Grid(DEFAULT_ARTIFACTS,[0,0],0)
        # README 7.1 provides the baseline cube endpoint command in degrees.
        np.testing.assert_allclose(np.rad2deg(grid.path(8)),
                                   [-28.29,0,1.82,87.31,-6.27,26.80,-7.40,-170.1],atol=0.02)
        np.testing.assert_array_equal(grid.path(0),np.zeros(8))
        np.testing.assert_allclose(grid.shift(99),grid.knots[-1])
        with self.assertRaises(ValueError):
            Grid(DEFAULT_ARTIFACTS,[1,0],0)

    def test_adc_units_and_quantization(self):
        actual = adc(np.array([0,5,100,4000])*9.80665/1000)
        expected = np.round(np.array([0,10/70,10/21.5,10/12.5])*1023)/1023
        np.testing.assert_allclose(actual,expected)

    def test_thumb_observation_differs_from_closing_force(self):
        forces = np.array([[-2,0,0], [0,3,4], [0,3,4], [-1,0,0], [2,0,0], [3,4,0]], dtype=float)
        rotations = np.tile(np.eye(3), (6,1,1))
        # Tangential/back contact must not be interpreted as a normal grip force.
        np.testing.assert_allclose(pad_measurements(forces, rotations), [2,0,0,1,0,0])
        # Original observation/done overrides the two thumbs only.
        np.testing.assert_allclose(pad_measurements(forces, rotations, True), [2,0,5,1,0,5])

    def test_real_policies_connected_to_simulation(self):
        controller = GraspController(layout="edge10")
        for _ in range(30):
            controller.step()
        self.assertEqual(controller.raise_calls,30)
        self.assertEqual(controller.last_obs.shape,(54,))
        np.testing.assert_allclose(controller.last_obs[37:43],controller.obs.last_input_adc,atol=1e-7)
        # Exercise a grasp phase at the actual grid-derived pose without claiming
        # this artificial reset reproduces the approach or constitutes grasp success.
        controller.target[controller.arm] = controller.grid.path(8)+controller.grid.insertion()
        controller.d.qpos[controller.qadr] = controller.target
        import mujoco
        mujoco.mj_forward(controller.m,controller.d)
        controller.start_grasp()
        controller.step()
        self.assertEqual(controller.grasp_calls,1)
        self.assertEqual(controller.last_obs.shape,(43,))
        np.testing.assert_allclose(controller.last_obs[37:43],controller.obs.last_input_adc,atol=1e-7)
        self.assertTrue(np.isfinite(controller.target).all())

    def test_invalid_cube_input_rejected(self):
        for xy in ([float("nan"),0],[1,0],[0,0]):
            with self.subTest(xy=xy),self.assertRaises(ValueError):
                GraspController(cube_xy=xy)


if __name__ == "__main__":
    unittest.main()
