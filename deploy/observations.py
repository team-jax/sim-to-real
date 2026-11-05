import mujoco
import numpy as np

ISAAC_NAMES = [
    "shoulder_pitch_to_shoulder_roll", "shoulder_roll_to_shoulder_yaw",
    "shoulder_yaw_to_upper_arm", "upper_arm_to_under_arm", "under_arm_to_wrist_pitch",
    "wrist_pitch_to_wrist_yaw", "wrist_yaw_to_hand_base", "hand_base_to_finger_1_1",
    "hand_base_to_finger_2_1", "hand_base_to_rotation_base", "finger_1_1_to_finger_1_2",
    "finger_1_2_to_finger_2_2", "rotation_base_to_finger_3_1", "finger_3_1_to_finger_3_2",
]
PADS = ["Finger_1_1", "Finger_2_1", "Finger_3_1", "Finger_1_2", "Finger_2_2", "Finger_3_2"]


def pad_measurements(world_forces, link_rotations, thumb_magnitude=False):
    """Original v11 override: observation/done thumbs use vector magnitude.

    Balanced close explicitly calls the base class and uses normals on all pads.
    """
    world_forces = np.asarray(world_forces)
    normal = np.maximum(0, -np.einsum("ni,ni->n", link_rotations[:, :, 0], world_forces))
    if thumb_magnitude:
        normal[[2, 5]] = np.linalg.norm(world_forces[[2, 5]], axis=1)
    return normal


def adc(forces):
    gf = np.asarray(forces)*1000/9.80665
    loads = [5,55,100,165,260,350,470,590,750,975,1290,1585,1920,2465,3240,4000]
    resistances = [60,19.5,11.5,8.8,7.5,6.8,5.4,4.4,4,3.5,3.1,2.9,2.7,2.6,2.55,2.5]
    resistance = np.exp(np.interp(np.log(np.clip(gf,5,4000)), np.log(loads), np.log(resistances)))
    return np.where(gf < 5, 0, np.round(1023*10/(resistance+10))/1023)


class Observations:
    def __init__(self, model, data, dz):
        self.m, self.d, self.dz = model, data, dz
        self.joints = np.array([model.joint(n).id for n in ISAAC_NAMES])
        self.qadr = model.jnt_qposadr[self.joints]
        self.vadr = model.jnt_dofadr[self.joints]
        self.limits = model.jnt_range[self.joints]
        self.pad_ids = [model.body(n).id for n in PADS]
        self.cube_id = model.body("cube").id
        self.base_id = model.body("robot_mount").id

    def midpoint(self):
        return self.d.xpos[self.pad_ids[3:]].mean(axis=0)

    def pad_forces(self, thumb_magnitude=False):
        world_forces = np.zeros((6,3))
        for i, contact in enumerate(self.d.contact):
            b1, b2 = (int(self.m.geom_bodyid[g]) for g in contact.geom)
            if b1 != self.cube_id and b2 != self.cube_id:
                continue
            link = b2 if b1 == self.cube_id else b1
            if link not in self.pad_ids:
                continue
            wrench = np.zeros(6)
            mujoco.mj_contactForce(self.m, self.d, i, wrench)
            # MuJoCo contact frame force acts on geom2; geom1 gets its negative.
            force = contact.frame.reshape(3,3).T @ wrench[:3]
            if link == b1:
                force = -force
            world_forces[self.pad_ids.index(link)] += force
        rotations = self.d.xmat[self.pad_ids].reshape(6, 3, 3)
        return pad_measurements(world_forces, rotations, thumb_magnitude)

    def build(self, grasp=False, extra=None):
        q, velocity = self.d.qpos[self.qadr], self.d.qvel[self.vadr]
        rotation = self.d.xmat[self.base_id].reshape(3,3)
        cube = self.d.xpos[self.cube_id]
        spatial = np.zeros(6)
        mujoco.mj_objectVelocity(self.m,self.d,mujoco.mjtObj.mjOBJ_BODY,self.cube_id,spatial,0)
        self.last_input_force_n = self.pad_forces(thumb_magnitude=True)
        self.last_input_adc = adc(self.last_input_force_n)
        obs = np.r_[2*(q-self.limits[:,0])/(self.limits[:,1]-self.limits[:,0])-1,
                    velocity*0.2, rotation.T@(cube-self.d.xpos[self.base_id]),
                    rotation.T@spatial[3:], rotation.T@(cube-self.midpoint()),
                    self.last_input_adc]
        obs = np.clip(obs,-5,5)
        if grasp:
            obs[9], obs[23] = -0.7222222222, 0
            obs[29] -= 0.115
            obs[30] -= self.dz
        if extra is not None:
            obs = np.r_[obs, extra]
        return obs.astype(np.float32)
