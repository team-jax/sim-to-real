"""Experimental NumPy port of the deployed two-policy chain; MuJoCo only."""
from collections import deque
import mujoco
import numpy as np

from sim_validation.play_scene import SceneController
from sim_validation.inspect_model import ARM
from sim_validation.scene import LAYOUTS, CUBE_HALF_M
from .policies import Policy, DEFAULT_ARTIFACTS
from .observations import Observations
from .trajectory import Grid, smooth

DT = 1/60
FINGERS = ["hand_base_to_finger_1_1", "hand_base_to_finger_2_1", "rotation_base_to_finger_3_1",
           "finger_1_1_to_finger_1_2", "finger_1_2_to_finger_2_2", "finger_3_1_to_finger_3_2"]


class GraspController:
    def __init__(self, layout="edge10", cube_xy=None, artifacts=DEFAULT_ARTIFACTS, mode="guarded"):
        height, edge = LAYOUTS[layout]
        if cube_xy is None:
            cube_xy = [edge-0.06, 0.26]
        cube_xy = np.asarray(cube_xy, dtype=float)
        if cube_xy.shape != (2,) or not np.isfinite(cube_xy).all():
            raise ValueError("Cube xy must contain two finite meter values")
        if mode not in ("guarded", "reference"):
            raise ValueError("Unknown execution mode")
        self.mode, self.dz = mode, height-0.74
        self.grid = Grid(artifacts, cube_xy-[-0.285,0.26], self.dz)
        # Full cube footprint must be on this table.
        if not edge-0.8+CUBE_HALF_M <= cube_xy[0] <= edge-CUBE_HALF_M or abs(cube_xy[1])+CUBE_HALF_M > 0.35:
            raise ValueError("Cube footprint is outside the table")
        self.scene = SceneController(dx=cube_xy[0]-(edge-0.06), dy=cube_xy[1]-0.26, layout=layout)
        self.m, self.d = self.scene.model, self.scene.data
        self.obs = Observations(self.m, self.d, self.dz)
        self.raise_policy = Policy(artifacts/"raise_v25_1197.onnx",54)
        self.grasp_policy = Policy(artifacts/"grasp_lift_13075.onnx",43)
        names = [mujoco.mj_id2name(self.m,mujoco.mjtObj.mjOBJ_JOINT,j) for j in self.scene.joints]
        self.arm = np.array([names.index(n) for n in ARM])
        self.fingers = np.array([names.index(n) for n in FINGERS])
        self.qadr = self.scene.addresses
        self.vadr = self.m.jnt_dofadr[self.scene.joints]
        self.limits = self.m.jnt_range[self.scene.joints]
        self.arm_qadr = self.qadr[self.arm]
        self.hand = self.m.body("MG996R").id
        self.ik_use = self.arm[[0,2,3,4,5,6]]
        self.ik_dofs = self.vadr[self.ik_use]
        self.phase, self.phase_k, self.k = "RAISE", 0, 0
        self.reason = ""
        self.target = np.zeros(self.m.nu)
        self.smoothed = np.zeros(8)
        self.history = deque(maxlen=16)
        self.calm = self.rise_count = self.done_count = 0
        self.max_speed = 0.0
        self.initial_cube_z = float(self.d.body("cube").xpos[2])
        self.max_cube_rise = 0.0
        self.events = []
        self.last_obs = self.last_action = None
        self.finger_q = np.deg2rad([0,0,-30,0,0,0])
        self.raise_calls = self.grasp_calls = 0

    def transition(self, phase, reason=""):
        self.events.append({"t":self.k*DT,"from":self.phase,"to":phase,"reason":reason})
        self.phase, self.phase_k, self.reason = phase, 0, reason

    def arm_q(self):
        return self.d.qpos[self.arm_qadr]

    def start_grasp(self):
        self.grasp_policy.reset()
        self.blend_start = self.target.copy()
        self.hold = self.target[self.arm].copy()
        self.offset = np.zeros(8)
        self.p_des = self.obs.midpoint().copy()
        self.orientation = self.d.xquat[self.hand].copy()
        self.lift_z = float(self.d.body("cube").xpos[2])
        self.transition("GRASP_LIFT")

    def close(self, action):
        pad = self.obs.pad_forces()
        forces = pad[:3]+pad[3:]
        on = forces >= 0.3
        wanted = np.array([0.5*(5+4*action[7])]*2+[5+4*action[9]]) if on.all() else np.full(3,0.3)
        delta = np.where(on,np.clip(0.003*(wanted-forces),-0.25*DT,0.25*DT),0.25*DT)
        self.finger_q[:3] = np.clip(self.finger_q[:3]+delta,np.deg2rad([0,0,-30]),np.deg2rad([30,30,0]))
        self.finger_q[3:] = np.deg2rad(60)*np.clip(action[[8,8,10]],0,1)
        self.target[self.fingers] = self.finger_q

    def ik(self, action):
        midpoint = self.obs.midpoint()
        velocity = np.array([0.005*action[3],0.005*action[5],-0.035*action[0]])
        self.p_des += velocity*DT
        velocity += 2*np.clip(self.p_des-midpoint,-0.02,0.02)
        inverse = self.d.xquat[self.hand].copy()
        inverse[1:] *= -1
        error_quat, angular = np.zeros(4), np.zeros(3)
        mujoco.mju_mulQuat(error_quat,self.orientation,inverse)
        if error_quat[0] < 0:
            error_quat *= -1
        mujoco.mju_quat2Vel(angular,error_quat,1)
        angular *= 2
        jv, jw = np.zeros((3,self.m.nv)),np.zeros((3,self.m.nv))
        # Jacobian of a point fixed to the hand at the current fingertip midpoint.
        mujoco.mj_jac(self.m,self.d,jv,jw,midpoint,self.hand)
        jac = np.vstack([jv[:,self.ik_dofs],jw[:,self.ik_dofs]])
        dq = jac.T@np.linalg.solve(jac@jac.T+0.02**2*np.eye(6),np.r_[velocity,angular])
        self.offset[[0,2,3,4,5,6]] += np.clip(dq,-0.6,0.6)*DT
        self.offset = np.clip(self.offset,-1,1)
        self.target[self.arm] = self.hold+self.offset

    def step(self):
        if self.phase in ("FAULT","DONE"):
            return
        try:
            old_phase = self.phase
            self._command()
            if not np.isfinite(self.target).all():
                raise ValueError("Nonfinite motor target")
            self.target = np.clip(self.target,self.limits[:,0],self.limits[:,1])
            # Do not apply manual runner's extra slew filter to the original chain.
            self.d.ctrl[:] = self.target
            for _ in range(2):
                mujoco.mj_step(self.m,self.d)
                if not np.isfinite(self.d.qpos).all() or not np.isfinite(self.d.qvel).all():
                    raise ValueError("Nonfinite physics state")
                if np.any(self.d.warning.number):
                    raise ValueError("MuJoCo numerical warning")
            mujoco.mj_forward(self.m,self.d)
            self.max_cube_rise = max(self.max_cube_rise,float(self.d.body("cube").xpos[2])-self.initial_cube_z)
            self.max_speed = max(self.max_speed,float(np.max(np.abs(self.d.qvel[self.vadr]))))
            if self.max_speed > 3:
                i = int(np.argmax(np.abs(self.d.qvel[self.vadr])))
                name = mujoco.mj_id2name(self.m,mujoco.mjtObj.mjOBJ_JOINT,self.scene.joints[i])
                raise ValueError(f"{name}: robot speed >3 rad/s experimental abort threshold")
            if self.d.body("cube").xpos[2] < self.initial_cube_z-0.05:
                raise ValueError("Cube fell more than 5 cm below its initial center")
            self.k += 1
            if old_phase == self.phase:
                self.phase_k += 1
            if self.k*DT >= 30 and self.phase != "DONE":
                self.transition("FAULT","30 s chain timeout")
        except (ValueError,np.linalg.LinAlgError) as error:
            self.transition("FAULT",str(error))

    def _command(self):
        t = self.k*DT
        if self.phase in ("RAISE","SETTLE"):
            extra = np.r_[min(t/8,1.25),np.clip(self.grid.path(t,True)-self.arm_q(),-3,3),-0.34,0.435]
            self.last_obs = self.obs.build(extra=extra)
            action = self.raise_policy(self.last_obs)
            self.raise_calls += 1
            self.last_action = action
            raw = self.grid.path(t)+np.deg2rad(8)*min(t/0.5,1)*action[:8]
            new = 0.85*self.smoothed+0.15*raw
            new[:7] = self.smoothed[:7]+np.clip(new[:7]-self.smoothed[:7],-0.5*DT,0.5*DT)
            self.smoothed = new
            self.target[self.arm] = new
            self.target[self.fingers] = np.deg2rad([0,0,-30*smooth((t-1)/3),0,0,0])
            self.history.append(self.arm_q().copy())
            if self.phase == "RAISE" and t >= 8:
                self.transition("SETTLE")
            still = len(self.history)==16 and np.max(np.abs(self.history[-1]-self.history[0])) < np.deg2rad(0.2)
            near = np.max(np.abs(self.arm_q()-self.grid.path(8,True))) < np.deg2rad(3)
            self.calm = self.calm+1 if t>=8.5 and still and near else 0
            if self.calm>=10 or (t>=11 and self.mode=="reference"):
                self.insert_start = self.target[self.arm].copy()
                self.insert_delta = self.grid.insertion()
                self.transition("INSERT", "settled" if self.calm>=10 else "reference forced timeout")
            elif t>=11:
                self.transition("FAULT","Approach did not settle within 11 s")
        elif self.phase == "INSERT":
            self.target[self.arm] = self.insert_start+smooth((self.phase_k+1)/30)*self.insert_delta
            if self.phase_k>=44:
                self.start_grasp()
        elif self.phase in ("GRASP_LIFT","RETURN"):
            self.last_obs = self.obs.build(grasp=True)
            action = self.grasp_policy(self.last_obs)
            self.grasp_calls += 1
            if self.phase == "RETURN":
                action[7:11] = self.frozen
            self.last_action = action
            self.close(action)
            self.target[self.arm[7]] = np.deg2rad(-170)
            rise = self.d.body("cube").xpos[2]-self.lift_z
            if self.phase == "GRASP_LIFT":
                self.ik(action)
                self.target[self.arm[7]] = np.deg2rad(-170)
                self.rise_count = self.rise_count+1 if rise>=0.04 else 0
                if self.rise_count>=30:
                    self.return_start = self.target[self.arm].copy()
                    self.frozen = action[7:11].copy()
                    self.previous_arm = self.arm_q().copy()
                    self.transition("RETURN")
                else:
                    self.target = self.blend_start+smooth((self.phase_k+1)*DT/0.5)*(self.target-self.blend_start)
            else:
                q_l = np.deg2rad([1.9,0,7.2,92.1,0,-1.9,-0.2])
                self.target[self.arm[:7]] = self.return_start[:7]+smooth(self.phase_k*DT/2.5)*(q_l-self.return_start[:7])
                pad = self.obs.pad_forces(thumb_magnitude=True)
                force = pad[:3]+pad[3:]
                still = np.max(np.abs(self.arm_q()[:7]-self.previous_arm[:7])) < np.deg2rad(0.05)
                self.previous_arm = self.arm_q().copy()
                ok = self.phase_k*DT>=2.5 and still and rise>=0.04 and force[2]>=0.3 and np.any(force[:2]>=0.3)
                self.done_count = self.done_count+1 if ok else 0
                if self.done_count>=30:
                    self.transition("DONE")

    def summary(self):
        return {"phase":self.phase,"reason":self.reason,"mode":self.mode,"time_s":float(self.d.time),
                "raise_inferences":self.raise_calls,"grasp_inferences":self.grasp_calls,
                "max_robot_speed_rad_s":self.max_speed,"max_cube_rise_m":self.max_cube_rise,
                "cube_position_m":self.d.body("cube").xpos.tolist(),"events":self.events}
