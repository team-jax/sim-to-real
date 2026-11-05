"""Interactive MuJoCo scene. Keyboard commands affect simulation only."""
import argparse
import threading
import time
from collections import deque

import mujoco
import numpy as np

from .model import load_robot
from .inspect_model import ARM, POSES
from .scene import LAYOUTS


class SceneController:
    def __init__(self, dx=0, dy=0, pose="zero", layout="reference"):
        self.model = load_robot(visuals=True, environment={"dx": dx, "dy": dy, "layout": layout})
        self.data = mujoco.MjData(self.model)
        self.joints = [int(self.model.actuator_trnid[i, 0]) for i in range(self.model.nu)]
        self.addresses = np.array([self.model.jnt_qposadr[j] for j in self.joints])
        self.targets = np.zeros(self.model.nu)
        self.reset(pose)

    def pose_target(self, pose):
        target = np.zeros(self.model.nu)
        named = dict(zip(ARM, np.deg2rad(POSES[pose])))
        if pose == "pregrasp":
            named["rotation_base_to_finger_3_1"] = np.deg2rad(-30)
        for i, j in enumerate(self.joints):
            target[i] = named.get(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j), 0)
        return target

    def reset(self, pose="zero"):
        mujoco.mj_resetData(self.model, self.data)
        self.targets = self.pose_target(pose)
        self.data.qpos[self.addresses] = self.targets
        self.data.ctrl[:] = self.targets
        mujoco.mj_forward(self.model, self.data)

    def command_joint(self, name, degrees):
        if not np.isfinite(degrees):
            raise ValueError("Angle must be finite")
        for i, j in enumerate(self.joints):
            if mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) == name:
                angle = np.deg2rad(degrees)
                lo, hi = self.model.jnt_range[j]
                if not lo <= angle <= hi:
                    raise ValueError(f"{name}: outside [{np.rad2deg(lo):.1f}, {np.rad2deg(hi):.1f}] degrees")
                self.targets[i] = angle
                return
        raise ValueError(f"Unknown actuated joint: {name}")

    def step(self):
        self.data.ctrl[:] += np.clip(self.targets-self.data.ctrl, -0.3/60, 0.3/60)
        for _ in range(2):
            mujoco.mj_step(self.model, self.data)
            if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
                raise RuntimeError("Nonfinite simulation state")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dx-cm", type=float, default=0)
    parser.add_argument("--dy-cm", type=float, default=0)
    parser.add_argument("--pose", choices=POSES, default="zero")
    parser.add_argument("--layout", choices=LAYOUTS, default="center10",
                        help="center10: base center gap; edge10: modeled base edge gap; reference: original README")
    parser.add_argument("--joint", help="Named joint receiving --degrees")
    parser.add_argument("--degrees", type=float)
    parser.add_argument("--headless-steps", type=int, default=0)
    args = parser.parse_args()
    if bool(args.joint) != (args.degrees is not None):
        parser.error("Use --joint and --degrees together")
    if args.headless_steps < 0:
        parser.error("--headless-steps must be nonnegative")
    scene = SceneController(args.dx_cm/100, args.dy_cm/100, args.pose, args.layout)
    height, edge = LAYOUTS[args.layout]
    print(f"Layout={args.layout}, table height={height*100:.1f} cm, edge x={edge*100:.1f} cm", flush=True)
    if args.joint:
        scene.command_joint(args.joint, args.degrees)
    print("Simulation only. Self-collision OFF (Isaac reference); table/cube contacts ON.", flush=True)
    print("SPACE: run/pause | 0/1/2: zero/pregrasp/return target | R: reset | C: contacts view", flush=True)
    print("Starts paused. Pose commands are direct targets, not trained approach paths.", flush=True)
    print("Green dot = placement (0,0), red object = cube. Policy/pressure sensors not connected.", flush=True)
    if args.headless_steps:
        for _ in range(args.headless_steps):
            scene.step()
        print(f"Completed {args.headless_steps} control steps; cube={scene.data.body('cube').xpos}")
        return
    from mujoco import viewer
    queue, lock = deque(), threading.Lock()
    def key_callback(key):
        with lock:
            queue.append(key)
    running, collision_view = False, False
    with viewer.launch_passive(scene.model, scene.data, key_callback=key_callback) as window:
        window.cam.lookat[:] = [-0.2, 0.1, 0.75]
        window.cam.distance = 2.1
        window.cam.azimuth, window.cam.elevation = 135, -25
        window.opt.geomgroup[0] = 0
        window.opt.geomgroup[1:4] = 1
        window.opt.label = mujoco.mjtLabel.mjLABEL_SITE
        while window.is_running():
            started = time.monotonic()
            with lock:
                keys = list(queue)
                queue.clear()
            for key in keys:
                if key == 32:
                    running = not running
                    print("RUN" if running else "PAUSE", flush=True)
                elif key in (48, 49, 50):
                    pose = ["zero", "pregrasp", "return"][key-48]
                    scene.targets = scene.pose_target(pose)
                    print(f"Target: {pose}; SPACE to run if paused", flush=True)
                elif key in (82, 114):
                    scene.reset(args.pose)
                    running = False
                elif key in (67, 99):
                    collision_view = not collision_view
                    window.opt.geomgroup[0] = int(collision_view)
                    window.opt.geomgroup[1] = int(not collision_view)
            if running:
                try:
                    scene.step()
                except RuntimeError as error:
                    running = False
                    print(f"FAULT: {error}", flush=True)
            # MuJoCo also uses 0/1/2 to toggle geom groups. Our pose shortcuts
            # must not hide the robot or table through that built-in handler.
            with window.lock():
                window.opt.geomgroup[0] = int(collision_view)
                window.opt.geomgroup[1] = int(not collision_view)
                window.opt.geomgroup[2:4] = 1
            window.sync()
            time.sleep(max(0, 1/60-(time.monotonic()-started)))


if __name__ == "__main__":
    main()
