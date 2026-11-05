"""Fake hardware loop: read MuJoCo joints, limit targets, apply bounded PD.

This exercises an adapter, not the learned grasp chain. No CAN/serial imports.
"""
import argparse
import csv
from pathlib import Path
import time

import mujoco
import numpy as np

from .model import contact_pairs, load_robot


class SimHardware:
    def __init__(self, ignore_contacts=False):
        self.model = load_robot(visuals=True)
        self.model.opt.timestep = 1 / 600
        self.data = mujoco.MjData(self.model)
        self.target = self.data.qpos.copy()
        # Add damping to the imported, otherwise unactuated joints.
        self.model.dof_damping[:] = 2.0
        self.model.dof_armature[:] = 0.1
        if ignore_contacts:
            self.model.geom_contype[:] = 0
            self.model.geom_conaffinity[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def read(self):
        return self.data.qpos.copy(), self.data.qvel.copy()

    def write(self, requested):
        requested = np.asarray(requested)
        if requested.shape != self.target.shape or not np.isfinite(requested).all():
            raise ValueError("Invalid joint command: shape or nonfinite value")
        # Target slew limit 0.1 rad/s at 60 Hz, independent of simulation dt.
        limited = np.clip(requested, self.model.jnt_range[:, 0], self.model.jnt_range[:, 1])
        self.target += np.clip(limited - self.target, -0.1 / 60, 0.1 / 60)

    def step(self):
        for _ in range(10):
            mujoco.mj_forward(self.model, self.data)
            # Bounded, gravity-compensated PD for adapter smoke testing only.
            torque = 40 * (self.target - self.data.qpos) - 8 * self.data.qvel + self.data.qfrc_bias
            self.data.qfrc_applied[:] = np.clip(torque, -10, 10)
            mujoco.mj_step(self.model, self.data)
            q, velocity = self.read()
            if not np.isfinite(q).all() or not np.isfinite(velocity).all():
                raise RuntimeError("Nonfinite simulation state")
            if np.max(np.abs(velocity)) > 1.0:
                raise RuntimeError("Simulated speed exceeded 1 rad/s")
            if np.any(q < self.model.jnt_range[:, 0] - 0.02) or np.any(q > self.model.jnt_range[:, 1] + 0.02):
                raise RuntimeError("Simulated joint limit exceeded")


def run(args):
    hw = SimHardware(args.ignore_contacts)
    initial = [c for c in contact_pairs(hw.model, hw.data) if c[2] < -0.001]
    if initial and not args.ignore_contacts:
        raise RuntimeError(f"Initial mesh penetration; stopped before movement: {initial}. "
                           "Use --ignore-contacts only for adapter testing.")
    joint = mujoco.mj_name2id(hw.model, mujoco.mjtObj.mjOBJ_JOINT, args.joint)
    if joint < 0:
        raise ValueError(f"Unknown joint: {args.joint}")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    viewer = None
    if args.viewer:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(hw.model, hw.data)
        viewer.opt.geomgroup[0] = 0
        viewer.opt.geomgroup[1] = 1
        viewer.cam.lookat[:] = np.mean(hw.data.xpos[1:], axis=0)
        viewer.cam.distance = 1.8
    print("FAKE HARDWARE: no motor connection, no grasp policy, no cube/pressure simulation.")
    print(f"Contacts disabled: {args.ignore_contacts}; log: {args.log}")
    try:
        with args.log.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["time_s", "joint", "target_rad", "position_rad", "velocity_rad_s"])
            for k in range(round(args.seconds * 60)):
                if viewer is not None and not viewer.is_running():
                    break
                started = time.monotonic()
                target = np.zeros(hw.model.nq)
                # Gentle 0 -> +2 degrees -> 0 trajectory for one named joint.
                target[joint] = np.deg2rad(2) * (1 - np.cos(2 * np.pi * k / (args.seconds * 60))) / 2
                hw.write(target)
                hw.step()
                q, v = hw.read()
                writer.writerow([hw.data.time, args.joint, hw.target[joint], q[joint], v[joint]])
                if viewer is not None:
                    viewer.sync()
                    time.sleep(max(0, 1 / 60 - (time.monotonic() - started)))
        print("Fake hardware run completed.")
    finally:
        if viewer is not None:
            viewer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--ignore-contacts", action="store_true")
    parser.add_argument("--seconds", type=float, default=5)
    parser.add_argument("--joint", default="shoulder_yaw_to_upper_arm")
    parser.add_argument("--log", type=Path, default=Path("runs/dry_run.csv"))
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be finite and positive")
    run(args)


if __name__ == "__main__":
    main()
