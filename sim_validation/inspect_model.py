"""View the unmodified robot at named poses, without running dynamics."""
import argparse
import time

import mujoco
import numpy as np

from .model import contact_pairs, load_robot

ARM = ["shoulder_pitch_to_shoulder_roll", "shoulder_roll_to_shoulder_yaw",
       "shoulder_yaw_to_upper_arm", "upper_arm_to_under_arm", "under_arm_to_wrist_pitch",
       "wrist_pitch_to_wrist_yaw", "wrist_yaw_to_hand_base", "hand_base_to_rotation_base"]
POSES = {"zero": [0]*8,
         "pregrasp": [-25.50, 0, 2.02, 84.01, -6.10, 22.59, 4.91, -170.1],
         "return": [1.9, 0, 7.2, 92.1, 0, -1.9, -0.2, -170]}


def set_pose(model, data, pose):
    data.qpos[:] = 0
    for name, degrees in zip(ARM, POSES[pose]):
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[j]] = np.deg2rad(degrees)
    if pose == "pregrasp":
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "rotation_base_to_finger_3_1")
        data.qpos[model.jnt_qposadr[j]] = np.deg2rad(-30)
    mujoco.mj_forward(model, data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose", choices=POSES, default="zero")
    parser.add_argument("--collision", action="store_true", help="Show collision meshes only")
    args = parser.parse_args()
    model = load_robot(visuals=True)
    data = mujoco.MjData(model)
    set_pose(model, data, args.pose)
    contacts = {}
    for a, b, distance in contact_pairs(model, data):
        key = tuple(sorted((a, b)))
        contacts[key] = min(contacts.get(key, 0), distance)
    print("Static inspection: no dynamics, no hardware, URDF base frame.")
    for pair, distance in sorted(contacts.items()):
        print(f"{' / '.join(pair)}: {distance * 1000:.3f} mm")
    from mujoco import viewer
    with viewer.launch_passive(model, data) as window:
        window.opt.geomgroup[0] = int(args.collision)
        window.opt.geomgroup[1] = int(not args.collision)
        window.cam.lookat[:] = np.mean(data.xpos[1:], axis=0)
        window.cam.distance = 1.8
        window.cam.azimuth = 135
        window.cam.elevation = -15
        while window.is_running():
            window.sync()
            time.sleep(1 / 30)


if __name__ == "__main__":
    main()
