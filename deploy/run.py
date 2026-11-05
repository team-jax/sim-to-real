"""Experimental real-policy sim2sim runner. No real robot backend exists."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import threading
import time
import subprocess

import mujoco
import numpy as np
import onnxruntime

from .controller import GraspController, DT
from .policies import DEFAULT_ARTIFACTS
from sim_validation.scene import (LAYOUTS, CUBE_SIDE_M, CUBE_MASS_KG, CUBE_FRICTION,
                                  CUBE_DYNAMIC_FRICTION, TABLE_STATIC_FRICTION,
                                  TABLE_DYNAMIC_FRICTION, CONTACT_FRICTION_BASIS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", choices=LAYOUTS, default="edge10")
    parser.add_argument("--cube-x", type=float, help="Cube center WORLD x in meters")
    parser.add_argument("--cube-y", type=float, help="Cube center WORLD y in meters")
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--mode", choices=["guarded","reference"], default="guarded")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (args.cube_x is None) != (args.cube_y is None):
        parser.error("Supply both --cube-x and --cube-y (meters)")
    if not np.isfinite(args.seconds) or args.seconds<=0 or args.seconds>30:
        parser.error("--seconds must be in (0,30]")
    xy = None if args.cube_x is None else [args.cube_x,args.cube_y]
    controller = GraspController(args.layout,xy,args.artifacts,args.mode)
    out = args.output or Path("runs")/datetime.now().strftime("policy_%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True,exist_ok=True)
    manifest = {"layout":args.layout,"mode":args.mode,"mujoco":mujoco.__version__,
                "cube_side_m":CUBE_SIDE_M,"cube_mass_kg":CUBE_MASS_KG,
                "cube_static_friction_reference":CUBE_FRICTION,
                "cube_sliding_friction":CUBE_DYNAMIC_FRICTION,
                "table_static_friction_reference":TABLE_STATIC_FRICTION,
                "table_sliding_friction":TABLE_DYNAMIC_FRICTION,
                "contact_friction_basis":CONTACT_FRICTION_BASIS,
                "cube_hand_contact_mu":CUBE_DYNAMIC_FRICTION,
                "cube_table_contact_mu":(CUBE_DYNAMIC_FRICTION+TABLE_DYNAMIC_FRICTION)/2,
                "onnxruntime":onnxruntime.__version__,"requested_cube_xy_m":xy,
                "policy_status":"experimental port; grasp success not established",
                "control_hz":60,"physics_hz":120,"self_collision":False,
                "artifacts":{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in args.artifacts.iterdir() if p.suffix in (".pt",".onnx")}}
    revision = subprocess.run(["git","-C",str(args.artifacts.parent.parent),"rev-parse","HEAD"],
                              capture_output=True,text=True)
    manifest["jaxisaaclab_commit"] = revision.stdout.strip() if revision.returncode==0 else None
    # Include actual scene/model provenance as well as policies.
    from sim_validation.model import DEFAULT_URDF
    model_files = [DEFAULT_URDF,*sorted((DEFAULT_URDF.parent.parent/"meshes").rglob("*"))]
    manifest["model_sha256"] = {str(p.relative_to(DEFAULT_URDF.parent.parent)):hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in model_files if p.is_file()}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print(f"MuJoCo ONLY | layout={args.layout} | cube={controller.d.body('cube').xpos} | logs={out}",flush=True)
    print("Real ONNX policies + lookup paths + force control + IK. Experimental, not validated grasp.",flush=True)
    window = None
    toggle = threading.Event()
    if not args.headless:
        from mujoco import viewer
        window = viewer.launch_passive(controller.m,controller.d,key_callback=lambda k: toggle.set() if k==32 else None)
        window.cam.lookat[:] = [-0.25,0.12,0.8]
        window.cam.distance,window.cam.azimuth,window.cam.elevation = 1.9,135,-25
        window.opt.geomgroup[0] = 0
        window.opt.geomgroup[1:4] = 1
        window.opt.label = mujoco.mjtLabel.mjLABEL_SITE
        print("Click viewer, SPACE to start/pause. Close window to stop. No manual pose shortcuts.",flush=True)
    running = args.headless
    records = []
    last_phase = None
    pressure_active = False
    try:
        while controller.k < round(args.seconds*60) and controller.phase not in ("DONE","FAULT"):
            began = time.monotonic()
            if window is not None and not window.is_running():
                break
            if toggle.is_set():
                running = not running
                toggle.clear()
            if running:
                controller.step()
                row = {"time_s":float(controller.d.time),"phase":controller.phase,
                       "robot_q":controller.d.qpos[controller.qadr].copy(),
                       "target":controller.target.copy(),"cube":controller.d.body("cube").xpos.copy(),
                       "robot_velocity":controller.d.qvel[controller.vadr].copy(),
                       "pad_n":controller.obs.pad_forces(),
                       "pad_policy_n":controller.obs.pad_forces(thumb_magnitude=True),
                       "input_pad_n":controller.obs.last_input_force_n.copy(),
                       "input_adc":controller.obs.last_input_adc.copy(),
                       "observation":np.pad(controller.last_obs,(0,54-len(controller.last_obs))) if controller.last_obs is not None else np.zeros(54),
                       "action":controller.last_action.copy() if controller.last_action is not None else np.zeros(11),
                       "raw_action":(controller.raise_policy.last_raw.copy() if controller.phase in ("RAISE","SETTLE","INSERT")
                                     else controller.grasp_policy.last_raw.copy()),
                       "compute_s":time.monotonic()-began}
                records.append(row)
                if controller.phase!=last_phase:
                    print(f"t={controller.k*DT:.2f}s {controller.phase} {controller.reason}",flush=True)
                    last_phase=controller.phase
                loaded = bool(np.any(controller.obs.last_input_adc > 0))
                if controller.k % 60 == 0 or loaded != pressure_active:
                    print("PAD N="+np.array2string(controller.obs.last_input_force_n,precision=2)
                          +" ADC="+np.array2string(controller.obs.last_input_adc,precision=3),flush=True)
                pressure_active = loaded
            if window is not None:
                window.opt.geomgroup[0] = 0
                window.opt.geomgroup[1:4] = 1
                window.sync()
                time.sleep(max(0,DT-(time.monotonic()-began)))
    finally:
        if window is not None:
            window.close()
        summary=controller.summary()
        summary["run_status"] = "success" if controller.phase=="DONE" else "fault" if controller.phase=="FAULT" else "incomplete"
        (out/"summary.json").write_text(json.dumps(summary,indent=2))
        if records:
            np.savez_compressed(out/"trace.npz",**{key:np.asarray([row[key] for row in records]) for key in records[0]})
            forces=np.asarray([row["input_pad_n"] for row in records])
            readings=np.asarray([row["input_adc"] for row in records])
            summary["pressure_max_n"] = forces.max(axis=0).tolist()
            summary["policy_adc_max"] = readings.max(axis=0).tolist()
            summary["pressure_source"] = "MuJoCo cube-pad contact force, not a physical sensor"
            (out/"summary.json").write_text(json.dumps(summary,indent=2))
        print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    if controller.phase=="FAULT":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
