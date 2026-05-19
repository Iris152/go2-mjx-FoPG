#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import types
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
POS_STOP_F = 2.146e9
VEL_STOP_F = 16000.0


def ensure_unitree_sdk2py_namespace() -> None:
    if "unitree_sdk2py" in sys.modules:
        return
    for entry in sys.path:
        pkg_dir = Path(entry) / "unitree_sdk2py"
        if pkg_dir.is_dir():
            pkg = types.ModuleType("unitree_sdk2py")
            pkg.__path__ = [str(pkg_dir)]
            pkg.__package__ = "unitree_sdk2py"
            pkg.__file__ = str(pkg_dir / "__init__.py")
            sys.modules["unitree_sdk2py"] = pkg
            return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch GO2 DDS simulator for sim2sim validation"
    )
    parser.add_argument(
        "--backend",
        choices=["menagerie", "unitree"],
        default="menagerie",
        help=(
            "menagerie uses the training-side mujoco_menagerie XML with Unitree DDS topics; "
            "unitree launches the original unitree_mujoco Python simulator."
        ),
    )
    parser.add_argument(
        "--unitree_mujoco_root",
        type=str,
        default=str(SCRIPT_DIR / "_deps" / "unitree_mujoco"),
        help="Path to the cloned unitree_mujoco repository",
    )
    parser.add_argument("--robot", type=str, default="go2")
    parser.add_argument("--scene", type=str, default="scene.xml")
    parser.add_argument(
        "--scene_xml",
        type=str,
        default=str(SCRIPT_DIR / "mujoco_menagerie" / "unitree_go2" / "scene_mjx.xml"),
        help="Menagerie XML used by --backend menagerie.",
    )
    parser.add_argument("--domain_id", type=int, default=1)
    parser.add_argument("--network", type=str, default="lo")
    parser.add_argument("--use_joystick", action="store_true")
    parser.add_argument("--print_scene_information", action="store_true")
    parser.add_argument(
        "--simulate_dt",
        type=float,
        default=None,
        help="Physics step. Defaults to 0.002 for menagerie backend and 0.005 for unitree backend.",
    )
    parser.add_argument("--viewer_dt", type=float, default=0.02)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max_time", type=float, default=0.0)
    parser.add_argument(
        "--no_reset_home",
        action="store_true",
        help="Do not reset the official unitree_mujoco scene to the home keyframe before starting threads",
    )
    parser.add_argument("--home_hold_kp", type=float, default=45.0)
    parser.add_argument("--home_hold_kd", type=float, default=3.5)
    parser.add_argument("--control_mode", choices=["auto", "position_servo", "pd_torque"], default="auto")
    parser.add_argument("--servo_kp", type=float, default=50.0)
    parser.add_argument("--servo_kd", type=float, default=0.5)
    parser.add_argument(
        "--initial_pose",
        choices=["home", "crouch", "prone"],
        default="prone",
        help="Menagerie backend initial pose before active LowCmd is received.",
    )
    parser.add_argument(
        "--idle_target",
        choices=["initial", "home"],
        default="initial",
        help="Menagerie backend hold target while no active LowCmd is available.",
    )
    return parser.parse_args()


def run_menagerie_sim(args: argparse.Namespace) -> int:
    local_sim = SCRIPT_DIR / "go2_mjx_lowlevel_dds_sim.py"
    if not local_sim.exists():
        raise SystemExit(f"Local menagerie simulator not found: {local_sim}")

    cmd = [
        sys.executable,
        str(local_sim),
        "--network",
        str(args.network),
        "--domain_id",
        str(args.domain_id),
        "--scene_xml",
        str(Path(args.scene_xml).expanduser().resolve()),
        "--sim_dt",
        str(args.simulate_dt if args.simulate_dt is not None else 0.002),
        "--viewer_dt",
        str(args.viewer_dt),
        "--control_mode",
        str(args.control_mode),
        "--servo_kp",
        str(args.servo_kp),
        "--servo_kd",
        str(args.servo_kd),
        "--initial_pose",
        str(args.initial_pose),
        "--idle_target",
        str(args.idle_target),
    ]
    if args.headless:
        cmd.append("--headless")
    if args.max_time > 0.0:
        cmd.extend(["--max_time", str(args.max_time)])
    if args.use_joystick:
        print("[menagerie] --use_joystick is ignored by the local simulator.", flush=True)
    print(
        "[menagerie] Launching training-XML MuJoCo DDS simulator:",
        local_sim,
        flush=True,
    )
    return subprocess.call(cmd)


def is_active_lowcmd(msg) -> bool:
    for i in range(12):
        motor = msg.motor_cmd[i]
        if int(motor.mode) != 0x01:
            continue
        if abs(float(motor.q) - POS_STOP_F) < 1e3 and abs(float(motor.dq) - VEL_STOP_F) < 1e-3:
            continue
        if (
            abs(float(motor.kp)) > 1e-6
            or abs(float(motor.kd)) > 1e-6
            or abs(float(motor.tau)) > 1e-6
            or abs(float(motor.q) - POS_STOP_F) >= 1e3
            or abs(float(motor.dq) - VEL_STOP_F) >= 1e-3
        ):
            return True
    return False


def main() -> int:
    args = parse_args()

    if args.backend == "menagerie":
        return run_menagerie_sim(args)

    repo_root = Path(args.unitree_mujoco_root).expanduser().resolve()
    sim_py_root = repo_root / "simulate_python"
    robot_scene = repo_root / "unitree_robots" / args.robot / args.scene

    if not sim_py_root.exists():
        print("[unitree] unitree_mujoco repo not available; falling back to menagerie backend.", flush=True)
        return run_menagerie_sim(args)
    if not robot_scene.exists():
        raise SystemExit(f"Robot scene not found: {robot_scene}")

    sys.path.insert(0, str(sim_py_root))

    import config

    config.ROBOT = args.robot
    config.ROBOT_SCENE = str(robot_scene)
    config.DOMAIN_ID = args.domain_id
    config.INTERFACE = args.network
    config.USE_JOYSTICK = 1 if args.use_joystick else 0
    config.PRINT_SCENE_INFORMATION = bool(args.print_scene_information)
    config.SIMULATE_DT = float(args.simulate_dt if args.simulate_dt is not None else 0.005)
    config.VIEWER_DT = float(args.viewer_dt)

    print(
        "[sim-config] root=%s robot=%s scene=%s domain_id=%s network=%s joystick=%s"
        % (
            repo_root,
            config.ROBOT,
            config.ROBOT_SCENE,
            config.DOMAIN_ID,
            config.INTERFACE,
            config.USE_JOYSTICK,
        ),
        flush=True,
    )

    ensure_unitree_sdk2py_namespace()
    import unitree_mujoco as sim
    import mujoco
    import numpy as np

    home_hold_target = None
    if not args.no_reset_home:
        key_id = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(sim.mj_model, sim.mj_data, key_id)
            mujoco.mj_forward(sim.mj_model, sim.mj_data)
            home_hold_target = np.array(sim.mj_model.key_ctrl[key_id, : sim.mj_model.nu], dtype=float)
            print("[sim-config] Reset scene to keyframe 'home'.", flush=True)
        else:
            print("[sim-config] Keyframe 'home' not found; leaving default initial state.", flush=True)

    if home_hold_target is None:
        home_hold_target = np.zeros(sim.mj_model.nu, dtype=float)

    original_lowcmd_handler = sim.UnitreeSdk2Bridge.LowCmdHandler

    def patched_lowcmd_handler(self, msg):
        self._codex_last_lowcmd_time = time.monotonic()
        self._codex_has_active_lowcmd = is_active_lowcmd(msg)
        if self._codex_has_active_lowcmd:
            original_lowcmd_handler(self, msg)

    sim.UnitreeSdk2Bridge.LowCmdHandler = patched_lowcmd_handler

    def patched_simulation_thread():
        sim.ChannelFactoryInitialize(sim.config.DOMAIN_ID, sim.config.INTERFACE)
        unitree = sim.UnitreeSdk2Bridge(sim.mj_model, sim.mj_data)
        unitree._codex_last_lowcmd_time = 0.0
        unitree._codex_has_active_lowcmd = False

        if sim.config.USE_JOYSTICK:
            unitree.SetupJoystick(device_id=0, js_type=sim.config.JOYSTICK_TYPE)
        if sim.config.PRINT_SCENE_INFORMATION:
            unitree.PrintSceneInformation()

        while sim.viewer.is_running():
            step_start = time.perf_counter()

            sim.locker.acquire()
            if not unitree._codex_has_active_lowcmd:
                q = np.array(sim.mj_data.sensordata[: sim.mj_model.nu], dtype=float)
                dq = np.array(
                    sim.mj_data.sensordata[sim.mj_model.nu : 2 * sim.mj_model.nu],
                    dtype=float,
                )
                sim.mj_data.ctrl[: sim.mj_model.nu] = (
                    args.home_hold_kp * (home_hold_target - q) - args.home_hold_kd * dq
                )

            if sim.config.ENABLE_ELASTIC_BAND:
                if sim.elastic_band.enable:
                    sim.mj_data.xfrc_applied[sim.band_attached_link, :3] = sim.elastic_band.Advance(
                        sim.mj_data.qpos[:3], sim.mj_data.qvel[:3]
                    )

            mujoco.mj_step(sim.mj_model, sim.mj_data)
            sim.locker.release()

            time_until_next_step = sim.mj_model.opt.timestep - (time.perf_counter() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    viewer_thread = sim.Thread(target=sim.PhysicsViewerThread)
    sim_thread = sim.Thread(target=patched_simulation_thread)
    viewer_thread.start()
    sim_thread.start()
    viewer_thread.join()
    sim_thread.join()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
