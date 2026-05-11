#!/usr/bin/env python3
from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _ensure_unitree_sdk2py_namespace() -> None:
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


try:
    import mujoco
    import numpy as np

    _ensure_unitree_sdk2py_namespace()
    from unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize,
        ChannelPublisher,
        ChannelSubscriber,
    )
    from unitree_sdk2py.idl.default import (
        unitree_go_msg_dds__LowCmd_,
        unitree_go_msg_dds__LowState_,
    )
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
except ImportError as exc:
    missing = getattr(exc, "name", "unknown")
    raise SystemExit(
        "Missing runtime dependency: "
        f"{missing}. Activate the environment that has mujoco, numpy "
        "and unitree_sdk2py installed."
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
SCENE_XML = SCRIPT_DIR / "mujoco_menagerie" / "unitree_go2" / "scene_mjx.xml"

PASSIVE_MOTOR_COUNT = 20
POS_STOP_F = 2.146e9
VEL_STOP_F = 16000.0

# Policy / local MuJoCo order: FL, FR, RL, RR
# Unitree low-level order: FR, FL, RR, RL
POLICY_TO_UNITREE = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], dtype=np.int64)
UNITREE_TO_POLICY = POLICY_TO_UNITREE.copy()

POLICY_JOINT_NAMES = [
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
]


@dataclass
class CommandSnapshot:
    active: bool
    q: np.ndarray
    dq: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    tau: np.ndarray
    received_at: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local MuJoCo <-> Unitree DDS low-level simulator for GO2"
    )
    parser.add_argument("--scene_xml", type=str, default=str(SCENE_XML))
    parser.add_argument("--network", type=str, default="lo")
    parser.add_argument("--domain_id", type=int, default=1)
    parser.add_argument("--sim_dt", type=float, default=0.002)
    parser.add_argument("--viewer_dt", type=float, default=0.02)
    parser.add_argument("--cmd_timeout", type=float, default=0.25)
    parser.add_argument("--idle_kp", type=float, default=80.0)
    parser.add_argument("--idle_kd", type=float, default=6.0)
    parser.add_argument(
        "--initial_pose",
        choices=["home", "crouch", "prone"],
        default="home",
        help="Initial simulator pose before active LowCmd is received.",
    )
    parser.add_argument(
        "--idle_target",
        choices=["initial", "home"],
        default="initial",
        help="Joint target used while no active LowCmd is available.",
    )
    parser.add_argument(
        "--control_mode",
        choices=["auto", "position_servo", "pd_torque"],
        default="auto",
        help=(
            "auto uses position_servo for menagerie *_mjx.xml general actuators "
            "and pd_torque for torque-motor XMLs."
        ),
    )
    parser.add_argument(
        "--servo_kp",
        type=float,
        default=230.0,
        help="Position-servo gain applied to menagerie go2_mjx/scene_mjx XMLs.",
    )
    parser.add_argument(
        "--servo_kd",
        type=float,
        default=0.5,
        help="Position-servo damping term applied to menagerie go2_mjx/scene_mjx XMLs.",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--max_time",
        type=float,
        default=0.0,
        help="Exit after this many seconds; 0 means run until interrupted.",
    )
    return parser.parse_args()


class Go2MujocoDdsSim:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.scene_xml = Path(args.scene_xml).expanduser().resolve()
        if not self.scene_xml.exists():
            raise FileNotFoundError(f"Scene XML not found: {self.scene_xml}")

        self.model = mujoco.MjModel.from_xml_path(str(self.scene_xml))
        self.model.opt.timestep = float(args.sim_dt)

        self.base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        if self.base_body_id < 0:
            self.base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        if self.base_body_id < 0:
            raise RuntimeError("Could not find base body 'base' or 'base_link'.")

        self.home_key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if self.home_key_id < 0:
            raise RuntimeError("Keyframe 'home' not found in the MuJoCo model.")

        self.qpos_adr = []
        self.qvel_adr = []
        self.actuator_id = []
        for joint_name in POLICY_JOINT_NAMES:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise RuntimeError(f"Joint not found in MuJoCo model: {joint_name}")
            self.qpos_adr.append(int(self.model.jnt_qposadr[joint_id]))
            self.qvel_adr.append(int(self.model.jnt_dofadr[joint_id]))
        for actuator_name in [name.replace("_joint", "") for name in POLICY_JOINT_NAMES]:
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            if actuator_id < 0:
                raise RuntimeError(f"Actuator not found in MuJoCo model: {actuator_name}")
            self.actuator_id.append(int(actuator_id))

        self.qpos_adr = np.asarray(self.qpos_adr, dtype=np.int64)
        self.qvel_adr = np.asarray(self.qvel_adr, dtype=np.int64)
        self.actuator_id = np.asarray(self.actuator_id, dtype=np.int64)

        self.control_mode = self._resolve_control_mode(args.control_mode)
        if self.control_mode == "position_servo":
            self.model.actuator_gainprm[self.actuator_id, 0] = float(args.servo_kp)
            self.model.actuator_biasprm[self.actuator_id, 1] = -float(args.servo_kp)
            self.model.actuator_biasprm[self.actuator_id, 2] = -float(args.servo_kd)

        self.data = mujoco.MjData(self.model)
        self.ctrl_min = self.model.actuator_ctrlrange[self.actuator_id, 0].astype(np.float32)
        self.ctrl_max = self.model.actuator_ctrlrange[self.actuator_id, 1].astype(np.float32)

        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key_id)
        self.home_q = self.data.qpos[self.qpos_adr].astype(np.float32).copy()
        self.initial_q = self._resolve_initial_q(args.initial_pose)
        self.data.qpos[self.qpos_adr] = self.initial_q
        if args.initial_pose == "crouch":
            self.data.qpos[2] = 0.20
        elif args.initial_pose == "prone":
            self.data.qpos[2] = 0.13
        mujoco.mj_forward(self.model, self.data)
        self.idle_q = self.home_q.copy() if args.idle_target == "home" else self.initial_q.copy()

        self.lowstate_pub = ChannelPublisher("rt/lowstate", LowState_)
        self.lowstate_pub.Init()
        self.lowcmd_sub = ChannelSubscriber("rt/lowcmd", LowCmd_)
        self.lowcmd_sub.Init(self._lowcmd_cb, 10)

        self.lowstate = unitree_go_msg_dds__LowState_()
        self._init_lowstate_template()

        self._cmd_lock = threading.Lock()
        self._latest_cmd = CommandSnapshot(
            active=False,
            q=self.home_q.copy(),
            dq=np.zeros(12, dtype=np.float32),
            kp=np.zeros(12, dtype=np.float32),
            kd=np.zeros(12, dtype=np.float32),
            tau=np.zeros(12, dtype=np.float32),
            received_at=0.0,
        )
        self._lowcmd_seen = False
        self._shutdown = False
        self._sim_steps = 0

    def _resolve_control_mode(self, requested: str) -> str:
        if requested != "auto":
            return requested
        bias = self.model.actuator_biasprm[self.actuator_id]
        if np.any(np.abs(bias[:, 1]) > 1e-6):
            return "position_servo"
        return "pd_torque"

    def _resolve_initial_q(self, initial_pose: str) -> np.ndarray:
        if initial_pose == "home":
            return self.home_q.copy()
        if initial_pose == "crouch":
            return np.asarray([0.0, 1.25, -2.45] * 4, dtype=np.float32)
        if initial_pose == "prone":
            return np.asarray([0.0, 1.45, -2.60] * 4, dtype=np.float32)
        raise ValueError(f"Unsupported initial pose: {initial_pose}")

    def _init_lowstate_template(self) -> None:
        self.lowstate.head[0] = 0xFE
        self.lowstate.head[1] = 0xEF
        self.lowstate.level_flag = 0xFF
        for i in range(PASSIVE_MOTOR_COUNT):
            self.lowstate.motor_state[i].mode = 0x01
            self.lowstate.motor_state[i].q = 0.0
            self.lowstate.motor_state[i].dq = 0.0
            self.lowstate.motor_state[i].ddq = 0.0
            self.lowstate.motor_state[i].tau_est = 0.0
            self.lowstate.motor_state[i].temperature = 0
            self.lowstate.motor_state[i].lost = 0

    def request_shutdown(self) -> None:
        self._shutdown = True

    def _is_active_motor_cmd(self, motor_cmd) -> bool:
        if int(motor_cmd.mode) != 0x01:
            return False
        if abs(float(motor_cmd.q) - POS_STOP_F) < 1e3:
            return False
        if abs(float(motor_cmd.dq) - VEL_STOP_F) < 1e-3:
            return False
        return True

    def _lowcmd_cb(self, msg: LowCmd_) -> None:
        q_unitree = np.array([msg.motor_cmd[i].q for i in range(12)], dtype=np.float32)
        dq_unitree = np.array([msg.motor_cmd[i].dq for i in range(12)], dtype=np.float32)
        kp_unitree = np.array([msg.motor_cmd[i].kp for i in range(12)], dtype=np.float32)
        kd_unitree = np.array([msg.motor_cmd[i].kd for i in range(12)], dtype=np.float32)
        tau_unitree = np.array([msg.motor_cmd[i].tau for i in range(12)], dtype=np.float32)

        active = any(self._is_active_motor_cmd(msg.motor_cmd[i]) for i in range(12))
        snapshot = CommandSnapshot(
            active=active,
            q=q_unitree[UNITREE_TO_POLICY].copy(),
            dq=dq_unitree[UNITREE_TO_POLICY].copy(),
            kp=kp_unitree[UNITREE_TO_POLICY].copy(),
            kd=kd_unitree[UNITREE_TO_POLICY].copy(),
            tau=tau_unitree[UNITREE_TO_POLICY].copy(),
            received_at=time.monotonic(),
        )
        with self._cmd_lock:
            self._latest_cmd = snapshot
        if not self._lowcmd_seen:
            self._lowcmd_seen = True
            print("[sim] First rt/lowcmd received.", flush=True)

    def _get_command(self) -> CommandSnapshot:
        with self._cmd_lock:
            return CommandSnapshot(
                active=self._latest_cmd.active,
                q=self._latest_cmd.q.copy(),
                dq=self._latest_cmd.dq.copy(),
                kp=self._latest_cmd.kp.copy(),
                kd=self._latest_cmd.kd.copy(),
                tau=self._latest_cmd.tau.copy(),
                received_at=self._latest_cmd.received_at,
            )

    def _read_policy_joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        q = self.data.qpos[self.qpos_adr].astype(np.float32)
        dq = self.data.qvel[self.qvel_adr].astype(np.float32)
        return q, dq

    def _compute_control(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        cmd = self._get_command()
        cmd_is_fresh = (time.monotonic() - cmd.received_at) <= self.args.cmd_timeout

        if self.control_mode == "position_servo":
            if cmd.active and cmd_is_fresh:
                ctrl = cmd.q
            else:
                ctrl = self.idle_q
        else:
            if cmd.active and cmd_is_fresh:
                ctrl = cmd.tau + cmd.kp * (cmd.q - q) + cmd.kd * (cmd.dq - dq)
            else:
                ctrl = self.args.idle_kp * (self.idle_q - q) - self.args.idle_kd * dq

        ctrl = np.clip(ctrl, self.ctrl_min, self.ctrl_max)
        return ctrl.astype(np.float32)

    def _publish_lowstate(self) -> None:
        q_policy, dq_policy = self._read_policy_joint_state()
        actuator_force_policy = self.data.actuator_force[self.actuator_id].astype(np.float32)

        quat_wxyz = self.data.qpos[3:7].astype(np.float32)
        body_vel = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.base_body_id,
            body_vel,
            1,
        )
        gyro_body = body_vel[:3].astype(np.float32)
        linvel_body = body_vel[3:].astype(np.float32)

        q_unitree = q_policy[POLICY_TO_UNITREE]
        dq_unitree = dq_policy[POLICY_TO_UNITREE]
        tau_unitree = actuator_force_policy[POLICY_TO_UNITREE]

        for i in range(12):
            motor_state = self.lowstate.motor_state[i]
            motor_state.mode = 0x01
            motor_state.q = float(q_unitree[i])
            motor_state.dq = float(dq_unitree[i])
            motor_state.ddq = 0.0
            motor_state.tau_est = float(tau_unitree[i])
            motor_state.temperature = 0
            motor_state.lost = 0

        self.lowstate.imu_state.quaternion[0] = float(quat_wxyz[0])
        self.lowstate.imu_state.quaternion[1] = float(quat_wxyz[1])
        self.lowstate.imu_state.quaternion[2] = float(quat_wxyz[2])
        self.lowstate.imu_state.quaternion[3] = float(quat_wxyz[3])

        self.lowstate.imu_state.gyroscope[0] = float(gyro_body[0])
        self.lowstate.imu_state.gyroscope[1] = float(gyro_body[1])
        self.lowstate.imu_state.gyroscope[2] = float(gyro_body[2])

        self.lowstate.imu_state.accelerometer[0] = float(linvel_body[0] / max(self.args.sim_dt, 1e-6))
        self.lowstate.imu_state.accelerometer[1] = float(linvel_body[1] / max(self.args.sim_dt, 1e-6))
        self.lowstate.imu_state.accelerometer[2] = float(linvel_body[2] / max(self.args.sim_dt, 1e-6))

        self.lowstate.tick = int(self._sim_steps)
        self.lowstate.crc = 0
        self.lowstate_pub.Write(self.lowstate)

    def step(self) -> None:
        q, dq = self._read_policy_joint_state()
        ctrl_policy = self._compute_control(q, dq)
        self.data.ctrl[self.actuator_id] = ctrl_policy
        mujoco.mj_step(self.model, self.data)
        self._sim_steps += 1
        self._publish_lowstate()

    def run(self) -> None:
        print(
            "[sim-config] scene=%s network=%s domain_id=%s sim_dt=%.4f viewer_dt=%.4f"
            % (
                self.scene_xml,
                self.args.network,
                self.args.domain_id,
                self.args.sim_dt,
                self.args.viewer_dt,
            ),
            flush=True,
        )
        print("[sim-config] unitree order -> policy order map:", UNITREE_TO_POLICY.tolist(), flush=True)
        print(
            "[sim-config] control_mode=%s servo_kp=%.1f servo_kd=%.2f initial_pose=%s idle_target=%s"
            % (
                self.control_mode,
                self.args.servo_kp,
                self.args.servo_kd,
                self.args.initial_pose,
                self.args.idle_target,
            ),
            flush=True,
        )
        print("[sim] Publishing rt/lowstate and listening on rt/lowcmd.", flush=True)

        t0 = time.monotonic()
        next_step = time.perf_counter()

        if self.args.headless:
            while not self._shutdown:
                if self.args.max_time > 0.0 and (time.monotonic() - t0) >= self.args.max_time:
                    break
                self.step()
                next_step += self.args.sim_dt
                sleep_dt = next_step - time.perf_counter()
                if sleep_dt > 0.0:
                    time.sleep(sleep_dt)
                else:
                    next_step = time.perf_counter()
            return

        import mujoco.viewer

        next_viewer_sync = time.perf_counter()
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running() and not self._shutdown:
                if self.args.max_time > 0.0 and (time.monotonic() - t0) >= self.args.max_time:
                    break
                self.step()
                now = time.perf_counter()
                if now >= next_viewer_sync:
                    viewer.sync()
                    next_viewer_sync = now + self.args.viewer_dt
                next_step += self.args.sim_dt
                sleep_dt = next_step - time.perf_counter()
                if sleep_dt > 0.0:
                    time.sleep(sleep_dt)
                else:
                    next_step = time.perf_counter()


def main() -> int:
    args = parse_args()
    ChannelFactoryInitialize(args.domain_id, args.network)
    sim = Go2MujocoDdsSim(args)

    def _signal_handler(signum, _frame):
        print(f"[signal] received {signum}", flush=True)
        sim.request_shutdown()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    sim.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
