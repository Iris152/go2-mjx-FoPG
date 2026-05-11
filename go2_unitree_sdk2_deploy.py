#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import math
import select
import signal
import struct
import sys
import threading
import time
import types
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional, Sequence


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
    import numpy as np
    import onnxruntime as ort
    _ensure_unitree_sdk2py_namespace()
    from unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize,
        ChannelPublisher,
        ChannelSubscriber,
    )
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
except ImportError as exc:
    missing = getattr(exc, "name", "unknown")
    raise SystemExit(
        "Missing runtime dependency: "
        f"{missing}. Activate the environment that has numpy, onnxruntime "
        "and unitree_sdk2py installed."
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
POS_STOP_F = 2.146e9
VEL_STOP_F = 16000.0

ACTION_SIZE = 12
BASELINE_OBS_SIZE = 40
FORWARD_OBS_SIZE = 52
KEYFRAME_NAME = "home"
ACTION_SCALE = np.array([0.2, 0.8, 0.8] * 4, dtype=np.float32)

# Policy order used by the MJX training code: FL, FR, RL, RR
# Unitree low-level order used by unitree_mujoco / hardware: FR, FL, RR, RL
POLICY_TO_UNITREE = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], dtype=np.int64)
UNITREE_TO_POLICY = POLICY_TO_UNITREE.copy()
OFFICIAL_STAND_UP_UNITREE = np.array(
    [
        0.00571868,
        0.608813,
        -1.21763,
        -0.00571868,
        0.608813,
        -1.21763,
        0.00571868,
        0.608813,
        -1.21763,
        -0.00571868,
        0.608813,
        -1.21763,
    ],
    dtype=np.float32,
)

PASSIVE_MOTOR_COUNT = 20
LOWCMD_PACK_FMT = "<4B4IH2x" + "B3x5f3I" * 20 + "4B" + "55Bx2I"


def crc32_core(words: Sequence[int]) -> int:
    crc = 0xFFFFFFFF
    polynomial = 0x04C11DB7
    for current in words:
        bit = 1 << 31
        for _ in range(32):
            if crc & 0x80000000:
                crc = ((crc << 1) & 0xFFFFFFFF) ^ polynomial
            else:
                crc = (crc << 1) & 0xFFFFFFFF
            if current & bit:
                crc ^= polynomial
            bit >>= 1
    return crc & 0xFFFFFFFF


def pack_lowcmd_words(cmd) -> list[int]:
    data: list[object] = []
    data.extend(cmd.head)
    data.append(cmd.level_flag)
    data.append(cmd.frame_reserve)
    data.extend(cmd.sn)
    data.extend(cmd.version)
    data.append(cmd.bandwidth)

    for i in range(PASSIVE_MOTOR_COUNT):
        motor = cmd.motor_cmd[i]
        data.append(motor.mode)
        data.append(motor.q)
        data.append(motor.dq)
        data.append(motor.tau)
        data.append(motor.kp)
        data.append(motor.kd)
        data.extend(motor.reserve)

    data.append(cmd.bms_cmd.off)
    data.extend(cmd.bms_cmd.reserve)
    data.extend(cmd.wireless_remote)
    data.extend(cmd.led)
    data.extend(cmd.fan)
    data.append(cmd.gpio)
    data.append(cmd.reserve)
    data.append(0)

    packed = struct.pack(LOWCMD_PACK_FMT, *data)
    words: list[int] = []
    calc_len = (len(packed) >> 2) - 1
    for i in range(calc_len):
        base = i * 4
        word = (
            (packed[base + 3] << 24)
            | (packed[base + 2] << 16)
            | (packed[base + 1] << 8)
            | packed[base]
        )
        words.append(word)
    return words


def compute_lowcmd_crc(cmd) -> int:
    return crc32_core(pack_lowcmd_words(cmd))


def normalize_quaternion_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return quat / norm


def quat_wxyz_to_rotmat(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize_quaternion_wxyz(quat)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def gravity_body_from_quaternion(quat_wxyz: np.ndarray) -> np.ndarray:
    rot = quat_wxyz_to_rotmat(quat_wxyz)
    gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    return (rot.T @ gravity_world).astype(np.float32)


def cos_wave(step_index: np.ndarray, step_period: float, scale: float) -> np.ndarray:
    wave = -np.cos(((2.0 * np.pi) / step_period) * step_index)
    return wave * (scale / 2.0) + (scale / 2.0)


def make_kinematic_ref(step_k: int, control_dt: float, scale: float = 0.3) -> np.ndarray:
    steps = np.arange(step_k, dtype=np.float32)
    step_period = step_k * control_dt
    t = steps * control_dt
    wave = cos_wave(t, step_period, scale)
    leg_block = np.concatenate(
        [
            np.zeros((step_k, 1), dtype=np.float32),
            wave.reshape(step_k, 1),
            -2.0 * wave.reshape(step_k, 1),
        ],
        axis=1,
    )
    block1 = np.concatenate(
        [
            np.zeros((step_k, 3), dtype=np.float32),
            leg_block,
            leg_block,
            np.zeros((step_k, 3), dtype=np.float32),
        ],
        axis=1,
    )
    block2 = np.concatenate(
        [
            leg_block,
            np.zeros((step_k, 3), dtype=np.float32),
            np.zeros((step_k, 3), dtype=np.float32),
            leg_block,
        ],
        axis=1,
    )
    return np.concatenate([block1, block2], axis=0)


def find_home_keyframe_qpos(xml_path: Path, key_name: str = KEYFRAME_NAME) -> np.ndarray:
    root = ET.parse(xml_path).getroot()
    for keyframe in root.findall(".//keyframe"):
        for key in keyframe.findall("key"):
            if key.attrib.get("name") == key_name:
                qpos_text = key.attrib.get("qpos")
                if not qpos_text:
                    raise ValueError(f"keyframe '{key_name}' in {xml_path} has no qpos")
                qpos = np.fromstring(qpos_text, sep=" ", dtype=np.float32)
                if qpos.shape[0] < 19:
                    raise ValueError(f"keyframe '{key_name}' in {xml_path} has invalid qpos length")
                return qpos
    raise ValueError(f"Could not find keyframe '{key_name}' in {xml_path}")


def resolve_training_xml(candidate: Path) -> Path:
    if candidate.exists():
        try:
            qpos = find_home_keyframe_qpos(candidate)
            if qpos.shape[0] >= 19:
                return candidate
        except ValueError:
            pass
    fallback = candidate.parent / "go2.xml"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        "Could not resolve a training XML with a home keyframe. "
        f"Tried: {candidate} and {fallback}"
    )


def resolve_default_onnx(candidates: Sequence[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Could not find any of: " + ", ".join(str(p) for p in candidates))


def ort_variant_path(model_path: Path) -> Path:
    if model_path.name.endswith("_ort.onnx"):
        return model_path
    if model_path.suffix != ".onnx":
        return model_path.with_name(model_path.name + "_ort")
    return model_path.with_name(model_path.stem + "_ort.onnx")


def _load_sanitize_module():
    sanitize_path = SCRIPT_DIR / "sanitize_onnx_for_ort.py"
    spec = importlib.util.spec_from_file_location("sanitize_onnx_for_ort_local", sanitize_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load sanitize helper from {sanitize_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_ort_compatible_model(model_path: Path) -> Path:
    model_path = model_path.resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    ort_path = ort_variant_path(model_path)
    if ort_path.exists():
        return ort_path

    if model_path.name.endswith("_ort.onnx"):
        return model_path

    module = _load_sanitize_module()
    import onnx

    print(f"[onnx] Sanitizing {model_path.name} -> {ort_path.name}", flush=True)
    model = onnx.load(str(model_path))
    model, replaced_expm1, replaced_pg = module.sanitize_model(model)
    print(
        f"[onnx] Replaced {replaced_expm1} Expm1 node(s), {replaced_pg} PreventGradient node(s).",
        flush=True,
    )
    onnx.checker.check_model(model)
    onnx.save(model, str(ort_path))
    return ort_path


def create_onnx_session(model_path: Path) -> ort.InferenceSession:
    candidate_path = ensure_ort_compatible_model(model_path)
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    return ort.InferenceSession(str(candidate_path), sess_options=session_options, providers=["CPUExecutionProvider"])


def run_onnx_policy(session: ort.InferenceSession, obs: np.ndarray) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32)
    input_meta = session.get_inputs()[0]
    if len(input_meta.shape) == 1:
        feed = obs
    elif len(input_meta.shape) == 2:
        feed = obs.reshape(1, -1)
    else:
        raise ValueError(f"Unsupported input shape: {input_meta.shape}")
    output = session.run(None, {input_meta.name: feed})[0]
    output = np.asarray(output, dtype=np.float32)
    return output[0] if output.ndim == 2 else output


def remap_unitree_to_policy(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)[UNITREE_TO_POLICY]


def remap_policy_to_unitree(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    remapped = np.zeros_like(values)
    remapped[POLICY_TO_UNITREE] = values
    return remapped


@dataclass
class RobotState:
    joint_angles_policy: np.ndarray
    joint_speeds_policy: np.ndarray
    gyro_body_policy: np.ndarray
    gravity_body: np.ndarray
    quat_wxyz: np.ndarray
    received_at: float


class Phase(Enum):
    WAIT_START = auto()
    STAND_RAMP = auto()
    STAND_HOLD = auto()
    WAIT_POLICY = auto()
    POLICY_RAMP = auto()
    POLICY = auto()
    FAULT = auto()


class EnterLatch:
    def __init__(self, enabled: bool):
        self.enabled = bool(enabled and sys.stdin.isatty())

    def poll(self) -> bool:
        if not self.enabled:
            return False
        readable, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not readable:
            return False
        sys.stdin.readline()
        return True


class Go2DeployRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.control_dt = float(args.control_dt)
        self.phase = Phase.WAIT_START
        self._shutdown_requested = False
        self._fault_reason: Optional[str] = None

        train_xml = resolve_training_xml(Path(args.train_xml_path).expanduser())
        qpos_home = find_home_keyframe_qpos(train_xml)
        self.action_loc = qpos_home[7:19].astype(np.float32)
        if args.command_center == "official_standup":
            self.command_center_policy = remap_unitree_to_policy(OFFICIAL_STAND_UP_UNITREE)
        else:
            self.command_center_policy = self.action_loc.copy()

        kin_ref = make_kinematic_ref(step_k=args.step_k, control_dt=self.control_dt, scale=args.gait_scale)
        self.kinematic_ref_qpos = (kin_ref + self.action_loc[None, :]).astype(np.float32)
        self.l_cycle = int(self.kinematic_ref_qpos.shape[0])

        baseline_path = Path(args.baseline_onnx).expanduser()
        forward_path = Path(args.forward_onnx).expanduser() if args.forward_onnx else None
        self.baseline_session = create_onnx_session(baseline_path)
        self.forward_session = create_onnx_session(forward_path) if forward_path else None

        self.robot_state: Optional[RobotState] = None
        self.robot_state_lock = threading.Lock()
        self.lowstate_pub_seen = threading.Event()

        self.last_joint_target_policy = self.command_center_policy.copy()
        self.control_steps = 0
        self.phase_started_at = time.monotonic()
        self.phase_start_joint_policy: Optional[np.ndarray] = None
        self.desired_joint_target_policy = self.command_center_policy.copy()
        self.desired_kp = 0.0
        self.desired_kd = 0.0
        self.desired_passive = True

        self.cmd_pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.cmd_pub.Init()
        self.lowstate_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_sub.Init(self._lowstate_cb, 10)

        self.lowcmd = unitree_go_msg_dds__LowCmd_()
        self._init_lowcmd()

        self.start_latch = EnterLatch(not args.auto_start)
        self.policy_latch = EnterLatch(not args.auto_policy)
        self._start_prompt_shown = False
        self._policy_prompt_shown = False

    def _init_lowcmd(self) -> None:
        self.lowcmd.head[0] = 0xFE
        self.lowcmd.head[1] = 0xEF
        self.lowcmd.level_flag = 0xFF
        self.lowcmd.gpio = 0
        for i in range(PASSIVE_MOTOR_COUNT):
            self.lowcmd.motor_cmd[i].mode = 0x01
            self.lowcmd.motor_cmd[i].q = POS_STOP_F
            self.lowcmd.motor_cmd[i].dq = VEL_STOP_F
            self.lowcmd.motor_cmd[i].kp = 0.0
            self.lowcmd.motor_cmd[i].kd = 0.0
            self.lowcmd.motor_cmd[i].tau = 0.0

    def _transition(self, phase: Phase, note: str) -> None:
        self.phase = phase
        self.phase_started_at = time.monotonic()
        print(f"[phase] {phase.name}: {note}", flush=True)

    def _lowstate_cb(self, msg: LowState_) -> None:
        q_unitree = np.array([msg.motor_state[i].q for i in range(ACTION_SIZE)], dtype=np.float32)
        dq_unitree = np.array([msg.motor_state[i].dq for i in range(ACTION_SIZE)], dtype=np.float32)
        quat_wxyz = np.array(msg.imu_state.quaternion, dtype=np.float32)
        gyro_unitree = np.array(msg.imu_state.gyroscope, dtype=np.float32)

        state = RobotState(
            joint_angles_policy=remap_unitree_to_policy(q_unitree),
            joint_speeds_policy=remap_unitree_to_policy(dq_unitree),
            gyro_body_policy=gyro_unitree.astype(np.float32),
            gravity_body=gravity_body_from_quaternion(quat_wxyz),
            quat_wxyz=normalize_quaternion_wxyz(quat_wxyz),
            received_at=time.monotonic(),
        )
        with self.robot_state_lock:
            self.robot_state = state
        self.lowstate_pub_seen.set()

    def get_robot_state(self) -> Optional[RobotState]:
        with self.robot_state_lock:
            if self.robot_state is None:
                return None
            return RobotState(
                joint_angles_policy=self.robot_state.joint_angles_policy.copy(),
                joint_speeds_policy=self.robot_state.joint_speeds_policy.copy(),
                gyro_body_policy=self.robot_state.gyro_body_policy.copy(),
                gravity_body=self.robot_state.gravity_body.copy(),
                quat_wxyz=self.robot_state.quat_wxyz.copy(),
                received_at=self.robot_state.received_at,
            )

    def wait_for_connection(self) -> None:
        print("[info] Waiting for rt/lowstate ...", flush=True)
        while not self.lowstate_pub_seen.wait(timeout=0.2):
            if self._shutdown_requested:
                return
        print("[info] Connected to rt/lowstate.", flush=True)

    def publish_passive(self) -> None:
        for i in range(PASSIVE_MOTOR_COUNT):
            self.lowcmd.motor_cmd[i].mode = 0x01
            self.lowcmd.motor_cmd[i].q = POS_STOP_F
            self.lowcmd.motor_cmd[i].dq = VEL_STOP_F
            self.lowcmd.motor_cmd[i].kp = 0.0
            self.lowcmd.motor_cmd[i].kd = 0.0
            self.lowcmd.motor_cmd[i].tau = 0.0
        self.lowcmd.crc = compute_lowcmd_crc(self.lowcmd)
        self.cmd_pub.Write(self.lowcmd)

    def set_passive_command(self) -> None:
        self.desired_passive = True
        self.desired_kp = 0.0
        self.desired_kd = 0.0

    def set_joint_target_command(self, joint_target_policy: np.ndarray, kp: float, kd: float) -> None:
        self.desired_joint_target_policy = np.asarray(joint_target_policy, dtype=np.float32).copy()
        self.desired_kp = float(kp)
        self.desired_kd = float(kd)
        self.desired_passive = False

    def publish_joint_target(self, joint_target_policy: np.ndarray, kp: float, kd: float) -> None:
        joint_target_unitree = remap_policy_to_unitree(joint_target_policy)
        for i in range(PASSIVE_MOTOR_COUNT):
            self.lowcmd.motor_cmd[i].mode = 0x01
            self.lowcmd.motor_cmd[i].tau = 0.0
            if i < ACTION_SIZE:
                self.lowcmd.motor_cmd[i].q = float(joint_target_unitree[i])
                self.lowcmd.motor_cmd[i].dq = 0.0
                self.lowcmd.motor_cmd[i].kp = float(kp)
                self.lowcmd.motor_cmd[i].kd = float(kd)
            else:
                self.lowcmd.motor_cmd[i].q = POS_STOP_F
                self.lowcmd.motor_cmd[i].dq = VEL_STOP_F
                self.lowcmd.motor_cmd[i].kp = 0.0
                self.lowcmd.motor_cmd[i].kd = 0.0
        self.lowcmd.crc = compute_lowcmd_crc(self.lowcmd)
        self.cmd_pub.Write(self.lowcmd)

    def publish_current_command(self) -> None:
        if self.desired_passive:
            self.publish_passive()
        else:
            self.publish_joint_target(
                self.desired_joint_target_policy,
                self.desired_kp,
                self.desired_kd,
            )

    def _tilt_angle_rad(self, gravity_body: np.ndarray) -> float:
        aligned = float(np.clip(-gravity_body[2], -1.0, 1.0))
        return math.acos(aligned)

    def _check_safety(self, state: RobotState) -> None:
        if time.monotonic() - state.received_at > self.args.lowstate_timeout:
            self._enter_fault("lowstate timeout")
            return
        tilt = self._tilt_angle_rad(state.gravity_body)
        if tilt > self.args.tilt_limit_rad:
            self._enter_fault(f"body tilt too large: {tilt:.3f} rad")

    def _enter_fault(self, reason: str) -> None:
        if self.phase == Phase.FAULT:
            return
        self._fault_reason = reason
        self._transition(Phase.FAULT, reason)

    def _build_inner_obs(self, state: RobotState) -> np.ndarray:
        kin_ref = self.kinematic_ref_qpos[self.control_steps % self.l_cycle]
        obs = np.concatenate(
            [
                np.array([state.gyro_body_policy[2] * 0.25], dtype=np.float32),
                state.gravity_body.astype(np.float32),
                (state.joint_angles_policy - self.action_loc).astype(np.float32),
                self.last_joint_target_policy.astype(np.float32),
                kin_ref.astype(np.float32),
            ],
            axis=0,
        )
        if obs.shape[0] != BASELINE_OBS_SIZE:
            raise ValueError(f"baseline obs mismatch: {obs.shape[0]} != {BASELINE_OBS_SIZE}")
        return np.clip(obs, -100.0, 100.0).astype(np.float32)

    def _compute_policy_target(self, state: RobotState) -> np.ndarray:
        inner_obs = self._build_inner_obs(state)
        baseline_action = run_onnx_policy(self.baseline_session, inner_obs)
        baseline_action = np.clip(np.asarray(baseline_action, dtype=np.float32), -1.0, 1.0)

        if self.args.mode == "trot":
            final_action = baseline_action
        else:
            if self.forward_session is None:
                raise RuntimeError("forward mode requires forward onnx")
            outer_obs = np.concatenate([inner_obs, baseline_action], axis=0).astype(np.float32)
            if outer_obs.shape[0] != FORWARD_OBS_SIZE:
                raise ValueError(f"forward obs mismatch: {outer_obs.shape[0]} != {FORWARD_OBS_SIZE}")
            residual_action = run_onnx_policy(self.forward_session, outer_obs)
            residual_action = np.clip(np.asarray(residual_action, dtype=np.float32), -1.0, 1.0)
            final_action = residual_action + baseline_action
            if self.args.clip_final_action:
                final_action = np.clip(final_action, -1.0, 1.0)

        return (self.command_center_policy + final_action * ACTION_SCALE).astype(np.float32)

    def _phase_elapsed(self) -> float:
        return time.monotonic() - self.phase_started_at

    def request_shutdown(self) -> None:
        self._shutdown_requested = True

    def _maybe_start(self, state: RobotState) -> None:
        if self.args.auto_start:
            self.phase_start_joint_policy = state.joint_angles_policy.copy()
            self._transition(Phase.STAND_RAMP, "auto start")
            return
        if not self._start_prompt_shown:
            print("[input] Press Enter to start stand-up.", flush=True)
            self._start_prompt_shown = True
        if self.start_latch.poll():
            self.phase_start_joint_policy = state.joint_angles_policy.copy()
            self._transition(Phase.STAND_RAMP, "manual start")

    def _stand_target(self) -> np.ndarray:
        if self.phase_start_joint_policy is None:
            raise RuntimeError("phase_start_joint_policy is not initialized")
        alpha = min(self._phase_elapsed() / self.args.stand_ramp_duration, 1.0)
        return ((1.0 - alpha) * self.phase_start_joint_policy + alpha * self.command_center_policy).astype(np.float32)

    def _policy_ramp_target(self, policy_target: np.ndarray) -> np.ndarray:
        alpha = min(self._phase_elapsed() / self.args.policy_ramp_duration, 1.0)
        return ((1.0 - alpha) * self.command_center_policy + alpha * policy_target).astype(np.float32)

    def loop_once(self) -> None:
        state = self.get_robot_state()
        if state is None:
            self.set_passive_command()
            return

        if self.phase not in (Phase.WAIT_START, Phase.FAULT):
            self._check_safety(state)

        if self.phase == Phase.WAIT_START:
            self.set_passive_command()
            self._maybe_start(state)
            return

        if self.phase == Phase.STAND_RAMP:
            target = self._stand_target()
            self.set_joint_target_command(target, self.args.stand_kp, self.args.stand_kd)
            self.last_joint_target_policy = target.copy()
            if self._phase_elapsed() >= self.args.stand_ramp_duration:
                self._transition(Phase.STAND_HOLD, "stand pose reached")
            return

        if self.phase == Phase.STAND_HOLD:
            self.set_joint_target_command(self.command_center_policy, self.args.stand_kp, self.args.stand_kd)
            self.last_joint_target_policy = self.command_center_policy.copy()
            if self._phase_elapsed() >= self.args.stand_hold_duration:
                if self.args.auto_policy:
                    self._transition(Phase.POLICY_RAMP, "auto policy start")
                else:
                    self._transition(Phase.WAIT_POLICY, "waiting for policy arm")
            return

        if self.phase == Phase.WAIT_POLICY:
            self.set_joint_target_command(self.command_center_policy, self.args.stand_kp, self.args.stand_kd)
            self.last_joint_target_policy = self.command_center_policy.copy()
            if not self._policy_prompt_shown:
                print("[input] Press Enter to start the policy.", flush=True)
                self._policy_prompt_shown = True
            if self.policy_latch.poll():
                self._transition(Phase.POLICY_RAMP, "manual policy arm")
            return

        if self.phase == Phase.POLICY_RAMP:
            target = self._compute_policy_target(state)
            blended = self._policy_ramp_target(target)
            self.set_joint_target_command(blended, self.args.policy_kp, self.args.policy_kd)
            self.last_joint_target_policy = blended.copy()
            self.control_steps += 1
            if self._phase_elapsed() >= self.args.policy_ramp_duration:
                self._transition(Phase.POLICY, "policy fully enabled")
            return

        if self.phase == Phase.POLICY:
            target = self._compute_policy_target(state)
            self.set_joint_target_command(target, self.args.policy_kp, self.args.policy_kd)
            self.last_joint_target_policy = target.copy()
            self.control_steps += 1
            return

        if self.phase == Phase.FAULT:
            self.set_joint_target_command(self.command_center_policy, self.args.fault_kp, self.args.fault_kd)
            self.last_joint_target_policy = self.command_center_policy.copy()
            return

        raise RuntimeError(f"Unhandled phase: {self.phase}")

    def run(self) -> None:
        self.wait_for_connection()
        next_policy_tick = time.perf_counter()
        next_cmd_tick = next_policy_tick
        while not self._shutdown_requested:
            now = time.perf_counter()

            if now >= next_policy_tick:
                loop_start = time.perf_counter()
                self.loop_once()
                next_policy_tick += self.control_dt
                if next_policy_tick <= now:
                    next_policy_tick = now + self.control_dt
                if self.args.print_timing and self.phase == Phase.POLICY:
                    elapsed_ms = (time.perf_counter() - loop_start) * 1000.0
                    print(f"[timing] {elapsed_ms:.3f} ms", flush=True)

            now = time.perf_counter()
            if now >= next_cmd_tick:
                self.publish_current_command()
                next_cmd_tick += self.args.lowcmd_dt
                if next_cmd_tick <= now:
                    next_cmd_tick = now + self.args.lowcmd_dt

            sleep_dt = min(next_policy_tick, next_cmd_tick) - time.perf_counter()
            if sleep_dt > 0:
                time.sleep(sleep_dt)

    def graceful_shutdown(self) -> None:
        print("[shutdown] Sending stand pose, then passive.", flush=True)
        deadline = time.perf_counter() + self.args.shutdown_stand_duration
        while time.perf_counter() < deadline:
            self.publish_joint_target(self.command_center_policy, self.args.stand_kp, self.args.stand_kd)
            time.sleep(self.control_dt)
        deadline = time.perf_counter() + self.args.shutdown_passive_duration
        while time.perf_counter() < deadline:
            self.publish_passive()
            time.sleep(self.control_dt)


def parse_args() -> argparse.Namespace:
    default_baseline = resolve_default_onnx(
        [
            SCRIPT_DIR / "exported_onnx" / "trotting_2hz_policy_ort.onnx",
            SCRIPT_DIR / "exported_onnx" / "trotting_2hz_policy.onnx",
        ]
    )
    default_forward = resolve_default_onnx(
        [
            SCRIPT_DIR / "exported_onnx" / "forward_locomotion_policy_ort.onnx",
            SCRIPT_DIR / "exported_onnx" / "forward_locomotion_policy.onnx",
        ]
    )
    default_train_xml = SCRIPT_DIR / "mujoco_menagerie" / "unitree_go2" / "scene_mjx.xml"

    parser = argparse.ArgumentParser(description="GO2 low-level deploy runner for unitree_mujoco and real robot")
    parser.add_argument("--mode", choices=["forward", "trot"], default="forward")
    parser.add_argument("--baseline_onnx", type=str, default=str(default_baseline))
    parser.add_argument("--forward_onnx", type=str, default=str(default_forward))
    parser.add_argument("--train_xml_path", type=str, default=str(default_train_xml))
    parser.add_argument(    
        "--command_center",
        choices=["official_standup", "train_home"],
        default="train_home",
        help="Reference pose used for stand / output targets. Observation center remains the training home pose.",
    )
    parser.add_argument(
        "--clip_final_action",
        action="store_true",
        help="Clip baseline+residual before scaling. Default is off to match GO2_train.ipynb/local training.",
    )
    parser.add_argument("--network", type=str, default="lo", help="Use lo for unitree_mujoco, e.g. enp5s0 for the real robot")
    parser.add_argument("--domain_id", type=int, default=1, help="Use 1 for simulation, 0 for the real robot")
    parser.add_argument("--control_dt", type=float, default=0.02)
    parser.add_argument("--lowcmd_dt", type=float, default=0.002, help="Low-level command resend period; keep small for unitree_mujoco / hardware bridges")
    parser.add_argument("--step_k", type=int, default=13)
    parser.add_argument("--gait_scale", type=float, default=0.3)
    parser.add_argument("--stand_ramp_duration", type=float, default=2.5)
    parser.add_argument("--stand_hold_duration", type=float, default=0.5)
    parser.add_argument("--policy_ramp_duration", type=float, default=2.0)
    parser.add_argument("--stand_kp", type=float, default=60.0)
    parser.add_argument("--stand_kd", type=float, default=5.0)
    parser.add_argument("--policy_kp", type=float, default=50.0)
    parser.add_argument("--policy_kd", type=float, default=3.5)
    parser.add_argument("--fault_kp", type=float, default=60.0)
    parser.add_argument("--fault_kd", type=float, default=5.0)
    parser.add_argument("--tilt_limit_rad", type=float, default=0.9)
    parser.add_argument("--lowstate_timeout", type=float, default=0.2)
    parser.add_argument("--shutdown_stand_duration", type=float, default=0.5)
    parser.add_argument("--shutdown_passive_duration", type=float, default=0.2)
    parser.add_argument("--auto_start", action="store_true", help="Start stand-up without waiting for Enter")
    parser.add_argument("--auto_policy", action="store_true", help="Start policy automatically after stand-up")
    parser.add_argument("--print_timing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(
        "[config] network=%s domain_id=%s mode=%s command_center=%s clip_final_action=%s baseline=%s forward=%s"
        % (
            args.network,
            args.domain_id,
            args.mode,
            args.command_center,
            args.clip_final_action,
            args.baseline_onnx,
            args.forward_onnx,
        ),
        flush=True,
    )
    print("[config] policy order -> unitree order map:", POLICY_TO_UNITREE.tolist(), flush=True)

    ChannelFactoryInitialize(args.domain_id, args.network)
    runner = Go2DeployRunner(args)

    def _signal_handler(signum, _frame):
        print(f"[signal] received {signum}", flush=True)
        runner.request_shutdown()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        runner.run()
    except KeyboardInterrupt:
        runner.request_shutdown()
    finally:
        runner.graceful_shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
