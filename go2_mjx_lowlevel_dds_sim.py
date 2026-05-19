#!/usr/bin/env python3
"""基于 mujoco_menagerie 训练 XML 的 GO2 低层 DDS 仿真器。

这个脚本通常作为“双终端部署验证”的 Terminal B：
1. 订阅 `rt/lowcmd`，接收 deploy runner 发出的 Unitree LowCmd。
2. 在本地 MuJoCo 里推进 GO2 仿真。
3. 发布 `rt/lowstate`，让 deploy runner 以为自己正在和 Unitree 低层通信。

默认 `control_mode=auto` 会在 menagerie 的 *_mjx.xml 上走 position_servo：
LowCmd.q -> MuJoCo ctrl，仿真器内部用 servo_kp/servo_kd 作为 XML actuator 增益。
这种模式用于保持“训练 XML 同源验证”。如果使用 torque-motor XML，则可切到 pd_torque，
此时才会按 LowCmd 的 kp/kd/tau 显式计算力矩。
"""

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
    """兼容某些 unitree_sdk2py 安装缺少标准包入口的环境。"""
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

# Unitree LowCmd 里的特殊停止值。收到这些值时，不应把它当作有效目标。
POS_STOP_F = 2.146e9
VEL_STOP_F = 16000.0

# Policy / local MuJoCo order: FL, FR, RL, RR
# Unitree low-level order: FR, FL, RR, RL
# 仿真内部按训练/策略顺序组织关节，DDS topic 按 Unitree 低层顺序组织关节。
POLICY_TO_UNITREE = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], dtype=np.int64)
UNITREE_TO_POLICY = POLICY_TO_UNITREE.copy()

# 训练 XML 中 12 个腿部关节的策略顺序。
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


def rotate_world_to_body(vec_world: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """把世界系向量旋到机体系，和训练 observation 的角速度坐标一致。"""
    rot = quat_wxyz_to_rotmat(quat_wxyz)
    return (rot.T @ np.asarray(vec_world, dtype=np.float32)).astype(np.float32)


@dataclass
class CommandSnapshot:
    """最近一次收到的 LowCmd，已经转换为策略/本地 MuJoCo 顺序。"""

    active: bool
    q: np.ndarray
    dq: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    tau: np.ndarray
    received_at: float


def parse_args() -> argparse.Namespace:
    """解析仿真器参数。

    常用双终端验证：
    Terminal B: python launch_unitree_mujoco_python_sim.py --backend menagerie --network lo --domain_id 1
    Terminal A: python go2_unitree_sdk2_deploy.py --network lo --domain_id 1
    """
    parser = argparse.ArgumentParser(
        description="Local MuJoCo <-> Unitree DDS low-level simulator for GO2"
    )

    # MuJoCo 和 DDS 基础配置。
    parser.add_argument("--scene_xml", type=str, default=str(SCENE_XML))
    parser.add_argument("--network", type=str, default="lo")
    parser.add_argument("--domain_id", type=int, default=1)
    parser.add_argument("--sim_dt", type=float, default=0.002)
    parser.add_argument("--viewer_dt", type=float, default=0.02)

    # LowCmd 超时后会退回 idle_q，避免仿真继续执行过期命令。
    parser.add_argument("--cmd_timeout", type=float, default=0.25)
    parser.add_argument("--idle_kp", type=float, default=80.0)
    parser.add_argument("--idle_kd", type=float, default=6.0)

    # 用于模拟真机“趴着/半蹲/站立”等启动状态。
    parser.add_argument(
        "--initial_pose",
        choices=["home", "crouch", "prone"],
        default="prone",
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

    # menagerie scene_mjx.xml 的原始 actuator 是位置伺服语义；新训练默认使用 50/0.5。
    # 这里默认同样覆盖到 50/0.5，让双终端仿真贴近当前训练时的 actuator 设置。
    parser.add_argument(
        "--servo_kp",
        type=float,
        default=50.0,
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
    """MuJoCo 与 Unitree DDS 的桥接仿真器。"""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.scene_xml = Path(args.scene_xml).expanduser().resolve()
        if not self.scene_xml.exists():
            raise FileNotFoundError(f"Scene XML not found: {self.scene_xml}")

        # 加载训练同源 XML；默认是 mujoco_menagerie/unitree_go2/scene_mjx.xml。
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_xml))
        self.model.opt.timestep = float(args.sim_dt)

        # base body 名称在不同 XML 里可能叫 base 或 base_link。
        self.base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        if self.base_body_id < 0:
            self.base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        if self.base_body_id < 0:
            raise RuntimeError("Could not find base body 'base' or 'base_link'.")

        self.home_key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if self.home_key_id < 0:
            raise RuntimeError("Keyframe 'home' not found in the MuJoCo model.")

        # 记录 12 个策略关节在 qpos/qvel/actuator 数组中的索引，后续读写都用这些索引。
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
            # position_servo 模式下，MuJoCo ctrl 表示“目标关节位置”。
            # 修改 gain/bias 后，近似为 torque = servo_kp * (ctrl - q) - servo_kd * dq。
            self.model.actuator_gainprm[self.actuator_id, 0] = float(args.servo_kp)
            self.model.actuator_biasprm[self.actuator_id, 1] = -float(args.servo_kp)
            self.model.actuator_biasprm[self.actuator_id, 2] = -float(args.servo_kd)

        self.data = mujoco.MjData(self.model)
        self.ctrl_min = self.model.actuator_ctrlrange[self.actuator_id, 0].astype(np.float32)
        self.ctrl_max = self.model.actuator_ctrlrange[self.actuator_id, 1].astype(np.float32)

        # 先读取 XML 的 home，再按参数覆盖初始姿态。home_q 用于 idle 或站立参考。
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key_id)
        self.home_q = self.data.qpos[self.qpos_adr].astype(np.float32).copy()
        self.initial_q = self._resolve_initial_q(args.initial_pose)
        self.data.qpos[self.qpos_adr] = self.initial_q
        if args.initial_pose == "crouch":
            self.data.qpos[2] = 0.20
        elif args.initial_pose == "prone":
            self.data.qpos[2] = 0.13
        mujoco.mj_forward(self.model, self.data)

        # 没收到有效 LowCmd 时保持哪个姿态：initial 表示保持启动姿态，home 表示拉回 home。
        self.idle_q = self.home_q.copy() if args.idle_target == "home" else self.initial_q.copy()

        # DDS topic 与真机一致：仿真器发布 lowstate，订阅 lowcmd。
        self.lowstate_pub = ChannelPublisher("rt/lowstate", LowState_)
        self.lowstate_pub.Init()
        self.lowcmd_sub = ChannelSubscriber("rt/lowcmd", LowCmd_)
        self.lowcmd_sub.Init(self._lowcmd_cb, 10)

        self.lowstate = unitree_go_msg_dds__LowState_()
        self._init_lowstate_template()

        self._cmd_lock = threading.Lock()
        # 初始命令标记为 inactive，直到收到 deploy runner 的有效 LowCmd。
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
        """根据 XML actuator 语义决定控制模式。

        menagerie 的 *_mjx.xml 带 affine bias，适合 position_servo；
        纯 torque motor XML 没有这种 bias，则使用 pd_torque。
        """
        if requested != "auto":
            return requested
        bias = self.model.actuator_biasprm[self.actuator_id]
        if np.any(np.abs(bias[:, 1]) > 1e-6):
            return "position_servo"
        return "pd_torque"

    def _resolve_initial_q(self, initial_pose: str) -> np.ndarray:
        """根据启动姿态名字生成 12 个关节的初始角。"""
        if initial_pose == "home":
            return self.home_q.copy()
        if initial_pose == "crouch":
            return np.asarray([0.0, 1.25, -2.45] * 4, dtype=np.float32)
        if initial_pose == "prone":
            return np.asarray([0.0, 1.45, -2.60] * 4, dtype=np.float32)
        raise ValueError(f"Unsupported initial pose: {initial_pose}")

    def _init_lowstate_template(self) -> None:
        """初始化 LowState 消息的固定字段。"""
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
        """收到 Ctrl+C 或 SIGTERM 时请求退出主循环。"""
        self._shutdown = True

    def _is_active_motor_cmd(self, motor_cmd) -> bool:
        """判断某个 motor_cmd 是否包含有效目标。

        passive 命令一般会把 q/dq 设成 POS_STOP_F/VEL_STOP_F，因此不能拿来驱动仿真。
        """
        if int(motor_cmd.mode) != 0x01:
            return False
        if abs(float(motor_cmd.q) - POS_STOP_F) < 1e3:
            return False
        if abs(float(motor_cmd.dq) - VEL_STOP_F) < 1e-3:
            return False
        return True

    def _lowcmd_cb(self, msg: LowCmd_) -> None:
        """DDS 回调：接收 rt/lowcmd，并转换为本地策略顺序。"""
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
        """线程安全地复制最近一次 LowCmd。"""
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
        """读取本地 MuJoCo 中 12 个策略关节的 q/dq。"""
        q = self.data.qpos[self.qpos_adr].astype(np.float32)
        dq = self.data.qvel[self.qvel_adr].astype(np.float32)
        return q, dq

    def _compute_control(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        """根据最新 LowCmd 计算 MuJoCo actuator ctrl。

        position_servo:
            ctrl 直接等于 LowCmd.q，LowCmd.kp/kd/tau 在默认 menagerie 仿真中不参与力矩计算。
        pd_torque:
            ctrl 被当作 torque，显式使用 LowCmd 的 tau/kp/kd/q/dq 计算。
        """
        cmd = self._get_command()
        cmd_is_fresh = (time.monotonic() - cmd.received_at) <= self.args.cmd_timeout

        if self.control_mode == "position_servo":
            if cmd.active and cmd_is_fresh:
                # menagerie position actuator：ctrl 是目标关节角。
                ctrl = cmd.q
            else:
                ctrl = self.idle_q
        else:
            if cmd.active and cmd_is_fresh:
                # torque actuator：模拟 Unitree 低层 PD + 前馈力矩。
                ctrl = cmd.tau + cmd.kp * (cmd.q - q) + cmd.kd * (cmd.dq - dq)
            else:
                ctrl = self.args.idle_kp * (self.idle_q - q) - self.args.idle_kd * dq

        # ctrlrange 来自 XML，防止目标位置/力矩越界。
        ctrl = np.clip(ctrl, self.ctrl_min, self.ctrl_max)
        return ctrl.astype(np.float32)

    def _publish_lowstate(self) -> None:
        """把当前 MuJoCo 状态打包成 Unitree LowState 并发布。"""
        q_policy, dq_policy = self._read_policy_joint_state()
        actuator_force_policy = self.data.actuator_force[self.actuator_id].astype(np.float32)

        # base 四元数，wxyz 顺序。使用 xquat 和本地 viewer 的观测构造保持一致。
        quat_wxyz = self.data.xquat[self.base_body_id].astype(np.float32)
        body_vel_world = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.base_body_id,
            body_vel_world,
            0,
        )
        # 训练/本地 viewer 的 yaw-rate obs 是“世界角速度旋到机体系”。
        # 不能直接使用 mj_objectVelocity(..., flg_local=1)，它是 MuJoCo 空间速度
        # 的局部表达，和策略 observation 里的 body gyro 不完全等价。
        gyro_body = rotate_world_to_body(body_vel_world[:3], quat_wxyz)
        linvel_body = rotate_world_to_body(body_vel_world[3:], quat_wxyz)

        # 发布给 deploy runner 前，再转回 Unitree 低层关节顺序。
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

        # 这里用线速度/步长给 accelerometer 占位，当前 deploy runner 主要使用 quaternion/gyro。
        self.lowstate.imu_state.accelerometer[0] = float(linvel_body[0] / max(self.args.sim_dt, 1e-6))
        self.lowstate.imu_state.accelerometer[1] = float(linvel_body[1] / max(self.args.sim_dt, 1e-6))
        self.lowstate.imu_state.accelerometer[2] = float(linvel_body[2] / max(self.args.sim_dt, 1e-6))

        self.lowstate.tick = int(self._sim_steps)
        self.lowstate.crc = 0
        self.lowstate_pub.Write(self.lowstate)

    def step(self) -> None:
        """推进一个 MuJoCo 物理步，并发布对应 LowState。"""
        q, dq = self._read_policy_joint_state()
        ctrl_policy = self._compute_control(q, dq)
        self.data.ctrl[self.actuator_id] = ctrl_policy
        mujoco.mj_step(self.model, self.data)
        self._sim_steps += 1
        self._publish_lowstate()

    def run(self) -> None:
        """运行仿真主循环。

        headless 模式只推进物理和 DDS；非 headless 模式额外打开 MuJoCo viewer。
        """
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
            # 无窗口模式适合自动化链路测试。
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
    """初始化 DDS participant 并启动仿真器。"""
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
