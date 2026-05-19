#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用 unitree_mujoco + unitree_ros2 的低层接口，加载已经导出的 GO2 ONNX 策略，
在仿真里做一遍更接近真机接口的 sim2sim 验证。

这份脚本针对你当前已经验证通过的一套固定“前进”策略：
- baseline  : trotting_2hz_policy_ort.onnx
- residual  : forward_locomotion_policy_ort.onnx

它严格复现你 notebook / viewer 里用到的两阶段控制逻辑：
1. 先构造 40 维 baseline 观测
2. baseline 输出 12 维动作
3. 再拼成 52 维 forward 观测
4. residual + baseline -> 最终动作
5. 动作映射为关节目标位置，发布到 /lowcmd

注意：
- 这是“固定向前走”的策略，不读取手柄命令，也不读取 odometry。
- 观测仅依赖 lowstate/imu/motor_state，与当前训练出来的 52 维策略保持一致。
- 更适合先在 unitree_mujoco 里做 sim2sim，再去接真实 GO2。
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Callable, Optional

import mujoco
import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from unitree_go.msg import LowCmd, LowState


ACTION_SIZE = 12
BASELINE_OBS_SIZE = 40
FORWARD_OBS_SIZE = 52
KEYFRAME_NAME = "home"
ACTION_SCALE = np.array([0.2, 0.6, 0.6] * 4, dtype=np.float32)


@dataclass
class PolicySessions:
    baseline: ort.InferenceSession
    forward: ort.InferenceSession


@dataclass
class PolicyMeta:
    action_loc: np.ndarray
    kinematic_ref_qpos: np.ndarray
    l_cycle: int


@dataclass
class RobotState:
    gyro: np.ndarray
    gravity_body: np.ndarray
    joint_angles: np.ndarray
    joint_speeds: np.ndarray


class CRCResolver:
    """尽量复用你本地已经存在的 Unitree CRC 工具。"""

    def __init__(self, allow_zero_crc: bool = False):
        self._get_crc = self._try_import_crc()
        self.allow_zero_crc = allow_zero_crc
        if self._get_crc is None and not self.allow_zero_crc:
            raise RuntimeError(
                "没有找到 get_crc() 实现。\n"
                "请把你已有工程里的 CRC 辅助函数放到当前环境里，例如：\n"
                "- walking_task_go2.utils.motor_crc.get_crc\n"
                "- motor_crc.get_crc\n"
                "或者使用 --allow_zero_crc 先在某些仿真桥里试跑。"
            )

    @staticmethod
    def _try_import_crc() -> Optional[Callable[[LowCmd], int]]:
        candidates = [
            "walking_task_go2.utils.motor_crc",
            "motor_crc",
            "utils.motor_crc",
        ]
        for mod_name in candidates:
            try:
                mod = __import__(mod_name, fromlist=["get_crc"])
                if hasattr(mod, "get_crc"):
                    return getattr(mod, "get_crc")
            except Exception:
                continue
        return None

    def __call__(self, msg: LowCmd) -> int:
        if self._get_crc is None:
            return 0
        return int(self._get_crc(msg))


def cos_wave(t: np.ndarray, step_period: float, scale: float) -> np.ndarray:
    wave = -np.cos(((2.0 * np.pi) / step_period) * t)
    return wave * (scale / 2.0) + (scale / 2.0)


def make_kinematic_ref(step_k: int, dt: float, scale: float = 0.3) -> np.ndarray:
    steps = np.arange(step_k, dtype=np.float32)
    step_period = step_k * dt
    t = steps * dt
    wave = cos_wave(t, step_period, scale)

    fleg_cmd_block = np.concatenate(
        [
            np.zeros((step_k, 1), dtype=np.float32),
            wave.reshape(step_k, 1),
            -2.0 * wave.reshape(step_k, 1),
        ],
        axis=1,
    )

    # 与你当前 GO2 notebook / viewer 保持一致。
    h_leg_cmd_block = fleg_cmd_block.copy()

    block1 = np.concatenate(
        [
            np.zeros((step_k, 3), dtype=np.float32),
            fleg_cmd_block,
            h_leg_cmd_block,
            np.zeros((step_k, 3), dtype=np.float32),
        ],
        axis=1,
    )
    block2 = np.concatenate(
        [
            fleg_cmd_block,
            np.zeros((step_k, 3), dtype=np.float32),
            np.zeros((step_k, 3), dtype=np.float32),
            h_leg_cmd_block,
        ],
        axis=1,
    )
    return np.concatenate([block1, block2], axis=0)


def load_training_meta(train_xml_path: str, control_dt: float, step_k: int) -> PolicyMeta:
    if not os.path.exists(train_xml_path):
        raise FileNotFoundError(f"未找到训练 XML: {train_xml_path}")

    mj_model = mujoco.MjModel.from_xml_path(train_xml_path)
    key = mj_model.keyframe(KEYFRAME_NAME)
    action_loc = np.asarray(key.qpos[7:19], dtype=np.float32)

    kin_ref = make_kinematic_ref(step_k=step_k, dt=control_dt, scale=0.3)
    kinematic_ref_qpos = kin_ref + action_loc[None, :]

    return PolicyMeta(
        action_loc=action_loc,
        kinematic_ref_qpos=kinematic_ref_qpos.astype(np.float32),
        l_cycle=int(kinematic_ref_qpos.shape[0]),
    )


def create_onnx_session(model_path: str) -> ort.InferenceSession:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到 ONNX 模型: {model_path}")
    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


def run_onnx_policy(session: ort.InferenceSession, obs: np.ndarray) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32)
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    input_shape = input_meta.shape

    if len(input_shape) == 1:
        feed = obs
    elif len(input_shape) == 2:
        feed = obs.reshape(1, -1)
    else:
        raise ValueError(f"不支持的 ONNX 输入形状: {input_shape}")

    out = session.run(None, {input_name: feed})[0]
    out = np.asarray(out, dtype=np.float32)
    if out.ndim == 2:
        out = out[0]
    return out


class Go2PolicyLowLevelNode(Node):
    def __init__(self, args):
        super().__init__("go2_policy_lowlevel_sim2sim")

        self.args = args
        self.crc_fn = CRCResolver(allow_zero_crc=args.allow_zero_crc)
        self.sessions = PolicySessions(
            baseline=create_onnx_session(args.baseline_onnx),
            forward=create_onnx_session(args.forward_onnx),
        )
        self.meta = load_training_meta(
            train_xml_path=args.train_xml_path,
            control_dt=args.control_dt,
            step_k=args.step_k,
        )

        self.low_state_sub = self.create_subscription(
            LowState, args.lowstate_topic, self.low_state_callback, 10
        )
        self.low_cmd_pub = self.create_publisher(LowCmd, args.lowcmd_topic, 10)
        self.timer = self.create_timer(args.control_dt, self.control_callback)

        self.have_state = False
        self.robot_state: Optional[RobotState] = None
        self.last_action = np.zeros(ACTION_SIZE, dtype=np.float32)
        self.control_steps = 0

        self.stand_started = False
        self.stand_finished = False
        self.stand_start_joint = None
        self.stand_ramp_steps = max(1, int(args.stand_ramp_duration / args.control_dt))
        self.stand_hold_steps = max(0, int(args.stand_hold_duration / args.control_dt))
        self.stand_counter = 0

        self.low_cmd = LowCmd()
        self.low_cmd.head = [0xFE, 0xEF]
        self.low_cmd.level_flag = 0xFF
        for i in range(12):
            self.low_cmd.motor_cmd[i].mode = 0x01
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].tau = 0.0

        self.get_logger().info("GO2 unitree_mujoco sim2sim 节点已启动。")
        self.get_logger().info(f"lowstate topic: {args.lowstate_topic}")
        self.get_logger().info(f"lowcmd  topic: {args.lowcmd_topic}")
        self.get_logger().info(f"baseline onnx: {args.baseline_onnx}")
        self.get_logger().info(f"forward  onnx: {args.forward_onnx}")
        if args.allow_zero_crc:
            self.get_logger().warn("当前允许 zero CRC，仅建议在部分仿真桥调试时使用。")

    def low_state_callback(self, msg: LowState):
        q = np.array(msg.imu_state.quaternion, dtype=np.float32)
        # Unitree 低层四元数顺序通常是 [w, x, y, z]，SciPy 需要 [x, y, z, w]。
        rot = R.from_quat([q[1], q[2], q[3], q[0]])
        gravity_body = rot.inv().apply(np.array([0.0, 0.0, -1.0], dtype=np.float32)).astype(np.float32)

        gyro = np.array(msg.imu_state.gyroscope, dtype=np.float32)
        joint_angles = np.array([msg.motor_state[i].q for i in range(12)], dtype=np.float32)
        joint_speeds = np.array([msg.motor_state[i].dq for i in range(12)], dtype=np.float32)

        self.robot_state = RobotState(
            gyro=gyro,
            gravity_body=gravity_body,
            joint_angles=joint_angles,
            joint_speeds=joint_speeds,
        )
        self.have_state = True

    def _build_inner_obs(self) -> np.ndarray:
        assert self.robot_state is not None
        kin_ref = self.meta.kinematic_ref_qpos[self.control_steps % self.meta.l_cycle]
        obs = np.concatenate(
            [
                np.array([self.robot_state.gyro[2] * 0.25], dtype=np.float32),
                self.robot_state.gravity_body.astype(np.float32),
                (self.robot_state.joint_angles - self.meta.action_loc).astype(np.float32),
                self.last_action.astype(np.float32),
                kin_ref.astype(np.float32),
            ],
            axis=0,
        )
        if obs.shape[0] != BASELINE_OBS_SIZE:
            raise ValueError(f"baseline 观测维度不对: {obs.shape[0]} != {BASELINE_OBS_SIZE}")
        return np.clip(obs, -100.0, 100.0).astype(np.float32)

    def _compute_policy_target(self) -> np.ndarray:
        inner_obs = self._build_inner_obs()
        baseline_action_raw = np.asarray(run_onnx_policy(self.sessions.baseline, inner_obs), dtype=np.float32)
        outer_obs = np.concatenate([inner_obs, baseline_action_raw], axis=0).astype(np.float32)

        if outer_obs.shape[0] != FORWARD_OBS_SIZE:
            raise ValueError(f"forward 观测维度不对: {outer_obs.shape[0]} != {FORWARD_OBS_SIZE}")

        residual_action = np.asarray(run_onnx_policy(self.sessions.forward, outer_obs), dtype=np.float32)
        residual_action = np.clip(residual_action, -1.0, 1.0)
        final_action = residual_action + baseline_action_raw
        joint_target = self.meta.action_loc + final_action * ACTION_SCALE
        return joint_target.astype(np.float32)

    def _publish_joint_target(self, joint_target: np.ndarray, kp: float, kd: float):
        for i in range(12):
            self.low_cmd.motor_cmd[i].mode = 0x01
            self.low_cmd.motor_cmd[i].q = float(joint_target[i])
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].tau = 0.0
            self.low_cmd.motor_cmd[i].kp = float(kp)
            self.low_cmd.motor_cmd[i].kd = float(kd)

        self.low_cmd.crc = self.crc_fn(self.low_cmd)
        self.low_cmd_pub.publish(self.low_cmd)

    def _run_stand_ramp(self):
        assert self.robot_state is not None
        if not self.stand_started:
            self.stand_started = True
            self.stand_start_joint = self.robot_state.joint_angles.copy()
            self.stand_counter = 0
            self.get_logger().info("开始站立缓启动。")

        self.stand_counter += 1
        if self.stand_counter <= self.stand_ramp_steps:
            alpha = self.stand_counter / self.stand_ramp_steps
            target = (1.0 - alpha) * self.stand_start_joint + alpha * self.meta.action_loc
            self._publish_joint_target(target, self.args.stand_kp, self.args.stand_kd)
            return

        hold_count = self.stand_counter - self.stand_ramp_steps
        self._publish_joint_target(self.meta.action_loc, self.args.stand_kp, self.args.stand_kd)
        if hold_count >= self.stand_hold_steps:
            self.stand_finished = True
            self.last_action = self.meta.action_loc.copy().astype(np.float32)
            self.get_logger().info("站立完成，开始进入策略控制。")

    def control_callback(self):
        if not self.have_state or self.robot_state is None:
            return

        if not self.stand_finished:
            self._run_stand_ramp()
            return

        joint_target = self._compute_policy_target()
        self._publish_joint_target(joint_target, self.args.policy_kp, self.args.policy_kd)
        self.last_action = joint_target.copy().astype(np.float32)
        self.control_steps += 1


def parse_args():
    parser = argparse.ArgumentParser(description="GO2 + unitree_mujoco 的 sim2sim 低层策略节点")
    parser.add_argument(
        "--baseline_onnx",
        type=str,
        default="exported_onnx/trotting_2hz_policy_ort.onnx",
        help="第一阶段 baseline ONNX 路径",
    )
    parser.add_argument(
        "--forward_onnx",
        type=str,
        default="exported_onnx/forward_locomotion_policy_ort.onnx",
        help="第二阶段前进策略 ONNX 路径",
    )
    parser.add_argument(
        "--train_xml_path",
        type=str,
        default="mujoco_menagerie/unitree_go2/scene_mjx.xml",
        help="训练时使用的 GO2 XML，用来读取 home 姿态",
    )
    parser.add_argument("--lowstate_topic", type=str, default="lf/lowstate")
    parser.add_argument("--lowcmd_topic", type=str, default="lowcmd")
    parser.add_argument("--control_dt", type=float, default=0.02)
    parser.add_argument("--step_k", type=int, default=13)
    parser.add_argument("--stand_ramp_duration", type=float, default=2.5)
    parser.add_argument("--stand_hold_duration", type=float, default=0.5)
    parser.add_argument("--stand_kp", type=float, default=60.0)
    parser.add_argument("--stand_kd", type=float, default=5.0)
    parser.add_argument("--policy_kp", type=float, default=50.0)
    parser.add_argument("--policy_kd", type=float, default=0.5)
    parser.add_argument(
        "--allow_zero_crc",
        action="store_true",
        help="找不到 CRC 辅助函数时，允许发送 crc=0（仅某些仿真桥调试可用）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = Go2PolicyLowLevelNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
