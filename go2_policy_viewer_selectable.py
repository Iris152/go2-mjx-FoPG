#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import functools
import os
import time
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jp
import mujoco
import mujoco.viewer
import numpy as np
from brax import math
from brax.io import model as brax_model
from brax.training.acme import running_statistics
from brax.training.agents.apg import networks as apg_networks
from jax import config

config.update("jax_enable_x64", True)
config.update("jax_default_matmul_precision", "high")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_POLICY_ROOT = os.path.join(SCRIPT_DIR, "go2_policy_export")
DEFAULT_XML_PATH = os.path.join(
    SCRIPT_DIR,
    "mujoco_menagerie",
    "unitree_go2",
    "scene_mjx.xml",
)

KEYFRAME_NAME = "home"
PHYSICS_STEPS_PER_CONTROL = 10
ACTION_SIZE = 12
BASELINE_OBS_SIZE = 40
FORWARD_OBS_SIZE = 52


@dataclass
class PolicyBundle:
    """保存策略推理函数。

    baseline_policy:
        第一阶段策略，用于原地 trot。
    forward_policy:
        第二阶段策略，用于在 baseline trot 上叠加残差，实现向前运动。
        只有 --mode forward 时才必须提供。
    """
    baseline_policy: callable
    forward_policy: Optional[callable] = None


def cos_wave(t: jp.ndarray, step_period: float, scale: float) -> jp.ndarray:
    """生成抬腿参考轨迹。"""
    wave = -jp.cos(((2.0 * jp.pi) / step_period) * t)
    return wave * (scale / 2.0) + (scale / 2.0)


def make_kinematic_ref(
    sinusoid,
    step_k: int,
    scale: float = 0.3,
    dt: float = 1.0 / 50.0,
) -> jp.ndarray:
    """构造四足 trot 的关节参考轨迹。"""
    steps = jp.arange(step_k)
    step_period = step_k * dt
    t = steps * dt

    wave = sinusoid(t, step_period, scale)

    # 单条前腿的摆动相参考。
    fleg_cmd_block = jp.concatenate(
        [
            jp.zeros((step_k, 1)),
            wave.reshape(step_k, 1),
            -2.0 * wave.reshape(step_k, 1),
        ],
        axis=1,
    )

    # 后腿关节参考模式与前腿一致。
    h_leg_cmd_block = fleg_cmd_block

    block1 = jp.concatenate(
        [
            jp.zeros((step_k, 3)),
            fleg_cmd_block,
            h_leg_cmd_block,
            jp.zeros((step_k, 3)),
        ],
        axis=1,
    )

    block2 = jp.concatenate(
        [
            fleg_cmd_block,
            jp.zeros((step_k, 3)),
            jp.zeros((step_k, 3)),
            h_leg_cmd_block,
        ],
        axis=1,
    )

    return jp.concatenate([block1, block2], axis=0)


def build_apg_policy(
    obs_size: int,
    action_size: int,
    hidden_layer_sizes: tuple[int, ...],
    params_path: str,
):
    """按训练时相同的网络结构重建推理函数，并加载参数。"""
    make_networks_factory = functools.partial(
        apg_networks.make_apg_networks,
        hidden_layer_sizes=hidden_layer_sizes,
    )
    nets = make_networks_factory(
        observation_size=obs_size,
        action_size=action_size,
        preprocess_observations_fn=running_statistics.normalize,
    )
    make_inference_fn = apg_networks.make_inference_fn(nets)
    params = brax_model.load_params(params_path)
    return jax.jit(make_inference_fn(params))


def resolve_policy_paths(args) -> tuple[str, Optional[str]]:
    """解析策略路径。

    支持两种传参方式：
    1. 直接给 --policy_root
    2. 分别给 --baseline_policy_path / --forward_policy_path

    默认 policy_root 指向脚本同级目录下的 go2_policy_export。
    在 --mode trot 时只要求 baseline/trotting_2hz_policy 存在。
    在 --mode forward 时额外要求 forward_locomotion_policy 存在。
    """
    baseline_path = args.baseline_policy_path
    forward_path = args.forward_policy_path

    if args.policy_root:
        if baseline_path is None:
            baseline_path = os.path.join(args.policy_root, "trotting_2hz_policy")
        if forward_path is None:
            forward_path = os.path.join(args.policy_root, "forward_locomotion_policy")

    if not baseline_path or not os.path.exists(baseline_path):
        raise FileNotFoundError(
            f"未找到 baseline 策略路径：{baseline_path}\n"
            "请确认目录结构为：go2_policy_export/trotting_2hz_policy，"
            "或通过 --baseline_policy_path 指定。"
        )

    if args.mode == "forward":
        if not forward_path or not os.path.exists(forward_path):
            raise FileNotFoundError(
                f"未找到前进策略路径：{forward_path}\n"
                "当前 --mode forward 需要第二阶段策略。请确认目录结构为："
                "go2_policy_export/forward_locomotion_policy，"
                "或通过 --forward_policy_path 指定。"
            )
    else:
        # 原地 trot 模式不需要 forward policy；如果不存在就置空。
        if not forward_path or not os.path.exists(forward_path):
            forward_path = None

    return baseline_path, forward_path


class Go2PolicyViewer:
    """把训练好的 GO2 策略接到 MuJoCo 仿真里做可视化演示。

    --mode trot:
        只使用第一阶段 baseline_policy，显示原地 trot。
    --mode forward:
        使用 baseline_policy + forward_policy，保持原来的残差前进逻辑。
    """

    def __init__(
        self,
        xml_path: str,
        policy_bundle: PolicyBundle,
        mode: str = "forward",
        step_k: int = 13,
        seed: int = 0,
        control_decimation: int = PHYSICS_STEPS_PER_CONTROL,
        track_camera: bool = True,
    ):
        if mode not in ("trot", "forward"):
            raise ValueError(f"未知 mode: {mode}，可选值为 trot 或 forward。")
        if mode == "forward" and policy_bundle.forward_policy is None:
            raise ValueError("--mode forward 需要 forward_policy。")

        self.xml_path = xml_path
        self.policy_bundle = policy_bundle
        self.mode = mode
        self.step_k = step_k
        self.control_decimation = control_decimation
        self.track_camera = track_camera
        self.rng = jax.random.PRNGKey(seed)

        if not os.path.exists(xml_path):
            raise FileNotFoundError(
                f"未找到 XML：{xml_path}\n"
                "请先克隆 mujoco_menagerie，并确认 GO2 的 scene_mjx.xml 路径正确。"
            )

        # 载入原生 MuJoCo 模型。
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # 与训练时环境定义保持一致：手动覆盖 position actuator 的增益。
        kp = 230.0
        self.model.actuator_gainprm[:, 0] = kp
        self.model.actuator_biasprm[:, 1] = -kp

        self.control_dt = self.model.opt.timestep * self.control_decimation
        self.action_loc = self.model.keyframe(KEYFRAME_NAME).qpos[7:].copy()
        self.action_scale = np.array([0.2, 0.8, 0.8] * 4, dtype=np.float64)
        self.init_q = self.model.keyframe(KEYFRAME_NAME).qpos.copy()

        self.l_cycle = int(
            make_kinematic_ref(cos_wave, self.step_k, scale=0.3, dt=self.control_dt).shape[0]
        )
        self.kinematic_ref_qpos = np.asarray(
            make_kinematic_ref(cos_wave, self.step_k, scale=0.3, dt=self.control_dt)
            + self.action_loc
        )

        # 运行时缓存：与训练环境里的 state.info 概念对应。
        self.last_action = np.zeros(ACTION_SIZE, dtype=np.float64)
        self.inner_obs = jp.zeros(BASELINE_OBS_SIZE, dtype=jp.float64)
        self.outer_obs = jp.zeros(FORWARD_OBS_SIZE, dtype=jp.float64)
        self.control_steps = 0
        self.physics_steps = 0

    def reset(self):
        """把机器人重置到 home 姿态，并构造初始观测。"""
        self.data.qpos[:] = self.init_q
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self.action_loc
        mujoco.mj_forward(self.model, self.data)

        # 仿照训练环境 reset：如果初始姿态与地面有轻微穿透，就把 base 高度抬起来。
        if self.data.ncon > 0:
            pen = np.min(self.data.contact.dist[: self.data.ncon])
            self.data.qpos[2] -= pen
            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)

        self.last_action[:] = 0.0
        self.control_steps = 0
        self.physics_steps = 0

        self.inner_obs = self._get_inner_obs()

        if self.mode == "forward":
            baseline_action = self._sample_policy(
                self.policy_bundle.baseline_policy,
                self.inner_obs,
            )
            self.outer_obs = jp.concatenate([self.inner_obs, baseline_action])

    def _split_rng(self):
        act_rng, self.rng = jax.random.split(self.rng)
        return act_rng

    def _sample_policy(self, policy_fn, obs: jp.ndarray) -> jp.ndarray:
        """给策略一个观测，拿到一个 12 维动作。"""
        action, _ = policy_fn(obs, self._split_rng())
        return action

    def _get_base_quat(self) -> jp.ndarray:
        """MuJoCo 的 xquat[1] 对应世界坐标系下 base 的四元数。"""
        return jp.array(self.data.xquat[1])

    def _get_base_angvel_world(self) -> jp.ndarray:
        """MuJoCo 的 cvel[1, :3] 是 base 的世界系角速度。"""
        return jp.array(self.data.cvel[1, :3])

    def _get_inner_obs(self) -> jp.ndarray:
        """复现训练环境里的 baseline 观测构造。"""
        base_quat = self._get_base_quat()
        inv_base_orientation = math.quat_inv(base_quat)
        local_rpyrate = math.rotate(self._get_base_angvel_world(), inv_base_orientation)

        obs_list = []
        # 1) 偏航角速度。
        obs_list.append(jp.array([local_rpyrate[2]]) * 0.25)
        # 2) 重力方向在机体坐标系下的投影。
        obs_list.append(math.rotate(jp.array([0.0, 0.0, -1.0]), inv_base_orientation))
        # 3) 电机关节角，使用相对默认站立位姿的偏移。
        obs_list.append(jp.array(self.data.qpos[7:19] - self.action_loc))
        # 4) 上一步已经执行的实际关节目标。
        obs_list.append(jp.array(self.last_action))
        # 5) 当前步态相位对应的参考运动学关节角。
        kin_ref = jp.array(self.kinematic_ref_qpos[self.control_steps % self.l_cycle])
        obs_list.append(kin_ref)

        obs = jp.clip(jp.concatenate(obs_list), -100.0, 100.0)
        return obs

    def _baseline_action(self) -> jp.ndarray:
        """第一阶段原地 trot 策略输出。"""
        return self._sample_policy(self.policy_bundle.baseline_policy, self.inner_obs)

    def _forward_action(self) -> jp.ndarray:
        """第二阶段残差前进策略输出。"""
        if self.policy_bundle.forward_policy is None:
            raise RuntimeError("forward_policy 为空，无法运行 forward 模式。")

        residual_action = self._sample_policy(
            self.policy_bundle.forward_policy,
            self.outer_obs,
        )

        # 保持原来的残差控制逻辑：
        # final_action = baseline_action + residual_action
        cur_base = self.outer_obs[-ACTION_SIZE:]
        return jp.clip(residual_action + cur_base, -1.0, 1.0)

    def compute_control(self):
        """在控制周期开始时，根据当前模式计算下一拍关节目标。"""
        if self.mode == "trot":
            action = jp.clip(self._baseline_action(), -1.0, 1.0)
        elif self.mode == "forward":
            action = self._forward_action()
        else:
            raise ValueError(f"未知 mode: {self.mode}")

        # 映射回真实关节目标位置。
        joint_target = self.action_loc + np.asarray(action, dtype=np.float64) * self.action_scale

        self.data.ctrl[:] = joint_target
        self.last_action[:] = joint_target

    def refresh_observation(self):
        """在一个控制周期的物理积分完成后，刷新下一拍要用的观测。"""
        self.inner_obs = self._get_inner_obs()

        if self.mode == "forward":
            baseline_action = self._sample_policy(
                self.policy_bundle.baseline_policy,
                self.inner_obs,
            )
            self.outer_obs = jp.concatenate([self.inner_obs, baseline_action])

        self.control_steps += 1

    def warmup(self):
        """JIT 预热，避免打开 viewer 后第一次动作明显卡顿。"""
        self.reset()
        self.compute_control()
        self.refresh_observation()
        self.reset()

    def run(self):
        """打开 MuJoCo 可视化窗口，实时播放 GO2 策略。"""
        self.warmup()

        mode_name = "原地 trot" if self.mode == "trot" else "残差前进"
        print(f"启动 MuJoCo 仿真：{mode_name}，按 ESC 退出...")
        print(f"控制周期: {self.control_dt:.4f}s, 物理步长: {self.model.opt.timestep:.4f}s")
        print(f"当前模式: --mode {self.mode}")

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # 默认视角。
            viewer.cam.distance = 2.5
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -20
            viewer.cam.lookat[:] = self.data.qpos[:3]

            while viewer.is_running():
                step_start = time.time()

                # 每个控制周期开始时，计算一次新的目标动作。
                self.compute_control()

                # 用更小的物理步长积分若干次，和训练时 action repeat / decimation 对齐。
                for _ in range(self.control_decimation):
                    mujoco.mj_step(self.model, self.data)
                    self.physics_steps += 1

                # 一个控制周期结束后，根据新状态构造下一次策略输入。
                self.refresh_observation()

                # 跟随机体 base。trot 模式也可能有轻微漂移，默认仍跟随。
                if self.track_camera:
                    viewer.cam.lookat[:] = self.data.qpos[:3]

                viewer.sync()

                # 尽量把仿真播放速度控制在接近真实时间。
                elapsed = time.time() - step_start
                time_until_next_step = self.control_dt - elapsed
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)


def parse_args():
    parser = argparse.ArgumentParser(
        description="用训练好的 GO2 策略打开 MuJoCo 可视化窗口，可选择原地 trot 或向前运动。"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=("trot", "forward"),
        default="forward",
        help=(
            "选择播放模式："
            "trot=只使用第一阶段 trotting_2hz_policy 原地跑；"
            "forward=使用 baseline + forward_locomotion_policy 残差前进。默认 forward。"
        ),
    )
    parser.add_argument(
        "--policy_root",
        type=str,
        default=DEFAULT_POLICY_ROOT,
        help="导出策略解压后的根目录；默认脚本同级目录下的 go2_policy_export。",
    )
    parser.add_argument(
        "--baseline_policy_path",
        type=str,
        default=None,
        help="第一阶段 trot 策略路径；若不填，则尝试从 policy_root/trotting_2hz_policy 读取。",
    )
    parser.add_argument(
        "--forward_policy_path",
        type=str,
        default=None,
        help="第二阶段前进策略路径；若不填，则尝试从 policy_root/forward_locomotion_policy 读取。",
    )
    parser.add_argument(
        "--xml_path",
        type=str,
        default=DEFAULT_XML_PATH,
        help="GO2 的 MuJoCo XML 路径；默认脚本同级目录下的 mujoco_menagerie/unitree_go2/scene_mjx.xml。",
    )
    parser.add_argument(
        "--step_k",
        type=int,
        default=13,
        help="与训练脚本一致的步态参数 step_k；默认 13。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="策略采样随机种子。",
    )
    parser.add_argument(
        "--control_decimation",
        type=int,
        default=PHYSICS_STEPS_PER_CONTROL,
        help="每次策略输出之间执行多少个 MuJoCo 物理步；默认 10。",
    )
    parser.add_argument(
        "--no_track_camera",
        action="store_true",
        help="关闭相机跟随 base；原地跑时如果想固定视角可以加这个参数。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    baseline_path, forward_path = resolve_policy_paths(args)

    print("正在加载策略...")
    print(f"- mode    : {args.mode}")
    print(f"- baseline: {baseline_path}")
    if args.mode == "forward":
        print(f"- forward : {forward_path}")
    else:
        print("- forward : 未加载；trot 模式不需要")
    print(f"- xml     : {args.xml_path}")

    baseline_policy = build_apg_policy(
        obs_size=BASELINE_OBS_SIZE,
        action_size=ACTION_SIZE,
        hidden_layer_sizes=(256, 128),
        params_path=baseline_path,
    )

    forward_policy = None
    if args.mode == "forward":
        forward_policy = build_apg_policy(
            obs_size=FORWARD_OBS_SIZE,
            action_size=ACTION_SIZE,
            hidden_layer_sizes=(128, 64),
            params_path=forward_path,
        )

    runner = Go2PolicyViewer(
        xml_path=args.xml_path,
        policy_bundle=PolicyBundle(
            baseline_policy=baseline_policy,
            forward_policy=forward_policy,
        ),
        mode=args.mode,
        step_k=args.step_k,
        seed=args.seed,
        control_decimation=args.control_decimation,
        track_camera=not args.no_track_camera,
    )
    runner.run()


if __name__ == "__main__":
    main()
