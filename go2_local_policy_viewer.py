#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import functools
import importlib.util
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


def _restart_with_mjx_python_if_needed() -> None:
    if importlib.util.find_spec("jax") is not None:
        return

    candidates = []
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(Path(conda_prefix) / "bin" / "python")
    candidates.append(Path.home() / "miniconda3" / "envs" / "mjx" / "bin" / "python")

    current = Path(sys.executable).resolve()
    for candidate in candidates:
        if candidate.exists() and candidate.resolve() != current:
            os.execv(str(candidate), [str(candidate), *sys.argv])


_restart_with_mjx_python_if_needed()

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

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_POLICY_ROOT = SCRIPT_DIR / "go2_policy_export_local"
DEFAULT_RUN_ROOT = SCRIPT_DIR / "local_training_runs"
DEFAULT_XML_PATH = SCRIPT_DIR / "mujoco_menagerie" / "unitree_go2" / "scene_mjx.xml"

KEYFRAME_NAME = "home"
PHYSICS_STEPS_PER_CONTROL = 10
ACTION_SIZE = 12
BASELINE_OBS_SIZE = 40
FORWARD_OBS_SIZE = 52


@dataclass
class PolicyBundle:
    """保存两阶段策略推理函数。"""

    baseline_policy: Callable
    forward_policy: Callable | None = None


def cos_wave(t: jp.ndarray, step_period: float, scale: float) -> jp.ndarray:
    wave = -jp.cos(((2.0 * jp.pi) / step_period) * t)
    return wave * (scale / 2.0) + (scale / 2.0)


def make_kinematic_ref(
    sinusoid: Callable,
    step_k: int,
    scale: float = 0.3,
    dt: float = 1.0 / 50.0,
) -> jp.ndarray:
    """构造和本地训练环境一致的 trot 运动学参考。"""
    steps = jp.arange(step_k)
    step_period = step_k * dt
    t = steps * dt
    wave = sinusoid(t, step_period, scale)

    fleg_cmd_block = jp.concatenate(
        [
            jp.zeros((step_k, 1)),
            wave.reshape(step_k, 1),
            -2.0 * wave.reshape(step_k, 1),
        ],
        axis=1,
    )
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
    deterministic: bool = True,
) -> Callable:
    """按训练网络结构重建 APG 推理函数，并加载 Brax 参数。

    默认使用 deterministic=True，与 ONNX 部署链路保持一致。
    """
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
    return jax.jit(make_inference_fn(params, deterministic=deterministic))


def _checkpoint_root_has_required_files(root: Path, mode: str) -> bool:
    if not (root / "trotting_2hz_policy").exists():
        return False
    if mode == "forward" and not (root / "forward_locomotion_policy").exists():
        return False
    return True


def find_latest_complete_run(mode: str, run_root: Path = DEFAULT_RUN_ROOT) -> Path | None:
    """Find the newest local_training_runs entry usable by the selected mode."""
    if not run_root.exists():
        return None

    candidates: list[tuple[float, Path]] = []
    for checkpoints_dir in run_root.glob("*/checkpoints"):
        if not checkpoints_dir.is_dir():
            continue
        if not _checkpoint_root_has_required_files(checkpoints_dir, mode):
            continue
        run_dir = checkpoints_dir.parent
        latest_mtime = max(
            (p.stat().st_mtime for p in checkpoints_dir.iterdir() if p.is_file()),
            default=checkpoints_dir.stat().st_mtime,
        )
        candidates.append((latest_mtime, run_dir))

    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


class LocalGo2PolicyViewer:
    def __init__(
        self,
        xml_path: str | Path,
        policy_bundle: PolicyBundle,
        mode: str = "forward",
        step_k: int = 13,
        seed: int = 0,
        control_decimation: int = PHYSICS_STEPS_PER_CONTROL,
        track_camera: bool = True,
        clip_final_action: bool = False,
        phase_stride: int = 1,
        deploy_style_start: bool = False,
        deploy_initial_pose: str = "prone",
        stand_ramp_duration: float = 2.5,
        stand_hold_duration: float = 0.5,
        policy_ramp_duration: float = 0.5,
        stand_kp: float = 60.0,
        stand_kd: float = 5.0,
        policy_kp: float = 50.0,
        policy_kd: float = 0.5,
    ):
        if mode not in ("trot", "forward"):
            raise ValueError(f"未知 mode: {mode}，可选值为 trot 或 forward。")
        if mode == "forward" and policy_bundle.forward_policy is None:
            raise ValueError("--mode forward 需要 forward_policy。")
        if deploy_initial_pose not in ("home", "crouch", "prone"):
            raise ValueError("--deploy_initial_pose 只能是 home/crouch/prone。")

        resolved_xml = Path(xml_path).expanduser().resolve()
        if not resolved_xml.exists():
            raise FileNotFoundError(
                f"未找到 XML：{resolved_xml}\n"
                "请确认 mujoco_menagerie/unitree_go2/scene_mjx.xml 路径正确。"
            )

        self.xml_path = str(resolved_xml)
        self.policy_bundle = policy_bundle
        self.mode = mode
        self.step_k = int(step_k)
        self.control_decimation = int(control_decimation)
        self.track_camera = bool(track_camera)
        self.clip_final_action = bool(clip_final_action)
        self.phase_stride = int(phase_stride)
        self.deploy_style_start = bool(deploy_style_start)
        self.deploy_initial_pose = deploy_initial_pose
        self.stand_ramp_duration = float(stand_ramp_duration)
        self.stand_hold_duration = float(stand_hold_duration)
        self.policy_ramp_duration = float(policy_ramp_duration)
        self.stand_kp = float(stand_kp)
        self.stand_kd = float(stand_kd)
        self.policy_kp = float(policy_kp)
        self.policy_kd = float(policy_kd)
        self.rng = jax.random.PRNGKey(seed)

        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        # 默认策略阶段沿用训练/viewer 使用的 position actuator PD。
        self._set_servo_pd(self.policy_kp, self.policy_kd)

        self.control_dt = self.model.opt.timestep * self.control_decimation
        self.action_loc = self.model.keyframe(KEYFRAME_NAME).qpos[7:].copy()
        self.action_scale = np.array([0.2, 0.6, 0.6] * 4, dtype=np.float64)
        self.init_q = self.model.keyframe(KEYFRAME_NAME).qpos.copy()

        self.l_cycle = int(
            make_kinematic_ref(cos_wave, self.step_k, scale=0.3, dt=self.control_dt).shape[0]
        )
        self.kinematic_ref_qpos = np.asarray(
            make_kinematic_ref(cos_wave, self.step_k, scale=0.3, dt=self.control_dt)
            + self.action_loc
        )

        self.last_action = np.zeros(ACTION_SIZE, dtype=np.float64)
        self.inner_obs = jp.zeros(BASELINE_OBS_SIZE, dtype=jp.float64)
        self.outer_obs = jp.zeros(FORWARD_OBS_SIZE, dtype=jp.float64)
        self.control_steps = 0
        self.physics_steps = 0
        self.phase = "policy"
        self.phase_steps = 0
        self.phase_start_q = self.action_loc.copy()

    def _set_servo_pd(self, kp: float, kd: float) -> None:
        self.model.actuator_gainprm[:, 0] = float(kp)
        self.model.actuator_biasprm[:, 1] = -float(kp)
        self.model.actuator_biasprm[:, 2] = -float(kd)

    def _initial_joint_pose(self, pose: str) -> np.ndarray:
        if pose == "home":
            return self.action_loc.copy()
        if pose == "crouch":
            return np.asarray([0.0, 1.25, -2.45] * 4, dtype=np.float64)
        if pose == "prone":
            return np.asarray([0.0, 1.45, -2.60] * 4, dtype=np.float64)
        raise ValueError(f"未知初始姿态: {pose}")

    def reset(self, initial_pose: str = "home") -> None:
        self._set_servo_pd(self.policy_kp, self.policy_kd)
        self.data.qpos[:] = self.init_q
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self.action_loc
        self.data.qpos[7:19] = self._initial_joint_pose(initial_pose)
        if initial_pose == "crouch":
            self.data.qpos[2] = 0.20
        elif initial_pose == "prone":
            self.data.qpos[2] = 0.13
        mujoco.mj_forward(self.model, self.data)

        if initial_pose == "home" and self.data.ncon > 0:
            pen = np.min(self.data.contact.dist[: self.data.ncon])
            self.data.qpos[2] -= pen
            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)

        self.last_action[:] = 0.0
        self.control_steps = 0
        self.physics_steps = 0
        self.inner_obs = self._get_inner_obs()

        if self.mode == "forward":
            baseline_action = self._sample_policy(self.policy_bundle.baseline_policy, self.inner_obs)
            self.outer_obs = jp.concatenate([self.inner_obs, baseline_action])

    def prepare_run(self) -> None:
        if not self.deploy_style_start:
            self.reset()
            self.phase = "policy"
            self.phase_steps = 0
            return

        self.reset(initial_pose=self.deploy_initial_pose)
        self._set_servo_pd(self.stand_kp, self.stand_kd)
        self.phase_steps = 0
        self.phase_start_q = self.data.qpos[7:19].copy()
        self.last_action[:] = self.action_loc
        self.control_steps = 0
        self.inner_obs = self._get_inner_obs()
        if self.mode == "forward":
            baseline_action = self._sample_policy(self.policy_bundle.baseline_policy, self.inner_obs)
            self.outer_obs = jp.concatenate([self.inner_obs, baseline_action])
        if self.stand_ramp_duration > 0.0:
            self.phase = "stand_ramp"
        elif self.stand_hold_duration > 0.0:
            self.phase = "stand_hold"
        else:
            self._start_policy_phase()

    def _start_policy_phase(self) -> None:
        self._set_servo_pd(self.policy_kp, self.policy_kd)
        self.phase = "policy_ramp" if self.policy_ramp_duration > 0.0 else "policy"
        self.phase_steps = 0
        self.control_steps = 0
        self.last_action[:] = self.action_loc
        self.inner_obs = self._get_inner_obs()
        if self.mode == "forward":
            baseline_action = self._sample_policy(self.policy_bundle.baseline_policy, self.inner_obs)
            self.outer_obs = jp.concatenate([self.inner_obs, baseline_action])

    def _split_rng(self) -> jp.ndarray:
        act_rng, self.rng = jax.random.split(self.rng)
        return act_rng

    def _sample_policy(self, policy_fn: Callable, obs: jp.ndarray) -> jp.ndarray:
        action, _ = policy_fn(obs, self._split_rng())
        return action

    def _get_base_quat(self) -> jp.ndarray:
        return jp.array(self.data.xquat[1])

    def _get_base_angvel_world(self) -> jp.ndarray:
        return jp.array(self.data.cvel[1, :3])

    def _get_inner_obs(self) -> jp.ndarray:
        base_quat = self._get_base_quat()
        inv_base_orientation = math.quat_inv(base_quat)
        local_rpyrate = math.rotate(self._get_base_angvel_world(), inv_base_orientation)

        obs_list = [
            jp.array([local_rpyrate[2]]) * 0.25,
            math.rotate(jp.array([0.0, 0.0, -1.0]), inv_base_orientation),
            jp.array(self.data.qpos[7:19] - self.action_loc),
            jp.array(self.last_action),
            jp.array(self.kinematic_ref_qpos[self.control_steps % self.l_cycle]),
        ]
        return jp.clip(jp.concatenate(obs_list), -100.0, 100.0)

    def _baseline_action(self) -> jp.ndarray:
        return self._sample_policy(self.policy_bundle.baseline_policy, self.inner_obs)

    def _forward_action(self) -> jp.ndarray:
        if self.policy_bundle.forward_policy is None:
            raise RuntimeError("forward_policy 为空，无法运行 forward 模式。")

        # Stage2 训练环境裁剪 residual action，但不裁剪 baseline action。
        residual_action = jp.clip(self._sample_policy(self.policy_bundle.forward_policy, self.outer_obs), -1.0, 1.0)
        baseline_action = self.outer_obs[-ACTION_SIZE:]
        action = residual_action + baseline_action
        if self.clip_final_action:
            action = jp.clip(action, -1.0, 1.0)
        return action

    def _compute_policy_joint_target(self) -> np.ndarray:
        if self.mode == "trot":
            action = jp.clip(self._baseline_action(), -1.0, 1.0)
        elif self.mode == "forward":
            action = self._forward_action()
        else:
            raise ValueError(f"未知 mode: {self.mode}")

        return self.action_loc + np.asarray(action, dtype=np.float64) * self.action_scale

    def compute_control(self) -> None:
        if self.deploy_style_start and self.phase == "stand_ramp":
            self._set_servo_pd(self.stand_kp, self.stand_kd)
            alpha = min(((self.phase_steps + 1) * self.control_dt) / self.stand_ramp_duration, 1.0)
            joint_target = (1.0 - alpha) * self.phase_start_q + alpha * self.action_loc
        elif self.deploy_style_start and self.phase == "stand_hold":
            self._set_servo_pd(self.stand_kp, self.stand_kd)
            joint_target = self.action_loc.copy()
        elif self.deploy_style_start and self.phase == "policy_ramp":
            self._set_servo_pd(self.policy_kp, self.policy_kd)
            policy_target = self._compute_policy_joint_target()
            alpha = min(((self.phase_steps + 1) * self.control_dt) / self.policy_ramp_duration, 1.0)
            joint_target = (1.0 - alpha) * self.action_loc + alpha * policy_target
        else:
            self._set_servo_pd(self.policy_kp, self.policy_kd)
            joint_target = self._compute_policy_joint_target()

        self.data.ctrl[:] = joint_target
        self.last_action[:] = joint_target

    def refresh_observation(self) -> None:
        if self.deploy_style_start and self.phase in ("stand_ramp", "stand_hold"):
            self.phase_steps += 1
            if self.phase == "stand_ramp":
                ramp_steps = max(1, int(np.ceil(self.stand_ramp_duration / self.control_dt)))
                if self.phase_steps >= ramp_steps:
                    self.phase = "stand_hold" if self.stand_hold_duration > 0.0 else "policy_ramp"
                    self.phase_steps = 0
                    if self.phase == "policy_ramp":
                        self._start_policy_phase()
            elif self.phase == "stand_hold":
                hold_steps = max(1, int(np.ceil(self.stand_hold_duration / self.control_dt)))
                if self.phase_steps >= hold_steps:
                    self._start_policy_phase()
            return

        # 本地训练环境是在构造下一拍观测前推进 control step。
        self.control_steps += self.phase_stride
        self.inner_obs = self._get_inner_obs()

        if self.mode == "forward":
            baseline_action = self._sample_policy(self.policy_bundle.baseline_policy, self.inner_obs)
            self.outer_obs = jp.concatenate([self.inner_obs, baseline_action])

        if self.deploy_style_start and self.phase == "policy_ramp":
            self.phase_steps += 1
            ramp_steps = max(1, int(np.ceil(self.policy_ramp_duration / self.control_dt)))
            if self.phase_steps >= ramp_steps:
                self.phase = "policy"
                self.phase_steps = 0

    def warmup(self) -> None:
        self.reset()
        self.compute_control()
        self.refresh_observation()
        self.reset()


def resolve_policy_paths(args: argparse.Namespace) -> tuple[Path, Path | None]:
    if args.run_dir:
        root = Path(args.run_dir).expanduser().resolve() / "checkpoints"
        print(f"[policy] using explicit run_dir: {Path(args.run_dir).expanduser().resolve()}")
    elif args.policy_root:
        root = Path(args.policy_root).expanduser().resolve()
        print(f"[policy] using explicit policy_root: {root}")
    else:
        root = DEFAULT_POLICY_ROOT.resolve()
        print(f"[policy] using default policy_root: {root}")

    baseline = (
        Path(args.baseline_policy_path).expanduser().resolve()
        if args.baseline_policy_path
        else root / "trotting_2hz_policy"
    )
    forward = (
        Path(args.forward_policy_path).expanduser().resolve()
        if args.forward_policy_path
        else root / "forward_locomotion_policy"
    )

    if not baseline.exists():
        raise FileNotFoundError(f"未找到第一阶段策略: {baseline}")
    if args.mode == "forward" and not forward.exists():
        raise FileNotFoundError(f"未找到第二阶段策略: {forward}")
    if args.mode == "trot" and not forward.exists():
        forward = None
    return baseline, forward


def build_policy_bundle(
    args: argparse.Namespace,
    baseline_path: Path,
    forward_path: Path | None,
) -> PolicyBundle:
    baseline_policy = build_apg_policy(
        obs_size=BASELINE_OBS_SIZE,
        action_size=ACTION_SIZE,
        hidden_layer_sizes=(256, 128),
        params_path=str(baseline_path),
        deterministic=not args.stochastic,
    )
    forward_policy = None
    if args.mode == "forward":
        if forward_path is None:
            raise RuntimeError("forward mode requires forward policy")
        forward_policy = build_apg_policy(
            obs_size=FORWARD_OBS_SIZE,
            action_size=ACTION_SIZE,
            hidden_layer_sizes=(128, 64),
            params_path=str(forward_path),
            deterministic=not args.stochastic,
        )
    return PolicyBundle(baseline_policy=baseline_policy, forward_policy=forward_policy)


def dry_run(runner: LocalGo2PolicyViewer, steps: int) -> None:
    runner.warmup()
    runner.prepare_run()
    for _ in range(steps):
        runner.compute_control()
        for _ in range(runner.control_decimation):
            mujoco.mj_step(runner.model, runner.data)
        runner.refresh_observation()
    print("dry_run ok")
    print("phase:", runner.phase)
    print("control_steps:", runner.control_steps)
    print("qpos[:7]:", runner.data.qpos[:7])
    print("ctrl:", runner.data.ctrl)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="观察本地 GO2 MJX 两阶段训练结果。默认读取 go2_policy_export_local 中最新确认的策略。"
    )
    parser.add_argument("--mode", choices=("trot", "forward"), default="forward")
    parser.add_argument(
        "--policy_root",
        type=str,
        default=None,
        help="手动指定策略根目录；不填时使用 ./go2_policy_export_local。",
    )
    parser.add_argument("--run_dir", type=str, default=None, help="可直接指向 local_training_runs/... 目录。")
    parser.add_argument("--baseline_policy_path", type=str, default=None)
    parser.add_argument("--forward_policy_path", type=str, default=None)
    parser.add_argument("--xml_path", type=str, default=str(DEFAULT_XML_PATH))
    parser.add_argument("--step_k", type=int, default=13)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--control_decimation", type=int, default=PHYSICS_STEPS_PER_CONTROL)
    parser.add_argument("--phase_stride", type=int, default=1)
    parser.add_argument(
        "--deploy_style_start",
        action="store_true",
        help="先模拟 deploy 的 stand_ramp + stand_hold + policy_ramp，再进入策略。",
    )
    parser.add_argument(
        "--deploy_initial_pose",
        choices=("home", "crouch", "prone"),
        default="prone",
        help="deploy-style start 的初始仿真姿态，默认匹配 menagerie sim 的 prone。",
    )
    parser.add_argument("--stand_ramp_duration", type=float, default=2.5)
    parser.add_argument("--stand_hold_duration", type=float, default=0.5)
    parser.add_argument("--policy_ramp_duration", type=float, default=0.5)
    parser.add_argument("--stand_kp", type=float, default=60.0)
    parser.add_argument("--stand_kd", type=float, default=5.0)
    parser.add_argument("--policy_kp", type=float, default=50.0)
    parser.add_argument("--policy_kd", type=float, default=0.5)
    parser.add_argument(
        "--clip_final_action",
        action="store_true",
        help="额外裁剪 baseline+residual。默认关闭，以匹配本地训练脚本默认行为。",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="使用 APG 随机采样动作。默认关闭，和 ONNX 部署保持 deterministic 一致。",
    )
    parser.add_argument(
        "--force_cpu",
        action="store_true",
        help="只在 CUDA driver 状态异常时使用；默认不强制平台，让 JAX 自动使用 GPU。",
    )
    parser.add_argument("--no_track_camera", action="store_true")
    parser.add_argument("--dry_run_steps", type=int, default=0, help="不打开窗口，只加载策略并跑若干控制步。")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.force_cpu:
        jax.config.update("jax_platforms", "cpu")
    baseline_path, forward_path = resolve_policy_paths(args)

    print("GO2 local policy viewer")
    print("jax devices:", jax.devices())
    print("mode:", args.mode)
    print("xml:", Path(args.xml_path).expanduser().resolve())
    print("baseline:", baseline_path)
    if args.mode == "forward":
        print("forward:", forward_path)
    print("clip_final_action:", args.clip_final_action)
    print("stochastic:", args.stochastic)
    print("phase_stride:", args.phase_stride)
    print("deploy_style_start:", args.deploy_style_start)
    if args.deploy_style_start:
        print(
            "deploy_start:",
            f"initial_pose={args.deploy_initial_pose}",
            f"stand_ramp={args.stand_ramp_duration}",
            f"stand_hold={args.stand_hold_duration}",
            f"policy_ramp={args.policy_ramp_duration}",
            f"stand_pd=({args.stand_kp}, {args.stand_kd})",
            f"policy_pd=({args.policy_kp}, {args.policy_kd})",
        )

    policy_bundle = build_policy_bundle(args, baseline_path, forward_path)
    runner = LocalGo2PolicyViewer(
        xml_path=args.xml_path,
        policy_bundle=policy_bundle,
        mode=args.mode,
        step_k=args.step_k,
        seed=args.seed,
        control_decimation=args.control_decimation,
        track_camera=not args.no_track_camera,
        clip_final_action=args.clip_final_action,
        phase_stride=args.phase_stride,
        deploy_style_start=args.deploy_style_start,
        deploy_initial_pose=args.deploy_initial_pose,
        stand_ramp_duration=args.stand_ramp_duration,
        stand_hold_duration=args.stand_hold_duration,
        policy_ramp_duration=args.policy_ramp_duration,
        stand_kp=args.stand_kp,
        stand_kd=args.stand_kd,
        policy_kp=args.policy_kp,
        policy_kd=args.policy_kd,
    )

    if args.dry_run_steps > 0:
        dry_run(runner, args.dry_run_steps)
        return 0

    runner.warmup()
    runner.prepare_run()
    print("启动 MuJoCo viewer，按 ESC 退出...")
    print(f"控制周期: {runner.control_dt:.4f}s, 物理步长: {runner.model.opt.timestep:.4f}s")

    with mujoco.viewer.launch_passive(runner.model, runner.data) as viewer:
        viewer.cam.distance = 2.5
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -20
        viewer.cam.lookat[:] = runner.data.qpos[:3]

        while viewer.is_running():
            step_start = time.time()
            runner.compute_control()
            for _ in range(runner.control_decimation):
                mujoco.mj_step(runner.model, runner.data)
                runner.physics_steps += 1
            runner.refresh_observation()
            if runner.track_camera:
                viewer.cam.lookat[:] = runner.data.qpos[:3]
            viewer.sync()

            sleep_s = runner.control_dt - (time.time() - step_start)
            if sleep_s > 0:
                time.sleep(sleep_s)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
