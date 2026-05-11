#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import mujoco.viewer

from go2_policy_viewer_selectable import (
    ACTION_SIZE,
    BASELINE_OBS_SIZE,
    DEFAULT_XML_PATH,
    FORWARD_OBS_SIZE,
    PHYSICS_STEPS_PER_CONTROL,
    PolicyBundle,
    Go2PolicyViewer,
    build_apg_policy,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_POLICY_ROOT = SCRIPT_DIR / "go2_policy_export_local"


class LocalGo2PolicyViewer(Go2PolicyViewer):
    """Viewer variant for policies produced by GO2_train_local.ipynb."""

    def __init__(
        self,
        *args,
        clip_final_action: bool = False,
        phase_stride: int = 1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.clip_final_action = bool(clip_final_action)
        self.phase_stride = int(phase_stride)

    def _forward_action(self) -> jp.ndarray:
        if self.policy_bundle.forward_policy is None:
            raise RuntimeError("forward_policy 为空，无法运行 forward 模式。")

        residual_action = self._sample_policy(self.policy_bundle.forward_policy, self.outer_obs)
        baseline_action = self.outer_obs[-ACTION_SIZE:]
        action = residual_action + baseline_action
        if self.clip_final_action:
            action = jp.clip(action, -1.0, 1.0)
        return action

    def refresh_observation(self):
        # The local training env updates steps before building the next obs.
        self.control_steps += self.phase_stride
        self.inner_obs = self._get_inner_obs()

        if self.mode == "forward":
            baseline_action = self._sample_policy(self.policy_bundle.baseline_policy, self.inner_obs)
            self.outer_obs = jp.concatenate([self.inner_obs, baseline_action])


def resolve_policy_paths(args: argparse.Namespace) -> tuple[Path, Path | None]:
    if args.run_dir:
        root = Path(args.run_dir).expanduser().resolve() / "checkpoints"
    else:
        root = Path(args.policy_root).expanduser().resolve()

    baseline = Path(args.baseline_policy_path).expanduser().resolve() if args.baseline_policy_path else root / "trotting_2hz_policy"
    forward = Path(args.forward_policy_path).expanduser().resolve() if args.forward_policy_path else root / "forward_locomotion_policy"

    if not baseline.exists():
        raise FileNotFoundError(f"未找到第一阶段策略: {baseline}")
    if args.mode == "forward" and not forward.exists():
        raise FileNotFoundError(f"未找到第二阶段策略: {forward}")
    if args.mode == "trot" and not forward.exists():
        forward = None
    return baseline, forward


def build_policy_bundle(args: argparse.Namespace, baseline_path: Path, forward_path: Path | None) -> PolicyBundle:
    baseline_policy = build_apg_policy(
        obs_size=BASELINE_OBS_SIZE,
        action_size=ACTION_SIZE,
        hidden_layer_sizes=(256, 128),
        params_path=str(baseline_path),
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
        )
    return PolicyBundle(baseline_policy=baseline_policy, forward_policy=forward_policy)


def dry_run(runner: LocalGo2PolicyViewer, steps: int) -> None:
    runner.warmup()
    runner.reset()
    for _ in range(steps):
        runner.compute_control()
        for _ in range(runner.control_decimation):
            mujoco.mj_step(runner.model, runner.data)
        runner.refresh_observation()
    print("dry_run ok")
    print("qpos[:7]:", runner.data.qpos[:7])
    print("ctrl:", runner.data.ctrl)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="观察本地 GO2 MJX 两阶段训练结果，默认读取 go2_policy_export_local。"
    )
    parser.add_argument("--mode", choices=("trot", "forward"), default="forward")
    parser.add_argument("--policy_root", type=str, default=str(DEFAULT_POLICY_ROOT))
    parser.add_argument("--run_dir", type=str, default=None, help="可直接指向 local_training_runs/... 目录。")
    parser.add_argument("--baseline_policy_path", type=str, default=None)
    parser.add_argument("--forward_policy_path", type=str, default=None)
    parser.add_argument("--xml_path", type=str, default=DEFAULT_XML_PATH)
    parser.add_argument("--step_k", type=int, default=13)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--control_decimation", type=int, default=PHYSICS_STEPS_PER_CONTROL)
    parser.add_argument("--phase_stride", type=int, default=1)
    parser.add_argument(
        "--clip_final_action",
        action="store_true",
        help="额外裁剪 baseline+residual。默认关闭，以匹配本地训练脚本默认行为。",
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
    print("phase_stride:", args.phase_stride)

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
    )

    if args.dry_run_steps > 0:
        dry_run(runner, args.dry_run_steps)
        return 0

    runner.warmup()
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
