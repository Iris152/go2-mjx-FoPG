#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""启动相位隔离测试版部署脚本。

用途：
    用同一套 DDS / ONNX / LowCmd 链路测试“起步异常是否来自 policy ramp 和步态相位”。

与 go2_unitree_sdk2_deploy.py 的主要区别：
    1. 默认把 --policy_ramp_duration 设为 0.0，直接从站立保持进入策略。
    2. 每次进入 POLICY_RAMP / POLICY 时重置 control_steps=0，让步态相位从第 0 拍开始。
    3. 可加 --skip_standup 完全跳过站立阶段，从当前姿态直接进入策略。
    4. 保留原部署脚本的 MCF 释放、安全检查、关节顺序映射和 ONNX 推理。

双终端仿真建议：
    Terminal B:
        conda run -n mjx python launch_unitree_mujoco_python_sim.py --backend menagerie --network lo --domain_id 1

    Terminal A:
        conda run -n mjx python deploy_test.py --network lo --domain_id 1 --mode trot --auto_start --auto_policy
"""

from __future__ import annotations

import signal
import sys

import numpy as np

from go2_unitree_sdk2_deploy import (
    ChannelFactoryInitialize,
    Go2DeployRunner,
    POLICY_TO_UNITREE,
    Phase,
    parse_args,
    release_mcf_if_needed,
)


class PhaseResetDeployRunner(Go2DeployRunner):
    """只改启动相位的测试 runner。"""

    def _maybe_start(self, state) -> None:
        if getattr(self.args, "skip_standup", False):
            # 直接从当前仿真姿态切入策略，用于隔离 stand-up 阶段的影响。
            self.phase_start_joint_policy = state.joint_angles_policy.copy()
            self._transition(Phase.POLICY, "skip stand-up; direct policy start")
            self.control_steps = 0
            self.last_joint_target_policy = state.joint_angles_policy.copy()
            print(
                "[test] direct policy start: last_joint_target=current_joint_state",
                flush=True,
            )
            return
        super()._maybe_start(state)

    def _transition(self, phase: Phase, note: str) -> None:
        super()._transition(phase, note)
        if phase in (Phase.POLICY_RAMP, Phase.POLICY):
            self.control_steps = 0
            if not getattr(self.args, "skip_standup", False):
                self.last_joint_target_policy = self.command_center_policy.copy()
                print(
                    "[test] policy gait phase reset: control_steps=0, "
                    "last_joint_target=command_center",
                    flush=True,
                )
            else:
                print("[test] policy gait phase reset: control_steps=0", flush=True)

    def _policy_ramp_target(self, policy_target: np.ndarray) -> np.ndarray:
        if self.args.policy_ramp_duration <= 0.0:
            return np.asarray(policy_target, dtype=np.float32).copy()
        return super()._policy_ramp_target(policy_target)


def main() -> int:
    skip_standup = "--skip_standup" in sys.argv
    original_argv = sys.argv[:]
    if skip_standup:
        sys.argv = [sys.argv[0], *[item for item in sys.argv[1:] if item != "--skip_standup"]]
    try:
        args = parse_args()
    finally:
        sys.argv = original_argv
    args.skip_standup = skip_standup

    if "--policy_ramp_duration" not in sys.argv:
        args.policy_ramp_duration = 0.0

    print(
        "[test-config] START-PHASE TEST: policy_ramp_duration=%s, "
        "reset_phase_on_policy_start=True, skip_standup=%s"
        % (args.policy_ramp_duration, args.skip_standup),
        flush=True,
    )
    print(
        "[config] network=%s domain_id=%s mode=%s command_center=%s "
        "clip_final_action=%s release_mcf=%s baseline=%s forward=%s"
        % (
            args.network,
            args.domain_id,
            args.mode,
            args.command_center,
            args.clip_final_action,
            args.release_mcf,
            args.baseline_onnx,
            args.forward_onnx,
        ),
        flush=True,
    )
    print("[config] policy order -> unitree order map:", POLICY_TO_UNITREE.tolist(), flush=True)

    ChannelFactoryInitialize(args.domain_id, args.network)
    release_mcf_if_needed(args)
    runner = PhaseResetDeployRunner(args)

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
