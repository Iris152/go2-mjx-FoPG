#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
对比原始 Brax/APG policy 与导出的 ONNX policy 是否一致。

默认目录约定：

your_project/
├── compare_original_vs_onnx.py
├── go2_policy_export/
│   ├── trotting_2hz_policy
│   └── forward_locomotion_policy
└── exported_onnx/
    ├── trotting_2hz_policy_ort.onnx
    └── forward_locomotion_policy_ort.onnx

常用用法：
    python compare_original_vs_onnx.py

只对比 forward：
    python compare_original_vs_onnx.py --policy_type forward

同时对比 baseline 和 forward：
    python compare_original_vs_onnx.py --policy_type all

增加随机测试样本数：
    python compare_original_vs_onnx.py --num_tests 50 --seed 0

如果你的 ONNX 还没 sanitize，也可以手动指定原始 onnx 路径，
但通常建议传已经修好的 *_ort.onnx。
"""

from __future__ import annotations

import argparse
import functools
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jp
import numpy as np
import onnxruntime as ort
from jax import config

from brax.io import model as brax_model
from brax.training.acme import running_statistics
from brax.training.agents.apg import networks as apg_networks

# 与之前 notebook / viewer / 导出脚本保持一致。
config.update("jax_enable_x64", True)
config.update("jax_default_matmul_precision", "high")

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_POLICY_ROOT = SCRIPT_DIR / "go2_policy_export"
DEFAULT_ONNX_ROOT = SCRIPT_DIR / "exported_onnx"

ACTION_SIZE = 12
BASELINE_OBS_SIZE = 40
FORWARD_OBS_SIZE = 52

POLICY_SPECS = {
    "baseline": {
        "obs_size": BASELINE_OBS_SIZE,
        "hidden_layer_sizes": (256, 128),
        "checkpoint": "trotting_2hz_policy",
        "onnx": "trotting_2hz_policy_ort.onnx",
    },
    "forward": {
        "obs_size": FORWARD_OBS_SIZE,
        "hidden_layer_sizes": (128, 64),
        "checkpoint": "forward_locomotion_policy",
        "onnx": "forward_locomotion_policy_ort.onnx",
    },
}


def tree_cast(tree: Any, dtype: jax.typing.DTypeLike):
    """把 checkpoint 参数树统一转换到指定 dtype。"""

    def _cast(x):
        if hasattr(x, "dtype"):
            return jp.asarray(x, dtype=dtype)
        return x

    return jax.tree_util.tree_map(_cast, tree)



def build_deterministic_apg_policy(
    obs_size: int,
    action_size: int,
    hidden_layer_sizes: tuple[int, ...],
    params_path: str,
    *,
    param_dtype: jax.typing.DTypeLike,
):
    """重建与训练一致的 APG 网络，并返回确定性 policy(obs)->action。"""
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
    params = tree_cast(params, param_dtype)
    policy = make_inference_fn(params, deterministic=True)

    def policy_action(obs: jax.Array) -> jax.Array:
        action, _ = policy(obs, jax.random.PRNGKey(0))
        return jp.asarray(action, dtype=param_dtype)

    return jax.jit(policy_action)



def make_test_observations(obs_size: int, num_tests: int, seed: int) -> list[tuple[str, np.ndarray]]:
    """生成一组测试观测。

    这里同时测：
    - 全零 obs
    - 小范围随机 obs
    - 中等范围随机 obs

    目的不是模拟真实机器人状态，而是验证 ONNX 与原始 policy 的数值一致性。
    """
    rng = np.random.default_rng(seed)
    tests: list[tuple[str, np.ndarray]] = []

    tests.append(("zeros", np.zeros((obs_size,), dtype=np.float32)))

    # 至少保留 1 个零输入测试；其余预算分给两种随机幅度。
    remaining = max(0, num_tests - 1)
    n_small = remaining // 2
    n_medium = remaining - n_small

    for i in range(n_small):
        obs = rng.normal(loc=0.0, scale=0.1, size=(obs_size,)).astype(np.float32)
        tests.append((f"small_rand_{i:03d}", obs))

    for i in range(n_medium):
        obs = rng.normal(loc=0.0, scale=1.0, size=(obs_size,)).astype(np.float32)
        tests.append((f"medium_rand_{i:03d}", obs))

    return tests



def run_single_compare(
    *,
    name: str,
    checkpoint_path: Path,
    onnx_path: Path,
    obs_size: int,
    hidden_layer_sizes: tuple[int, ...],
    num_tests: int,
    seed: int,
    param_dtype: jax.typing.DTypeLike,
    verbose: bool,
) -> bool:
    """执行单个策略的原始 policy vs ONNX 对比。"""
    print("=" * 88)
    print(f"开始对比策略: {name}")
    print(f"checkpoint : {checkpoint_path}")
    print(f"onnx       : {onnx_path}")
    print(f"obs_size   : {obs_size}")
    print(f"num_tests  : {num_tests}")
    print(f"param_dtype: {param_dtype}")
    print("=" * 88)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"未找到 checkpoint 目录: {checkpoint_path}")
    if not onnx_path.exists():
        raise FileNotFoundError(f"未找到 ONNX 文件: {onnx_path}")

    original_policy = build_deterministic_apg_policy(
        obs_size=obs_size,
        action_size=ACTION_SIZE,
        hidden_layer_sizes=hidden_layer_sizes,
        params_path=str(checkpoint_path),
        param_dtype=param_dtype,
    )

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    tests = make_test_observations(obs_size=obs_size, num_tests=num_tests, seed=seed)

    max_abs_err = 0.0
    mean_abs_err_acc = 0.0
    worst_case_name = None
    worst_jax = None
    worst_onnx = None

    for case_name, obs_np in tests:
        # JAX 原始策略输出。为了和 ONNX 对齐，这里统一转成 float32 再比较。
        jax_out = np.asarray(original_policy(jp.asarray(obs_np, dtype=param_dtype)), dtype=np.float32)
        onnx_out = sess.run([output_name], {input_name: obs_np.astype(np.float32)})[0]
        onnx_out = np.asarray(onnx_out, dtype=np.float32)

        abs_err = np.abs(jax_out - onnx_out)
        case_max = float(abs_err.max())
        case_mean = float(abs_err.mean())

        mean_abs_err_acc += case_mean
        if case_max > max_abs_err:
            max_abs_err = case_max
            worst_case_name = case_name
            worst_jax = jax_out
            worst_onnx = onnx_out

        if verbose:
            print(f"[{case_name}] max_abs_err={case_max:.8e}, mean_abs_err={case_mean:.8e}")

    avg_mean_abs_err = mean_abs_err_acc / max(1, len(tests))

    print("\n对比完成。")
    print(f"测试样本数            : {len(tests)}")
    print(f"平均 mean_abs_err    : {avg_mean_abs_err:.8e}")
    print(f"最大 max_abs_err     : {max_abs_err:.8e}")
    print(f"最差样本             : {worst_case_name}")

    if worst_jax is not None and worst_onnx is not None:
        print("\n最差样本下的输出对比：")
        print("JAX :", np.array2string(worst_jax, precision=8, suppress_small=False))
        print("ONNX:", np.array2string(worst_onnx, precision=8, suppress_small=False))
        print(
            "DIFF:",
            np.array2string(np.abs(worst_jax - worst_onnx), precision=8, suppress_small=False),
        )

    # 给一个经验判断阈值。
    passed = max_abs_err < 1e-3
    print("\n结论：", end="")
    if passed:
        print("✅ 一致性看起来是正常的（max_abs_err < 1e-3）。")
    else:
        print("⚠️ 一致性偏差较大，建议继续检查导出链路或 dtype。")

    return passed



def parse_args():
    parser = argparse.ArgumentParser(description="对比原始 Brax/APG policy 与 ONNX policy 的输出一致性。")
    parser.add_argument(
        "--policy_root",
        type=str,
        default=str(DEFAULT_POLICY_ROOT),
        help="包含 trotting_2hz_policy 和 forward_locomotion_policy 的目录。",
    )
    parser.add_argument(
        "--onnx_root",
        type=str,
        default=str(DEFAULT_ONNX_ROOT),
        help="包含 *_ort.onnx 文件的目录。",
    )
    parser.add_argument(
        "--policy_type",
        type=str,
        choices=["baseline", "forward", "all"],
        default="forward",
        help="要对比哪个策略。默认 forward。",
    )
    parser.add_argument(
        "--baseline_checkpoint",
        type=str,
        default=None,
        help="手动指定 baseline checkpoint 目录。",
    )
    parser.add_argument(
        "--forward_checkpoint",
        type=str,
        default=None,
        help="手动指定 forward checkpoint 目录。",
    )
    parser.add_argument(
        "--baseline_onnx",
        type=str,
        default=None,
        help="手动指定 baseline onnx 文件。",
    )
    parser.add_argument(
        "--forward_onnx",
        type=str,
        default=None,
        help="手动指定 forward onnx 文件。",
    )
    parser.add_argument(
        "--num_tests",
        type=int,
        default=11,
        help="测试样本数，默认 11（1 个全零 + 10 个随机）。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="随机种子。",
    )
    parser.add_argument(
        "--param_dtype",
        type=str,
        choices=["float32", "float64"],
        default="float32",
        help="重建原始 policy 时使用的参数 dtype。默认 float32，与当前 ONNX 导出更一致。",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印每个测试样本的误差。",
    )
    return parser.parse_args()



def main():
    args = parse_args()

    policy_root = Path(args.policy_root).resolve()
    onnx_root = Path(args.onnx_root).resolve()
    param_dtype = jp.float64 if args.param_dtype == "float64" else jp.float32

    to_run = ["baseline", "forward"] if args.policy_type == "all" else [args.policy_type]
    overall_ok = True

    for policy_name in to_run:
        spec = POLICY_SPECS[policy_name]
        checkpoint_path = (
            Path(getattr(args, f"{policy_name}_checkpoint")).resolve()
            if getattr(args, f"{policy_name}_checkpoint")
            else (policy_root / spec["checkpoint"]).resolve()
        )
        onnx_path = (
            Path(getattr(args, f"{policy_name}_onnx")).resolve()
            if getattr(args, f"{policy_name}_onnx")
            else (onnx_root / spec["onnx"]).resolve()
        )

        ok = run_single_compare(
            name=policy_name,
            checkpoint_path=checkpoint_path,
            onnx_path=onnx_path,
            obs_size=spec["obs_size"],
            hidden_layer_sizes=spec["hidden_layer_sizes"],
            num_tests=args.num_tests,
            seed=args.seed,
            param_dtype=param_dtype,
            verbose=args.verbose,
        )
        overall_ok = overall_ok and ok
        print()

    print("#" * 88)
    if overall_ok:
        print("总结果：✅ 所有被测策略的一致性都在可接受范围内。")
    else:
        print("总结果：⚠️ 至少有一个策略的一致性偏差较大，建议继续排查。")
    print("#" * 88)


if __name__ == "__main__":
    main()
