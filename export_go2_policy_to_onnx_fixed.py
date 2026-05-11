#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

目录约定：

your_project/
├── export_go2_policy_to_onnx_fixed.py
├── go2_policy_export/
│   ├── trotting_2hz_policy
│   └── forward_locomotion_policy
└── exported_onnx/

建议：
- 优先先用默认 float32 导出。
- 如果你的本地环境里 JAX/Brax checkpoint 明显是 float64，也可以试 --input_dtype float64。
"""

from __future__ import annotations

import argparse
import functools
import inspect
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jp
from jax import config
from jax.experimental import jax2tf
import tensorflow as tf
import tf2onnx

from brax.io import model as brax_model
from brax.training.acme import running_statistics
from brax.training.agents.apg import networks as apg_networks

# 与你当前 viewer / notebook 保持一致。
config.update("jax_enable_x64", True)
config.update("jax_default_matmul_precision", "high")

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_POLICY_ROOT = SCRIPT_DIR / "go2_policy_export"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "exported_onnx"

ACTION_SIZE = 12
BASELINE_OBS_SIZE = 40
FORWARD_OBS_SIZE = 52


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

    # deterministic=True 对应 NormalTanhDistribution 的 mode，适合部署。
    policy = make_inference_fn(params, deterministic=True)

    def policy_action(obs: jax.Array) -> jax.Array:
        # deterministic=True 时虽然仍要传 key，但不会真正做随机采样。
        action, _ = policy(obs, jax.random.PRNGKey(0))
        return jp.asarray(action, dtype=param_dtype)

    return jax.jit(policy_action)


def make_jax2tf_function(policy_fn, jax_dtype: jax.typing.DTypeLike):
    """构造更稳定的 JAX->TF 转换函数。"""

    def jax_forward(obs):
        obs = jp.asarray(obs, dtype=jax_dtype)
        action = policy_fn(obs)
        return jp.asarray(action, dtype=jax_dtype)

    # 不同 JAX 版本里 convert 的可选参数略有差异，做兼容处理。
    convert_sig = inspect.signature(jax2tf.convert)
    kwargs = {"with_gradient": False}

    # 这两个开关的目的：尽量让图展开成普通 TF op，减少 tf2onnx 解析函数属性时失败。
    if "enable_xla" in convert_sig.parameters:
        kwargs["enable_xla"] = False
    if "native_serialization" in convert_sig.parameters:
        kwargs["native_serialization"] = False

    return jax2tf.convert(jax_forward, **kwargs)


def export_policy_to_onnx(
    policy_fn,
    obs_size: int,
    output_path: Path,
    *,
    input_dtype: str = "float32",
    opset: int = 17,
):
    """把 JAX policy 导出为 ONNX。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if input_dtype == "float64":
        tf_dtype = tf.float64
        jax_dtype = jp.float64
    else:
        tf_dtype = tf.float32
        jax_dtype = jp.float32

    jax_as_tf = make_jax2tf_function(policy_fn, jax_dtype=jax_dtype)

    @tf.function(
        input_signature=[tf.TensorSpec([obs_size], tf_dtype, name="obs")],
        autograph=False,
    )
    def tf_forward(obs):
        action = jax_as_tf(obs)
        action = tf.cast(action, tf_dtype)
        return tf.identity(action, name="action")

    # 先构图。这样如果 JAX->TF 还有问题，会在转换前更早暴露。
    tf_forward.get_concrete_function()

    tf2onnx.convert.from_function(
        tf_forward,
        input_signature=[tf.TensorSpec([obs_size], tf_dtype, name="obs")],
        opset=opset,
        output_path=str(output_path),
    )


def write_export_metadata(output_dir: Path, payload: dict):
    meta_path = output_dir / "export_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return meta_path


def parse_args():
    parser = argparse.ArgumentParser(description="把 GO2 两阶段 Brax/APG 策略导出为 ONNX（修正版）。")
    parser.add_argument(
        "--policy_root",
        type=str,
        default=str(DEFAULT_POLICY_ROOT),
        help="包含 trotting_2hz_policy 和 forward_locomotion_policy 的目录。",
    )
    parser.add_argument(
        "--baseline_policy_path",
        type=str,
        default=None,
        help="第一阶段 trot 策略目录。若不填，则使用 policy_root/trotting_2hz_policy。",
    )
    parser.add_argument(
        "--forward_policy_path",
        type=str,
        default=None,
        help="第二阶段前进策略目录。若不填，则使用 policy_root/forward_locomotion_policy。",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="ONNX 输出目录。",
    )
    parser.add_argument(
        "--input_dtype",
        type=str,
        choices=["float32", "float64"],
        default="float32",
        help="ONNX 输入/输出精度。建议先用 float32。",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset 版本。默认 17。",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    policy_root = Path(args.policy_root).resolve()
    baseline_path = Path(args.baseline_policy_path).resolve() if args.baseline_policy_path else policy_root / "trotting_2hz_policy"
    forward_path = Path(args.forward_policy_path).resolve() if args.forward_policy_path else policy_root / "forward_locomotion_policy"
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not baseline_path.exists():
        raise FileNotFoundError(f"未找到 baseline 策略目录: {baseline_path}")
    if not forward_path.exists():
        raise FileNotFoundError(f"未找到 forward 策略目录: {forward_path}")

    param_dtype = jp.float64 if args.input_dtype == "float64" else jp.float32

    print("=" * 80)
    print("开始导出 GO2 策略到 ONNX（修正版）")
    print(f"baseline checkpoint : {baseline_path}")
    print(f"forward checkpoint  : {forward_path}")
    print(f"output_dir          : {output_dir}")
    print(f"input_dtype         : {args.input_dtype}")
    print(f"opset               : {args.opset}")
    print("=" * 80)

    # 1) baseline / trot policy
    print("\n[1/2] 重建并导出 trotting_2hz_policy ...")
    baseline_policy = build_deterministic_apg_policy(
        obs_size=BASELINE_OBS_SIZE,
        action_size=ACTION_SIZE,
        hidden_layer_sizes=(256, 128),
        params_path=str(baseline_path),
        param_dtype=param_dtype,
    )
    baseline_onnx = output_dir / "trotting_2hz_policy.onnx"
    export_policy_to_onnx(
        baseline_policy,
        obs_size=BASELINE_OBS_SIZE,
        output_path=baseline_onnx,
        input_dtype=args.input_dtype,
        opset=args.opset,
    )
    print(f"✅ 已导出: {baseline_onnx}")

    # 2) forward locomotion policy
    print("\n[2/2] 重建并导出 forward_locomotion_policy ...")
    forward_policy = build_deterministic_apg_policy(
        obs_size=FORWARD_OBS_SIZE,
        action_size=ACTION_SIZE,
        hidden_layer_sizes=(128, 64),
        params_path=str(forward_path),
        param_dtype=param_dtype,
    )
    forward_onnx = output_dir / "forward_locomotion_policy.onnx"
    export_policy_to_onnx(
        forward_policy,
        obs_size=FORWARD_OBS_SIZE,
        output_path=forward_onnx,
        input_dtype=args.input_dtype,
        opset=args.opset,
    )
    print(f"✅ 已导出: {forward_onnx}")

    meta = {
        "format": "onnx",
        "policy_type": "brax_apg_deterministic",
        "input_dtype": args.input_dtype,
        "opset": args.opset,
        "baseline": {
            "checkpoint": str(baseline_path),
            "onnx": str(baseline_onnx),
            "obs_size": BASELINE_OBS_SIZE,
            "action_size": ACTION_SIZE,
            "hidden_layer_sizes": [256, 128],
        },
        "forward": {
            "checkpoint": str(forward_path),
            "onnx": str(forward_onnx),
            "obs_size": FORWARD_OBS_SIZE,
            "action_size": ACTION_SIZE,
            "hidden_layer_sizes": [128, 64],
        },
    }
    meta_path = write_export_metadata(output_dir, meta)
    print(f"\n📝 已写出元数据: {meta_path}")

    print("\n导出完成。")
    print("下一步先用 onnxruntime 做一次离线推理验证，再接 MuJoCo / unitree_ros2。")


if __name__ == "__main__":
    main()
