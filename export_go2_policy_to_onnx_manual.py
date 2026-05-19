#!/usr/bin/env python3
"""Export GO2 Brax/APG policy checkpoints to deployment-ready ONNX graphs.

This exporter avoids the JAX -> TF -> tf2onnx path.  It reads the APG MLP
parameters directly and writes an ONNX graph that matches deterministic
deployment inference:

  obs -> running-stat normalization -> Dense/ELU/LayerNorm MLP
      -> split distribution logits -> tanh(loc)

By default it writes both raw ``*.onnx`` and ONNX Runtime compatible
``*_ort.onnx`` files, so this is the only export script needed after training.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
from typing import Any, Dict

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.numpy_helper import from_array

from brax.io import model as brax_model


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_POLICY_ROOT = SCRIPT_DIR / "go2_policy_export_local"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "exported_onnx"

ACTION_SIZE = 12
BASELINE_OBS_SIZE = 40
FORWARD_OBS_SIZE = 52


def _collect_existing_names(model: onnx.ModelProto) -> set[str]:
    names: set[str] = set()
    graph = model.graph
    for node in graph.node:
        if node.name:
            names.add(node.name)
        names.update(x for x in node.input if x)
        names.update(x for x in node.output if x)
    for init in graph.initializer:
        names.add(init.name)
    for vi in itertools.chain(graph.input, graph.output, graph.value_info):
        names.add(vi.name)
    return names


def _elem_type_map(model: onnx.ModelProto) -> Dict[str, int]:
    type_map: Dict[str, int] = {}
    graph = model.graph
    for vi in itertools.chain(graph.input, graph.output, graph.value_info):
        t = vi.type.tensor_type
        if t.HasField("elem_type"):
            type_map[vi.name] = t.elem_type
    for init in graph.initializer:
        type_map[init.name] = init.data_type
    return type_map


def _unique_name(base: str, used: set[str]) -> str:
    if base not in used:
        used.add(base)
        return base
    idx = 1
    while True:
        cand = f"{base}_{idx}"
        if cand not in used:
            used.add(cand)
            return cand
        idx += 1


def _scalar_one(elem_type: int) -> np.ndarray:
    if elem_type == TensorProto.DOUBLE:
        return np.array(1.0, dtype=np.float64)
    return np.array(1.0, dtype=np.float32)


def sanitize_model_for_ort(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int, int]:
    """Rewrite uncommon JAX-export ops into standard ONNX Runtime ops.

    The manual exporter currently emits simple ORT-friendly ops already, so the
    replacement counts are usually zero.  Keeping this pass here makes the
    script a single stable entry point even if the graph later gains Expm1 or
    PreventGradient nodes.
    """
    used_names = _collect_existing_names(model)
    elem_types = _elem_type_map(model)
    graph = model.graph

    new_nodes = []
    new_initializers = list(graph.initializer)
    replaced_expm1 = 0
    replaced_pg = 0

    for node in graph.node:
        if node.op_type == "Expm1":
            replaced_expm1 += 1
            in_name = node.input[0]
            out_name = node.output[0]
            elem_type = elem_types.get(in_name, elem_types.get(out_name, TensorProto.FLOAT))

            exp_out = _unique_name(f"{out_name}__exp", used_names)
            one_name = _unique_name(f"{out_name}__one", used_names)
            new_initializers.append(from_array(_scalar_one(elem_type), name=one_name))
            new_nodes.append(
                helper.make_node(
                    "Exp",
                    [in_name],
                    [exp_out],
                    name=_unique_name((node.name or "Expm1") + "_Exp", used_names),
                )
            )
            new_nodes.append(
                helper.make_node(
                    "Sub",
                    [exp_out, one_name],
                    list(node.output),
                    name=_unique_name((node.name or "Expm1") + "_Sub", used_names),
                )
            )
            continue

        if node.op_type == "PreventGradient":
            replaced_pg += 1
            new_nodes.append(
                helper.make_node(
                    "Identity",
                    list(node.input),
                    list(node.output),
                    name=_unique_name((node.name or "PreventGradient") + "_Identity", used_names),
                )
            )
            continue

        new_nodes.append(node)

    del graph.node[:]
    graph.node.extend(new_nodes)
    del graph.initializer[:]
    graph.initializer.extend(new_initializers)

    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception:
        pass
    onnx.checker.check_model(model)
    return model, replaced_expm1, replaced_pg


def _as_float32(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _add_initializer(initializers: list[onnx.TensorProto], name: str, value: np.ndarray) -> str:
    initializers.append(from_array(np.asarray(value), name=name))
    return name


class GraphBuilder:
    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []

    def const(self, name: str, value: np.ndarray) -> str:
        return _add_initializer(self.initializers, name, value)

    def dense(self, x: str, params: dict[str, Any], name: str) -> str:
        kernel = self.const(f"{name}_kernel", _as_float32(params["kernel"]))
        bias = self.const(f"{name}_bias", _as_float32(params["bias"]))
        mm = f"{name}_matmul"
        out = f"{name}_out"
        self.nodes.append(helper.make_node("MatMul", [x, kernel], [mm], name=f"{name}_MatMul"))
        self.nodes.append(helper.make_node("Add", [mm, bias], [out], name=f"{name}_Add"))
        return out

    def elu(self, x: str, name: str) -> str:
        out = f"{name}_out"
        self.nodes.append(helper.make_node("Elu", [x], [out], name=f"{name}_Elu", alpha=1.0))
        return out

    def layer_norm(self, x: str, params: dict[str, Any], name: str, eps: float = 1e-6) -> str:
        scale = self.const(f"{name}_scale", _as_float32(params["scale"]))
        bias = self.const(f"{name}_bias", _as_float32(params["bias"]))
        eps_name = self.const(f"{name}_eps", np.array([eps], dtype=np.float32))

        mean = f"{name}_mean"
        centered = f"{name}_centered"
        square = f"{name}_square"
        var = f"{name}_var"
        var_eps = f"{name}_var_eps"
        std = f"{name}_std"
        norm = f"{name}_norm"
        scaled = f"{name}_scaled"
        out = f"{name}_out"

        self.nodes.append(helper.make_node("ReduceMean", [x], [mean], name=f"{name}_Mean", axes=[0], keepdims=1))
        self.nodes.append(helper.make_node("Sub", [x, mean], [centered], name=f"{name}_Center"))
        self.nodes.append(helper.make_node("Mul", [centered, centered], [square], name=f"{name}_Square"))
        self.nodes.append(helper.make_node("ReduceMean", [square], [var], name=f"{name}_Var", axes=[0], keepdims=1))
        self.nodes.append(helper.make_node("Add", [var, eps_name], [var_eps], name=f"{name}_AddEps"))
        self.nodes.append(helper.make_node("Sqrt", [var_eps], [std], name=f"{name}_Sqrt"))
        self.nodes.append(helper.make_node("Div", [centered, std], [norm], name=f"{name}_Normalize"))
        self.nodes.append(helper.make_node("Mul", [norm, scale], [scaled], name=f"{name}_Scale"))
        self.nodes.append(helper.make_node("Add", [scaled, bias], [out], name=f"{name}_Bias"))
        return out


def export_single_policy(
    checkpoint_path: Path,
    output_path: Path,
    *,
    obs_size: int,
    action_size: int,
    policy_name: str,
    opset: int,
) -> onnx.ModelProto:
    normalizer, policy = brax_model.load_params(str(checkpoint_path))
    params = policy["params"]

    builder = GraphBuilder()

    mean = builder.const("obs_mean", _as_float32(normalizer.mean))
    std = builder.const("obs_std", _as_float32(normalizer.std))
    obs_centered = "obs_centered"
    obs_norm = "obs_norm"
    builder.nodes.append(helper.make_node("Sub", ["obs", mean], [obs_centered], name="ObsCenter"))
    builder.nodes.append(helper.make_node("Div", [obs_centered, std], [obs_norm], name="ObsNormalize"))

    x = obs_norm
    for i in range(2):
        x = builder.dense(x, params[f"hidden_{i}"], f"hidden_{i}")
        x = builder.elu(x, f"hidden_{i}_elu")
        x = builder.layer_norm(x, params[f"LayerNorm_{i}"], f"layernorm_{i}")

    logits = builder.dense(x, params["hidden_2"], "hidden_2")
    starts = builder.const("loc_starts", np.array([0], dtype=np.int64))
    ends = builder.const("loc_ends", np.array([action_size], dtype=np.int64))
    axes = builder.const("loc_axes", np.array([0], dtype=np.int64))
    loc = "loc"
    builder.nodes.append(helper.make_node("Slice", [logits, starts, ends, axes], [loc], name="TakeLoc"))
    builder.nodes.append(helper.make_node("Tanh", [loc], ["action"], name="TanhAction"))

    graph = helper.make_graph(
        builder.nodes,
        f"{policy_name}_deterministic_apg",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [obs_size])],
        [helper.make_tensor_value_info("action", TensorProto.FLOAT, [action_size])],
        initializer=builder.initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="go2_mjx_manual_apg_exporter",
        opset_imports=[helper.make_operatorsetid("", opset)],
    )
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    return model


def write_ort_model(raw_model: onnx.ModelProto, output_path: Path) -> tuple[int, int]:
    ort_model, replaced_expm1, replaced_pg = sanitize_model_for_ort(raw_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(ort_model, str(output_path))
    return replaced_expm1, replaced_pg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export GO2 local Brax/APG checkpoints to raw ONNX and "
            "ONNX Runtime compatible *_ort.onnx without TensorFlow."
        )
    )
    parser.add_argument("--policy_root", type=str, default=str(DEFAULT_POLICY_ROOT))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--raw_only",
        action="store_true",
        help="Only write *.onnx and skip *_ort.onnx. Default writes both.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy_root = Path(args.policy_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    baseline_checkpoint = policy_root / "trotting_2hz_policy"
    forward_checkpoint = policy_root / "forward_locomotion_policy"
    baseline_onnx = output_dir / "trotting_2hz_policy.onnx"
    forward_onnx = output_dir / "forward_locomotion_policy.onnx"
    baseline_ort = output_dir / "trotting_2hz_policy_ort.onnx"
    forward_ort = output_dir / "forward_locomotion_policy_ort.onnx"

    baseline_model = export_single_policy(
        baseline_checkpoint,
        baseline_onnx,
        obs_size=BASELINE_OBS_SIZE,
        action_size=ACTION_SIZE,
        policy_name="trotting_2hz_policy",
        opset=args.opset,
    )
    print(f"Exported baseline ONNX: {baseline_onnx}")
    if not args.raw_only:
        n_expm1, n_pg = write_ort_model(baseline_model, baseline_ort)
        print(f"Exported baseline ORT : {baseline_ort}  (replaced Expm1={n_expm1}, PreventGradient={n_pg})")

    forward_model = export_single_policy(
        forward_checkpoint,
        forward_onnx,
        obs_size=FORWARD_OBS_SIZE,
        action_size=ACTION_SIZE,
        policy_name="forward_locomotion_policy",
        opset=args.opset,
    )
    print(f"Exported forward ONNX : {forward_onnx}")
    if not args.raw_only:
        n_expm1, n_pg = write_ort_model(forward_model, forward_ort)
        print(f"Exported forward ORT  : {forward_ort}  (replaced Expm1={n_expm1}, PreventGradient={n_pg})")

    meta = {
        "format": "onnx",
        "policy_type": "brax_apg_deterministic",
        "exporter": "manual_mlp_no_tensorflow",
        "input_dtype": "float32",
        "opset": args.opset,
        "baseline": {
            "checkpoint": str(baseline_checkpoint),
            "onnx": str(baseline_onnx),
            "ort_onnx": None if args.raw_only else str(baseline_ort),
            "obs_size": BASELINE_OBS_SIZE,
            "action_size": ACTION_SIZE,
            "hidden_layer_sizes": [256, 128],
        },
        "forward": {
            "checkpoint": str(forward_checkpoint),
            "onnx": str(forward_onnx),
            "ort_onnx": None if args.raw_only else str(forward_ort),
            "obs_size": FORWARD_OBS_SIZE,
            "action_size": ACTION_SIZE,
            "hidden_layer_sizes": [128, 64],
        },
    }
    meta_path = output_dir / "export_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Wrote metadata       : {meta_path}")


if __name__ == "__main__":
    main()
