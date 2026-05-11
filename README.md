# go2-mjx-FoPG-deploy

This repository contains the cleaned GO2 MJX/FoPG training, export, simulation validation, and deployment code.

The current main path uses `mujoco_menagerie/unitree_go2/scene_mjx.xml` consistently for training-side MJX simulation and double-terminal DDS deployment validation. The older `unitree_mujoco` XML comparison route is kept only as background in the deploy guide; it is not the default path.

## Contents

- `train_go2_mjx_local.py`: local two-stage APG/FoPG training entrypoint for GO2.
- `GO2_train.ipynb`: original notebook record of the successful training workflow.
- `GO2_train_local.ipynb`: local notebook wrapper around the current training script.
- `go2_policy_viewer_selectable.py`, `go2_local_policy_viewer.py`: MuJoCo policy viewers for checkpoints.
- `export_go2_policy_to_onnx_fixed.py`: export Brax/APG checkpoints to deterministic ONNX policies.
- `sanitize_onnx_for_ort.py`: make exported ONNX files easier to run with ONNX Runtime.
- `compare_original_vs_onnx.py`: compare Brax checkpoint inference against ONNX inference.
- `launch_unitree_mujoco_python_sim.py`, `go2_mjx_lowlevel_dds_sim.py`: local menagerie XML DDS simulator for sim2sim validation.
- `go2_unitree_sdk2_deploy.py`: low-level Unitree SDK2 deployment runner for sim2sim and real GO2.
- `GO2_DEPLOY_GUIDE.md`: operational deployment guide.
- `mujoco_menagerie/unitree_go2/`: the GO2 XML and mesh assets required by this project.
- `go2_policy_export/`: checkpoint pair that matches the included ONNX export.
- `go2_policy_export_local/`: latest local training checkpoint pair.
- `exported_onnx/`: exported baseline and forward policies, including ORT-sanitized versions.

## Setup

Use a Python environment with MuJoCo/MJX, Brax, JAX, ONNX Runtime, and Unitree SDK2 Python bindings installed. For a local CPU/GPU training and validation environment:

```bash
pip install -r requirements.txt
```

`unitree_sdk2py` is usually installed from Unitree's SDK2 Python repository or package source, depending on your robot-side environment. It is listed separately in `requirements-deploy.txt` because real-robot deployment often uses a vendor-specific installation path.

## Train

```bash
python train_go2_mjx_local.py
```

By default the script uses:

- XML: `mujoco_menagerie/unitree_go2/scene_mjx.xml`
- output checkpoints: `go2_policy_export_local/`
- run artifacts: `local_training_runs/` (ignored by git)

## View Checkpoints

```bash
python go2_local_policy_viewer.py --mode forward
```

For a headless smoke check:

```bash
python go2_local_policy_viewer.py --mode forward --dry_run_steps 20
```

## Export And Check ONNX

```bash
python export_go2_policy_to_onnx_fixed.py
python sanitize_onnx_for_ort.py exported_onnx/trotting_2hz_policy.onnx exported_onnx/trotting_2hz_policy_ort.onnx
python sanitize_onnx_for_ort.py exported_onnx/forward_locomotion_policy.onnx exported_onnx/forward_locomotion_policy_ort.onnx
python compare_original_vs_onnx.py
```

## Sim2Sim Validation

Terminal B, simulator:

```bash
python launch_unitree_mujoco_python_sim.py --backend menagerie --network lo --domain_id 1
```

Terminal A, policy runner:

```bash
python go2_unitree_sdk2_deploy.py --network lo --domain_id 1 --auto_start --auto_policy
```

The simulator also supports startup-pose checks:

```bash
python launch_unitree_mujoco_python_sim.py \
  --backend menagerie \
  --network lo \
  --domain_id 1 \
  --initial_pose prone \
  --idle_target initial
```

## Real Robot Entry Point

Use `go2_unitree_sdk2_deploy.py` for the current real-robot deployment path. The first real-robot run should be manual, without `--auto_start` or `--auto_policy`:

```bash
python go2_unitree_sdk2_deploy.py --network <robot_nic> --domain_id 0
```

See `GO2_DEPLOY_GUIDE.md` for the full staged checklist.

## Not Included

The source MJX working directory contained local dependencies, generated training runs, videos, caches, and older experimental scripts. Those are intentionally excluded from this cleaned repository. Recreate dependencies from their upstream projects instead of committing `_deps/` or generated outputs.
