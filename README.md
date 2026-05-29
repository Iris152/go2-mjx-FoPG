# go2-mjx-FoPG

This repository contains the current GO2 MJX/FoPG training, export, local viewing, menagerie sim2sim validation, and Unitree SDK2 deployment runner.

The current main path is intentionally narrow:

- training and validation XML: `mujoco_menagerie/unitree_go2/scene_mjx.xml`
- training servo override: `servo_kp=50.0`, `servo_kd=0.5`
- action scale: `ACTION_SCALE=[0.2, 0.6, 0.6] * 4`
- stage-1 trot training includes an `action_rate` penalty and deploy-style reset distribution from the 2026-05-28 real-robot-tested backup
- stage-2 forward residual reward includes lateral drift, yaw-rate, and action-rate penalties
- ONNX export uses deterministic APG inference and writes both raw ONNX and ONNX Runtime compatible `*_ort.onnx`
- sim2sim validation uses the same menagerie XML instead of the older `unitree_mujoco` XML path

For the self-contained real-robot SDK2 Python bundle, use:

```text
https://github.com/Iris152/go2-mjx-FoPG-deploy
```

## Contents

- `train_go2_mjx_local.py`: local two-stage APG/FoPG training entrypoint.
- `GO2_train_local.ipynb`: local notebook wrapper around the current training script.
- `go2_local_policy_viewer.py`: independent MuJoCo viewer for trot and forward checkpoints. It defaults to deterministic inference to match ONNX deployment.
- `export_go2_policy_to_onnx_manual.py`: single ONNX export entrypoint. It exports both `*.onnx` and `*_ort.onnx`.
- `sanitize_onnx_for_ort.py`: standalone ONNX Runtime compatibility sanitizer.
- `launch_unitree_mujoco_python_sim.py`, `go2_mjx_lowlevel_dds_sim.py`: menagerie XML DDS simulator for double-terminal sim2sim validation.
- `go2_unitree_sdk2_deploy.py`: low-level Unitree SDK2 runner for sim2sim and real GO2.
- `deploy_test.py`: startup-phase diagnostic runner used to isolate stand-up, policy ramp, and gait phase issues.
- `go2_unitree_mujoco_sim2sim.py`: older ROS2/unitree_mujoco-style sim2sim runner kept as a reference path.
- `GO2_DEPLOY_GUIDE.md`: staged sim2sim and real-robot deployment checklist.
- `mujoco_menagerie/unitree_go2/`: GO2 XML and mesh assets required by this project.
- `go2_policy_export_local/`: current local checkpoint pair.
- `exported_onnx/`: exported baseline and forward policies, including ORT-compatible versions.

The older Colab notebook, selectable viewer wrapper, duplicated exporter, and old checkpoint folder have been removed from the current branch to keep the repository focused on the active local workflow.

## Setup

Use an environment with MuJoCo/MJX, Brax, JAX, ONNX Runtime, and Unitree SDK2 Python bindings installed:

```bash
pip install -r requirements.txt
```

Real-robot deployment also needs `unitree_sdk2py`; see `requirements-deploy.txt` and the deploy bundle repository.

## Train Locally

```bash
python train_go2_mjx_local.py
```

By default the script uses:

- XML: `mujoco_menagerie/unitree_go2/scene_mjx.xml`
- output checkpoints: `go2_policy_export_local/`
- run artifacts: `local_training_runs/` (ignored by git)
- stage-1 trot policy: APG MLP `(256, 128)`
- stage-2 forward residual policy: APG MLP `(128, 64)`
- forward target velocity: `0.75`
- `step_k=13`
- `servo_kp=50.0`, `servo_kd=0.5`
- `ACTION_SCALE=[0.2, 0.6, 0.6] * 4`
- stage-1 deploy-style resets: `stage1_deploy_reset_prob=0.35`, `stage1_low_reset_prob=0.15`
- additional forward residual penalties: `lateral_velocity=-1.0`, `yaw_rate=-0.5`, `action_rate=-0.01`

## View Checkpoints

```bash
python go2_local_policy_viewer.py --mode forward
```

For a headless smoke check:

```bash
python go2_local_policy_viewer.py --mode forward --dry_run_steps 20
```

The viewer now builds policy actions independently and does not depend on the old `go2_policy_viewer_selectable.py` wrapper.

## Export ONNX

After training:

```bash
python export_go2_policy_to_onnx_manual.py
```

This writes:

```text
exported_onnx/trotting_2hz_policy.onnx
exported_onnx/trotting_2hz_policy_ort.onnx
exported_onnx/forward_locomotion_policy.onnx
exported_onnx/forward_locomotion_policy_ort.onnx
exported_onnx/export_meta.json
```

The deployment scripts load the `*_ort.onnx` files by default.

## Sim2Sim Validation

Terminal B, menagerie DDS simulator:

```bash
python launch_unitree_mujoco_python_sim.py \
  --backend menagerie \
  --network lo \
  --domain_id 1
```

Default Terminal B behavior:

- XML: `mujoco_menagerie/unitree_go2/scene_mjx.xml`
- initial pose: `prone`
- idle target: `initial`
- control mode: `position_servo`
- position-servo mode writes active LowCmd `kp/kd` into the menagerie affine actuator
- gyro publication: world angular velocity rotated into the base frame before publishing `imu_state.gyroscope`

Terminal A, policy runner:

```bash
python go2_unitree_sdk2_deploy.py \
  --network lo \
  --domain_id 1 \
  --mode forward
```

Default Terminal A behavior is manual:

1. wait for `rt/lowstate`
2. press Enter to stand up
3. press Enter again to start the policy

If using `conda run`, use `--no-capture-output` so the Enter prompts are interactive:

```bash
conda run --no-capture-output -n mjx python go2_unitree_sdk2_deploy.py \
  --network lo \
  --domain_id 1 \
  --mode forward
```

For startup-phase diagnostics:

```bash
python deploy_test.py --network lo --domain_id 1 --mode forward --auto_start --auto_policy
```

The current sim2sim result after the yaw-rate fix is that forward deployment behavior matches `go2_local_policy_viewer.py` much more closely; any remaining small heading bias is now treated primarily as a policy/training issue rather than an ONNX, joint-remap, or stand-up phase issue.

## Real Robot Entry Point

Use `go2_unitree_sdk2_deploy.py` for the current real-robot path:

```bash
python go2_unitree_sdk2_deploy.py \
  --network <robot_nic> \
  --domain_id 0 \
  --mode forward
```

Do not add `--auto_start` or `--auto_policy` on the first real-robot run. With `--domain_id 0`, the runner defaults to `--release_mcf auto`: it queries Go2's MotionSwitcher service and calls `ReleaseMode()` before low-level control if a high-level motion service such as `sport_mode` is active.

Sim2sim keeps `policy_kp=50.0`, `policy_kd=0.5` by default. This branch intentionally matches the 2026-05-28 real-robot-tested backup. On a real robot (`--domain_id 0`), the runner keeps `policy_kd=0.5` unless explicitly overridden and applies these startup-conditioning defaults:

- `policy_ramp_duration=0.5`
- `policy_target_filter_tau=0.04`
- `max_policy_joint_delta=0.08`

If the legs twitch rapidly on hardware, first check the printed timing line with `--print_timing`; the expected policy rate is about 50 Hz, the LowCmd resend rate is about 500 Hz, and the gait cycle is about 0.52 s.

See `GO2_DEPLOY_GUIDE.md` for the full staged checklist.
