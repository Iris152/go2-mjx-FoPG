# Repository Scope

This repository keeps the current useful GO2 MJX/FoPG path:

- local MJX/FoPG training on `mujoco_menagerie/unitree_go2/scene_mjx.xml`
- explicit training servo override: `servo_kp=50.0`, `servo_kd=0.5`
- action scale: `[0.2, 0.6, 0.6] * 4`
- first-stage trot training with action-rate penalty and deploy-style reset distribution from the 2026-05-28 real-robot-tested backup
- second-stage forward residual training with `lateral_velocity`, `yaw_rate`, and action-rate penalties
- deterministic checkpoint viewer
- checkpoint to ONNX and ONNX Runtime export through one exporter script
- menagerie XML DDS sim2sim validation
- Unitree SDK2 low-level deployment runner

Included generated artifacts are intentionally limited to:

- `go2_policy_export_local/`: current local checkpoint pair
- `exported_onnx/`: current raw and ORT-compatible ONNX files
- `mujoco_menagerie/unitree_go2/`: GO2 model assets needed by training, viewing, and deployment validation

Excluded from the upload:

- `_deps/`: local clones, archives, and build products for Unitree MuJoCo and CycloneDDS
- full `mujoco_menagerie/`: only `unitree_go2/` is required here
- `local_training_runs/`, `outputs*`, `result/`, and video files: generated artifacts
- `test/`: exploratory notebooks and abandoned experiments
- Python caches and editor settings
- the original Colab notebook, which is now treated as historical backup outside this current repository
- the old `go2_policy_viewer_selectable.py` wrapper, replaced by the standalone `go2_local_policy_viewer.py`
- duplicated ONNX exporter scripts, replaced by `export_go2_policy_to_onnx_manual.py`
- the older `go2_policy_export/` checkpoint pair, replaced by `go2_policy_export_local/`

The separate deployment bundle repository, `go2-mjx-FoPG-deploy`, vendors the full Unitree SDK2 Python source so real-robot users do not need to clone Unitree's SDK separately.
