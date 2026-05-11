# Repository Scope

This cleaned repository keeps the current useful path:

- MJX/FoPG GO2 training on `mujoco_menagerie/unitree_go2/scene_mjx.xml`
- original Colab training notebook, kept as `GO2_train_Colab.ipynb`
- local training notebook/script, kept as `GO2_train_local.ipynb` and `train_go2_mjx_local.py`
- checkpoint viewing
- checkpoint to ONNX export
- ONNX Runtime validation
- local menagerie XML DDS sim2sim validation
- Unitree SDK2 low-level deployment runner

Excluded from the upload:

- `_deps/`: local clones and build products for Unitree MuJoCo and CycloneDDS
- full `mujoco_menagerie/`: only `unitree_go2/` is required here
- `local_training_runs/`, `outputs_study1*`, `result/`: generated artifacts
- `test/`: exploratory notebooks and abandoned official-MJX experiments
- Python caches and editor settings
- the old `GO2_train.ipynb` name: the original Colab workflow now lives in `GO2_train_Colab.ipynb`
- `go2_unitree_mujoco_sim2sim.py`: older ROS2/unitree_mujoco sim2sim path superseded by `go2_unitree_sdk2_deploy.py` plus the menagerie DDS simulator
- `export_go2_official_mjx_policy_to_onnx.py`: depends on the older official-MJX experiment support files and is not part of the current deployment path
