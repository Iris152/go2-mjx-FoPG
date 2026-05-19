# GO2 Deploy Guide

This guide matches the current `go2_unitree_sdk2_deploy.py` runner and the latest menagerie-based sim2sim validation path.

## 1. What the runner does

- Uses `unitree_sdk2py` directly for `rt/lowstate` and `rt/lowcmd`
- Defaults the sim2sim path to `mujoco_menagerie/unitree_go2/scene_mjx.xml`
- Keeps the two-stage policy chain:
  - baseline: `40 -> 12`
  - forward residual: `52 -> 12`
- Keeps the policy and Unitree joint order remap:
  - policy order: `FL, FR, RL, RR`
  - Unitree order: `FR, FL, RR, RL`
- Handles MCF release automatically on real robot when `--domain_id 0`
- Uses a manual two-step startup by default: stand up first, then enter policy

## 2. Default parameters

Current defaults that matter for deployment:

- training / sim servo: `servo_kp=50.0`, `servo_kd=0.5`
- policy-phase PD: `policy_kp=50.0`, `policy_kd=0.5`
- stand / fault PD: `60.0/5.0`
- policy ramp duration: `0.0`
- initial pose in Terminal B simulator: `prone`
- idle target in Terminal B simulator: `initial`

The old `230.0/0.5` sim-side servo setting is no longer the mainline default.

## 3. Menagerie DDS sim2sim

Terminal B:

```bash
conda run --no-capture-output -n mjx python launch_unitree_mujoco_python_sim.py \
  --backend menagerie \
  --network lo \
  --domain_id 1
```

Terminal B behavior:

- starts from a prone pose
- publishes `rt/lowstate`
- subscribes to `rt/lowcmd`
- uses `position_servo`
- writes the training servo PD into the actuator gain/bias
- publishes gyro in the body frame after rotating MuJoCo world angular velocity by the base quaternion

Terminal A:

```bash
conda run --no-capture-output -n mjx python go2_unitree_sdk2_deploy.py \
  --network lo \
  --domain_id 1 \
  --mode forward
```

Default manual flow:

1. wait for `rt/lowstate`
2. press Enter once to stand up
3. press Enter again to enter the policy

If you want phase isolation, use `deploy_test.py`:

```bash
conda run --no-capture-output -n mjx python deploy_test.py \
  --network lo \
  --domain_id 1 \
  --mode forward \
  --auto_start \
  --auto_policy
```

## 4. Real robot

On the robot, first make sure the machine is in a safe low-level state and the wired network is configured correctly.

Then run:

```bash
conda run --no-capture-output -n mjx python go2_unitree_sdk2_deploy.py \
  --network <robot_nic> \
  --domain_id 0 \
  --mode forward
```

With `--domain_id 0`, the runner checks the MotionSwitcher service and releases active high-level modes such as `sport_mode` before entering low-level control.

Do not use `--auto_start` or `--auto_policy` on the first real-robot run. Keep the first validation short and low-risk.

## 5. Verification order

1. menagerie DDS stand-up only
2. menagerie DDS trot
3. menagerie DDS forward
4. real robot stand-up only
5. short real-robot trot
6. short real-robot forward

## 6. Notes

The current forward sim2sim mismatch was reduced mainly by aligning the yaw-rate observation frame with training and the local viewer. The remaining small heading bias should be treated as a training-side issue first, not as an ONNX or stand-up-phase issue.
