#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import functools
import json
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.8")

import jax
import jax.numpy as jp
import mujoco
import mujoco.mjx as mjx
import numpy as np
from brax import envs, math
from brax.base import Motion, Transform
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from brax.io import model as brax_model
from brax.training.acme import running_statistics
from brax.training.agents.apg import networks as apg_networks
from brax.training.agents.apg import train as apg
from jax import config
from ml_collections import config_dict

if not hasattr(jax, "device_put_replicated"):
    def _device_put_replicated(value, devices):
        devices = list(devices)
        if not devices:
            raise ValueError("devices must be non-empty")
        n_devices = len(devices)

        def replicate_leaf(leaf):
            arr = jp.asarray(leaf)
            arr = jp.broadcast_to(arr, (n_devices,) + arr.shape)
            if n_devices == 1:
                return jax.device_put(arr, devices[0])
            mesh = jax.sharding.Mesh(np.asarray(devices), ("replica",))
            sharding = jax.sharding.NamedSharding(
                mesh, jax.sharding.PartitionSpec("replica")
            )
            return jax.device_put(arr, sharding)

        return jax.tree_util.tree_map(replicate_leaf, value)

    jax.device_put_replicated = _device_put_replicated

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_XML_PATH = SCRIPT_DIR / "mujoco_menagerie" / "unitree_go2" / "scene_mjx.xml"
DEFAULT_POLICY_ROOT = SCRIPT_DIR / "go2_policy_export_local"
DEFAULT_RUN_ROOT = SCRIPT_DIR / "local_training_runs"

KEYFRAME_NAME = "home"
FEET_GEOM_NAMES = ("FL", "FR", "RL", "RR")
HIP_BODY_NAMES = ("FL_hip", "FR_hip", "RL_hip", "RR_hip")

ACTION_SIZE = 12
BASELINE_OBS_SIZE = 40
FORWARD_OBS_SIZE = 52
ACTION_SCALE = np.array([0.2, 0.8, 0.8] * 4, dtype=np.float64)

STAGE1_HIDDEN = (256, 128)
STAGE2_HIDDEN = (128, 64)


@dataclass
class ProgressLog:
    x_data: list[float]
    y_data: list[float]
    y_err: list[float]
    rows: list[dict[str, float]]
    progress_fn: Callable[[int, dict[str, Any]], None]


def configure_jax(args: argparse.Namespace) -> None:
    config.update("jax_debug_nans", bool(args.debug_nans))
    config.update("jax_enable_x64", bool(args.use_float64))
    config.update("jax_default_matmul_precision", args.matmul_precision)


def require_xml(xml_path: Path) -> None:
    if xml_path.exists():
        return
    raise FileNotFoundError(
        "未找到旧成功训练使用的 GO2 MJX scene XML:\n"
        f"  {xml_path}\n\n"
        "当前脚本默认只使用 mujoco_menagerie/unitree_go2/scene_mjx.xml，"
        "不会自动切到 Unitree 官方 XML 或 official_aligned_mjx_go2。\n"
        "请把旧的 mujoco_menagerie/unitree_go2 文件补回本地，"
        "或通过 --xml_path 指向恢复后的 scene_mjx.xml。"
    )


def get_named_ids(mj_model: mujoco.MjModel, obj_type: mujoco.mjtObj, names: tuple[str, ...]) -> jp.ndarray:
    ids = []
    missing = []
    for name in names:
        obj_id = mujoco.mj_name2id(mj_model, obj_type, name)
        if obj_id < 0:
            missing.append(name)
        ids.append(obj_id)
    if missing:
        raise ValueError(f"XML 中缺少对象 {missing}; 请确认使用的是 GO2 MJX 训练 XML。")
    return jp.array(np.asarray(ids, dtype=np.int32))


def get_geom_ids(mj_model: mujoco.MjModel, geom_names: tuple[str, ...]) -> jp.ndarray:
    return get_named_ids(mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_names)


def get_body_ids(mj_model: mujoco.MjModel, body_names: tuple[str, ...]) -> jp.ndarray:
    body_ids = get_named_ids(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_names)
    return body_ids - jp.array(1, dtype=jp.int32)


def apply_training_servo_gain(mj_model: mujoco.MjModel, kp: float) -> None:
    # Matches the successful notebook and viewer: the XML keeps the actuator
    # structure, while training overwrites the position-servo proportional gain.
    mj_model.actuator_gainprm[:, 0] = float(kp)
    mj_model.actuator_biasprm[:, 1] = -float(kp)


def cos_wave(t: jax.Array, step_period: float, scale: float) -> jax.Array:
    wave = -jp.cos(((2.0 * jp.pi) / step_period) * t)
    return wave * (scale / 2.0) + (scale / 2.0)


def dcos_wave(t: jax.Array, step_period: float, scale: float) -> jax.Array:
    return ((scale * jp.pi) / step_period) * jp.sin(((2.0 * jp.pi) / step_period) * t)


def make_kinematic_ref(
    sinusoid,
    step_k: int,
    scale: float = 0.3,
    dt: float = 1.0 / 50.0,
) -> jax.Array:
    steps = jp.arange(step_k)
    step_period = step_k * dt
    t = steps * dt
    wave = sinusoid(t, step_period, scale)

    leg_block = jp.concatenate(
        [
            jp.zeros((step_k, 1)),
            wave.reshape(step_k, 1),
            -2.0 * wave.reshape(step_k, 1),
        ],
        axis=1,
    )

    # GO2 success notebook uses the same joint-space pattern for front and hind legs.
    front_leg_block = leg_block
    hind_leg_block = leg_block

    block1 = jp.concatenate(
        [
            jp.zeros((step_k, 3)),
            front_leg_block,
            hind_leg_block,
            jp.zeros((step_k, 3)),
        ],
        axis=1,
    )
    block2 = jp.concatenate(
        [
            front_leg_block,
            jp.zeros((step_k, 3)),
            jp.zeros((step_k, 3)),
            hind_leg_block,
        ],
        axis=1,
    )
    return jp.concatenate([block1, block2], axis=0)


def quaternion_to_matrix(quaternions: jax.Array) -> jax.Array:
    r = quaternions[..., 0]
    i = quaternions[..., 1]
    j = quaternions[..., 2]
    k = quaternions[..., 3]
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    out = jp.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        axis=-1,
    )
    return out.reshape(quaternions.shape[:-1] + (3, 3))


def matrix_to_rotation_6d(matrix: jax.Array) -> jax.Array:
    batch_dim = matrix.shape[:-2]
    return matrix[..., :2, :].reshape(batch_dim + (6,))


def quaternion_to_rotation_6d(quaternion: jax.Array) -> jax.Array:
    return matrix_to_rotation_6d(quaternion_to_matrix(quaternion))


def axis_angle_to_quaternion(axis: jax.Array, angle: jax.Array) -> jax.Array:
    half = 0.5 * angle
    return jp.concatenate([jp.cos(half).reshape(1), jp.sin(half) * axis.reshape(3)])


def get_stage1_config() -> config_dict.ConfigDict:
    return config_dict.ConfigDict(
        dict(
            rewards=config_dict.ConfigDict(
                dict(
                    scales=config_dict.ConfigDict(
                        dict(
                            min_reference_tracking=-2.5 * 3e-3,
                            reference_tracking=-1.0,
                            feet_height=-1.0,
                        )
                    )
                )
            )
        )
    )


def get_stage2_config() -> config_dict.ConfigDict:
    return config_dict.ConfigDict(
        dict(
            rewards=config_dict.ConfigDict(
                dict(
                    scales=config_dict.ConfigDict(
                        dict(
                            tracking_lin_vel=1.0,
                            orientation=-1.0,
                            height=0.5,
                            lin_vel_z=-1.0,
                            torque=-0.01,
                            feet_pos=-1.0,
                            feet_height=-1.0,
                            joint_velocity=-0.001,
                        )
                    )
                )
            )
        )
    )


class Go2MjxBaseEnv(PipelineEnv):
    def __init__(
        self,
        *,
        xml_path: str,
        step_k: int,
        servo_kp: float,
        termination_height: float,
        n_frames: int = 10,
        **kwargs,
    ):
        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)
        self.xml_path = str(Path(xml_path).expanduser().resolve())
        self.step_k = int(step_k)
        self.servo_kp = float(servo_kp)
        self.termination_height = float(termination_height)

        mj_model = mujoco.MjModel.from_xml_path(self.xml_path)
        apply_training_servo_gain(mj_model, self.servo_kp)
        self._mj_model = mj_model

        sys = mjcf.load_model(mj_model)
        super().__init__(sys=sys, **kwargs)

        key = mj_model.keyframe(KEYFRAME_NAME)
        self._init_q = jp.array(key.qpos, dtype=jp.float64)
        self._default_qvel = jp.zeros(mj_model.nv, dtype=jp.float64)
        self._default_ap_pose = jp.array(key.qpos[7:19], dtype=jp.float64)
        self.action_loc = self._default_ap_pose
        self.action_scale = jp.array(ACTION_SCALE, dtype=jp.float64)
        self.feet_inds = get_geom_ids(mj_model, FEET_GEOM_NAMES)

        kinematic_ref_qpos = make_kinematic_ref(cos_wave, self.step_k, scale=0.3, dt=self.dt)
        self.l_cycle = jp.array(int(kinematic_ref_qpos.shape[0]), dtype=jp.int32)
        self.kinematic_ref_action = jp.array(kinematic_ref_qpos + self._default_ap_pose)

        self.pipeline_step = jax.checkpoint(
            self.pipeline_step,
            policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable,
        )

    def _settle_on_ground(self, qpos: jax.Array, qvel: jax.Array):
        data = self.pipeline_init(qpos, qvel)
        pen = jp.min(data.contact.dist)
        qpos = qpos.at[2].set(qpos[2] - pen)
        data = self.pipeline_init(qpos, qvel)
        return qpos, data

    def _compute_done(self, x: Transform) -> jax.Array:
        done = jp.array(0.0)
        done = jp.where(x.pos[0, 2] < self.termination_height, 1.0, done)
        up = jp.array([0.0, 0.0, 1.0])
        done = jp.where(jp.dot(math.rotate(up, x.rot[0]), up) < 0.0, 1.0, done)
        return done

    def mjx_to_brax(self, data):
        q, qd = data.qpos, data.qvel
        x = Transform(pos=data.xpos[1:], rot=data.xquat[1:])
        cvel = Motion(vel=data.cvel[1:, 3:], ang=data.cvel[1:, :3])
        offset = data.xpos[1:, :] - data.subtree_com[self.sys.body_rootid[1:]]
        offset = Transform.create(pos=offset)
        xd = offset.vmap().do(cvel)
        return data.replace(q=q, qd=qd, x=x, xd=xd)

    def _get_inner_obs(
        self,
        qpos: jax.Array,
        x: Transform,
        xd: Motion,
        state_info: dict[str, Any],
    ) -> jax.Array:
        inv_base_orientation = math.quat_inv(x.rot[0])
        local_rpyrate = math.rotate(xd.ang[0], inv_base_orientation)
        ref_idx = jp.asarray(state_info["steps"] % self.l_cycle, dtype=jp.int32)
        kin_ref = self.kinematic_ref_action[ref_idx]

        obs_list = [
            jp.array([local_rpyrate[2]]) * 0.25,
            math.rotate(jp.array([0.0, 0.0, -1.0]), inv_base_orientation),
            qpos[7:19] - self._default_ap_pose,
            state_info["last_action"],
            kin_ref,
        ]
        return jp.clip(jp.concatenate(obs_list), -100.0, 100.0)


class TrotGo2Local(Go2MjxBaseEnv):
    def __init__(
        self,
        *,
        xml_path: str = str(DEFAULT_XML_PATH),
        step_k: int = 13,
        servo_kp: float = 230.0,
        termination_height: float = 0.25,
        **kwargs,
    ):
        super().__init__(
            xml_path=xml_path,
            step_k=step_k,
            servo_kp=servo_kp,
            termination_height=termination_height,
            **kwargs,
        )
        self.err_threshold = 0.4
        self.reward_config = get_stage1_config()

        ref_q_delta = make_kinematic_ref(cos_wave, self.step_k, scale=0.3, dt=self.dt)
        ref_qd_delta = make_kinematic_ref(dcos_wave, self.step_k, scale=0.3, dt=self.dt)
        cycle_len = int(ref_q_delta.shape[0])

        ref_qpos_full = np.tile(np.asarray(self._init_q)[None, :], (cycle_len, 1))
        ref_qvel_full = np.zeros((cycle_len, self._mj_model.nv), dtype=np.float64)
        ref_qpos_full[:, 7:19] = np.asarray(self._default_ap_pose + ref_q_delta)
        ref_qvel_full[:, 6:18] = np.asarray(ref_qd_delta)
        self.kinematic_ref_qpos = jp.array(ref_qpos_full)
        self.kinematic_ref_qvel = jp.array(ref_qvel_full)

    def reset(self, rng: jax.Array) -> State:
        qpos, data = self._settle_on_ground(self._init_q, self._default_qvel)
        state_info = {
            "rng": rng,
            "steps": jp.array(0.0),
            "reward_tuple": {
                "reference_tracking": jp.array(0.0),
                "min_reference_tracking": jp.array(0.0),
                "feet_height": jp.array(0.0),
            },
            "last_action": jp.zeros(ACTION_SIZE),
            "kinematic_ref": jp.zeros(self._mj_model.nq),
        }
        obs = self._get_inner_obs(qpos, data.x, data.xd, state_info)
        reward = jp.array(0.0)
        done = jp.array(0.0)
        metrics = {k: v for k, v in state_info["reward_tuple"].items()}
        return jax.lax.stop_gradient(State(data, obs, reward, done, metrics, state_info))

    def step(self, state: State, action: jax.Array) -> State:
        raw_action = jp.clip(action, -1.0, 1.0)
        desired_q = self.action_loc + raw_action * self.action_scale
        data = self.pipeline_step(state.pipeline_state, desired_q)

        cur_step = state.info["steps"]
        ref_idx = jp.asarray(cur_step % self.l_cycle, dtype=jp.int32)
        ref_qpos = self.kinematic_ref_qpos[ref_idx]
        ref_qvel = self.kinematic_ref_qvel[ref_idx]
        ref_data = data.replace(qpos=ref_qpos, qvel=ref_qvel)
        ref_data = mjx.forward(self.sys, ref_data)

        x, xd = data.x, data.xd
        ref_x, ref_xd = ref_data.x, ref_data.xd
        done = self._compute_done(x)

        reward_tuple = {
            "reference_tracking": (
                self._reward_reference_tracking(x, xd, ref_x, ref_xd)
                * self.reward_config.rewards.scales.reference_tracking
            ),
            "min_reference_tracking": (
                self._reward_min_reference_tracking(ref_qpos, ref_qvel, state)
                * self.reward_config.rewards.scales.min_reference_tracking
            ),
            "feet_height": (
                self._reward_feet_height(
                    data.geom_xpos[self.feet_inds][:, 2],
                    ref_data.geom_xpos[self.feet_inds][:, 2],
                )
                * self.reward_config.rewards.scales.feet_height
            ),
        }
        reward = sum(reward_tuple.values())

        error = (((x.pos - ref_x.pos) ** 2).sum(-1) ** 0.5).mean()
        to_reference = jp.asarray(jp.where(error > self.err_threshold, 1.0, 0.0), dtype=int)
        ref_data = self.mjx_to_brax(ref_data)
        data = jax.tree_util.tree_map(
            lambda a, b: jp.asarray((1 - to_reference) * a + to_reference * b, dtype=a.dtype),
            data,
            ref_data,
        )

        info = dict(state.info)
        info["reward_tuple"] = reward_tuple
        info["last_action"] = desired_q
        info["kinematic_ref"] = ref_qpos

        obs = self._get_inner_obs(data.qpos, data.x, data.xd, info)
        metrics = dict(state.metrics)
        for k, v in reward_tuple.items():
            metrics[k] = v

        return state.replace(
            pipeline_state=data,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
            info=info,
        )

    def _reward_reference_tracking(self, x, xd, ref_x, ref_xd):
        mse = lambda a, b: ((a - b) ** 2).sum(-1).mean()
        return (
            mse(x.pos, ref_x.pos)
            + 0.1 * mse(quaternion_to_rotation_6d(x.rot), quaternion_to_rotation_6d(ref_x.rot))
            + 0.01 * mse(xd.vel, ref_xd.vel)
            + 0.001 * mse(xd.ang, ref_xd.ang)
        )

    def _reward_min_reference_tracking(self, ref_qpos, ref_qvel, state):
        pos = jp.concatenate([state.pipeline_state.qpos[:3], state.pipeline_state.qpos[7:]])
        pos_targ = jp.concatenate([ref_qpos[:3], ref_qpos[7:]])
        pos_err = jp.linalg.norm(pos_targ - pos)
        vel_err = jp.linalg.norm(state.pipeline_state.qvel - ref_qvel)
        return pos_err + vel_err

    def _reward_feet_height(self, feet_pos, feet_pos_ref):
        return jp.sum(jp.abs(feet_pos - feet_pos_ref))


class FwdTrotGo2Local(Go2MjxBaseEnv):
    def __init__(
        self,
        *,
        baseline_inference_fn,
        xml_path: str = str(DEFAULT_XML_PATH),
        target_vel: float = 0.75,
        step_k: int = 13,
        servo_kp: float = 230.0,
        termination_height: float = 0.25,
        clip_final_action: bool = False,
        **kwargs,
    ):
        super().__init__(
            xml_path=xml_path,
            step_k=step_k,
            servo_kp=servo_kp,
            termination_height=termination_height,
            **kwargs,
        )
        self.baseline_inference_fn = baseline_inference_fn
        self.target_vel = float(target_vel)
        self.target_h = self._init_q[2]
        self.gait_period = float(self.step_k * 2) * self.dt
        self.hip_inds = get_body_ids(self._mj_model, HIP_BODY_NAMES)
        self.reward_config = get_stage2_config()
        self.clip_final_action = bool(clip_final_action)

    def reset(self, rng: jax.Array) -> State:
        rng, key_xyz, key_ang, key_ax, key_q, key_qd = jax.random.split(rng, 6)
        qpos = self._init_q
        qvel = self._default_qvel

        r_xyz = 0.2 * (jax.random.uniform(key_xyz, (3,)) - 0.5)
        r_angle = (jp.pi / 12.0) * (jax.random.uniform(key_ang, (1,)) - 0.5)
        r_axis = jax.random.uniform(key_ax, (3,)) - 0.5
        r_axis = r_axis / jp.linalg.norm(r_axis)
        r_quat = axis_angle_to_quaternion(r_axis, r_angle)
        r_joint_q = 0.2 * (jax.random.uniform(key_q, (ACTION_SIZE,)) - 0.5)
        r_joint_qd = 0.1 * (jax.random.uniform(key_qd, (ACTION_SIZE,)) - 0.5)

        qpos = qpos.at[0:3].set(qpos[0:3] + r_xyz)
        qpos = qpos.at[3:7].set(r_quat)
        qpos = qpos.at[7:19].set(qpos[7:19] + r_joint_q)
        qvel = qvel.at[6:18].set(qvel[6:18] + r_joint_qd)

        qpos, data = self._settle_on_ground(qpos, qvel)
        state_info = {
            "rng": rng,
            "steps": jp.array(0.0),
            "reward_tuple": {
                "tracking_lin_vel": jp.array(0.0),
                "orientation": jp.array(0.0),
                "height": jp.array(0.0),
                "lin_vel_z": jp.array(0.0),
                "torque": jp.array(0.0),
                "joint_velocity": jp.array(0.0),
                "feet_pos": jp.array(0.0),
                "feet_height": jp.array(0.0),
            },
            "last_action": jp.zeros(ACTION_SIZE),
            "baseline_action": jp.zeros(ACTION_SIZE),
            "xy0": jp.zeros((4, 2)),
            "k0": jp.array(0.0),
            "xy_star": jp.zeros((4, 2)),
        }

        inner_obs = self._get_inner_obs(data.qpos, data.x, data.xd, state_info)
        action_key, next_rng = jax.random.split(state_info["rng"])
        state_info["rng"] = next_rng
        baseline_action, _ = self.baseline_inference_fn(inner_obs, action_key)
        obs = jp.concatenate([inner_obs, baseline_action])

        reward = jp.array(0.0)
        done = jp.array(0.0)
        metrics = {k: v for k, v in state_info["reward_tuple"].items()}
        return jax.lax.stop_gradient(State(data, obs, reward, done, metrics, state_info))

    def step(self, state: State, action: jax.Array) -> State:
        residual_action = jp.clip(action, -1.0, 1.0)
        baseline_action = state.obs[-ACTION_SIZE:]
        full_action = residual_action + baseline_action
        full_action = jp.where(self.clip_final_action, jp.clip(full_action, -1.0, 1.0), full_action)
        desired_q = self.action_loc + full_action * self.action_scale
        data = self.pipeline_step(state.pipeline_state, desired_q)

        x, xd = data.x, data.xd
        inner_obs = self._get_inner_obs(data.qpos, x, xd, state.info)
        done = self._compute_done(x)

        s = state.info["steps"]
        step_num = s // self.step_k
        even_step = (step_num % 2) == 0
        new_step = (s % self.step_k) == 0
        new_even_step = jp.logical_and(new_step, even_step)
        new_odd_step = jp.logical_and(new_step, jp.logical_not(even_step))

        hip_xy = x.pos[self.hip_inds][:, :2]
        v_body = data.qvel[0:2]
        step_period = self.gait_period / 2.0
        raibert_xy = hip_xy + (step_period / 2.0) * v_body

        cur_targets = state.info["xy_star"]
        idx_frrl = jp.array([1, 2], dtype=jp.int32)
        idx_flrr = jp.array([0, 3], dtype=jp.int32)
        feet_xy = data.geom_xpos[self.feet_inds][:, :2]
        case1 = raibert_xy.at[idx_flrr].set(feet_xy[idx_flrr])
        case2 = raibert_xy.at[idx_frrl].set(feet_xy[idx_frrl])
        xy_targets = jp.where(new_even_step, case1, cur_targets)
        xy_targets = jp.where(new_odd_step, case2, xy_targets)

        info_for_rewards = dict(state.info)
        info_for_rewards["xy_star"] = xy_targets
        info_for_rewards["k0"] = jp.where(new_step, s, state.info["k0"])
        info_for_rewards["xy0"] = jp.where(new_step, feet_xy, state.info["xy0"])
        reward_tuple = {
            "tracking_lin_vel": (
                self._reward_tracking_lin_vel(jp.array([self.target_vel, 0.0, 0.0]), x, xd)
                * self.reward_config.rewards.scales.tracking_lin_vel
            ),
            "orientation": self._reward_orientation(x) * self.reward_config.rewards.scales.orientation,
            "lin_vel_z": self._reward_lin_vel_z(xd) * self.reward_config.rewards.scales.lin_vel_z,
            "height": self._reward_height(data.qpos) * self.reward_config.rewards.scales.height,
            "torque": self._reward_action(data.qfrc_actuator) * self.reward_config.rewards.scales.torque,
            "joint_velocity": (
                self._reward_joint_velocity(data.qvel) * self.reward_config.rewards.scales.joint_velocity
            ),
            "feet_pos": self._reward_feet_pos(data, info_for_rewards) * self.reward_config.rewards.scales.feet_pos,
            "feet_height": (
                self._reward_feet_height(data, info_for_rewards) * self.reward_config.rewards.scales.feet_height
            ),
        }
        reward = sum(reward_tuple.values())

        info = dict(state.info)
        info["reward_tuple"] = reward_tuple
        info["last_action"] = desired_q
        info["baseline_action"] = baseline_action
        info["xy_star"] = xy_targets
        info["k0"] = info_for_rewards["k0"]
        info["xy0"] = info_for_rewards["xy0"]

        action_key, next_rng = jax.random.split(info["rng"])
        info["rng"] = next_rng
        next_action, _ = self.baseline_inference_fn(inner_obs, action_key)
        obs = jp.concatenate([inner_obs, next_action])

        metrics = dict(state.metrics)
        for k, v in reward_tuple.items():
            metrics[k] = v

        return state.replace(
            pipeline_state=data,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
            info=info,
        )

    def _reward_tracking_lin_vel(self, commands: jax.Array, x: Transform, xd: Motion) -> jax.Array:
        local_vel = math.rotate(xd.vel[0], math.quat_inv(x.rot[0]))
        lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
        return jp.exp(-lin_vel_error)

    def _reward_orientation(self, x: Transform) -> jax.Array:
        up = jp.array([0.0, 0.0, 1.0])
        rot_up = math.rotate(up, x.rot[0])
        return jp.sum(jp.square(rot_up[:2]))

    def _reward_lin_vel_z(self, xd: Motion) -> jax.Array:
        return jp.clip(jp.square(xd.vel[0, 2]), 0.0, 10.0)

    def _reward_joint_velocity(self, qvel: jax.Array) -> jax.Array:
        return jp.clip(jp.sqrt(jp.sum(jp.square(qvel[6:]))), 0.0, 100.0)

    def _reward_height(self, qpos: jax.Array) -> jax.Array:
        return jp.exp(-jp.abs(qpos[2] - self.target_h))

    def _reward_action(self, actuator_force: jax.Array) -> jax.Array:
        return jp.sqrt(jp.sum(jp.square(actuator_force)))

    def _reward_feet_pos(self, data, state_info: dict[str, Any]) -> jax.Array:
        dt = (state_info["steps"] - state_info["k0"]) * self.dt
        step_period = self.gait_period / 2.0
        xyt = state_info["xy0"] + (state_info["xy_star"] - state_info["xy0"]) * (dt / step_period)
        feet_pos = data.geom_xpos[self.feet_inds][:, :2]
        errs = jp.sum(jp.square(feet_pos - xyt), axis=1)
        return jp.sum(jp.clip(errs, 0.0, 10.0))

    def _reward_feet_height(self, data, state_info: dict[str, Any]) -> jax.Array:
        h_tar = 0.1
        t = state_info["steps"] * self.dt
        offset = self.gait_period / 2.0
        ref1 = jp.sin((2.0 * jp.pi / self.gait_period) * t)
        ref2 = jp.sin((2.0 * jp.pi / self.gait_period) * (t - offset))
        ref1, ref2 = ref1 * h_tar, ref2 * h_tar
        h_targets = jp.array([ref2, ref1, ref1, ref2])
        h_targets = h_targets.clip(min=0.0) + 0.02
        feet_height = data.geom_xpos[self.feet_inds][:, 2]
        errs = jp.clip(jp.square(feet_height - h_targets), 0.0, 10.0)
        return jp.sum(errs)


def register_envs() -> None:
    try:
        envs.register_environment("go2_mjx_local_trot", TrotGo2Local)
    except Exception:
        pass
    try:
        envs.register_environment("go2_mjx_local_forward", FwdTrotGo2Local)
    except Exception:
        pass


def make_network_factory(hidden_layer_sizes: tuple[int, ...]):
    return functools.partial(apg_networks.make_apg_networks, hidden_layer_sizes=hidden_layer_sizes)


def make_inference_from_checkpoint(
    *,
    checkpoint_path: Path,
    obs_size: int,
    action_size: int,
    hidden_layer_sizes: tuple[int, ...],
):
    nets = make_network_factory(hidden_layer_sizes)(
        observation_size=obs_size,
        action_size=action_size,
        preprocess_observations_fn=running_statistics.normalize,
    )
    make_inference_fn = apg_networks.make_inference_fn(nets)
    params = brax_model.load_params(str(checkpoint_path))
    return jax.jit(make_inference_fn(params)), params


def make_progress_logger(total_updates: int, num_evals: int, label: str) -> ProgressLog:
    x_data: list[float] = []
    y_data: list[float] = []
    y_err: list[float] = []
    rows: list[dict[str, float]] = []
    updates_per_eval = max(1, int(round(total_updates / max(num_evals - 1, 1))))
    started = time.time()

    def progress(it: int, metrics: dict[str, Any]) -> None:
        completed_updates = min(int(it) * updates_per_eval, total_updates)
        reward_mean = float(np.asarray(metrics["eval/episode_reward"]))
        reward_std = float(np.asarray(metrics["eval/episode_reward_std"]))
        elapsed_s = time.time() - started

        x_data.append(completed_updates)
        y_data.append(reward_mean)
        y_err.append(reward_std)
        rows.append(
            {
                "update": float(completed_updates),
                "eval_episode_reward": reward_mean,
                "eval_episode_reward_std": reward_std,
                "elapsed_s": elapsed_s,
            }
        )
        print(
            f"[{label}] update {completed_updates:4d}/{total_updates} | "
            f"eval_reward={reward_mean: .6f} | std={reward_std: .6f} | "
            f"elapsed={elapsed_s: .1f}s",
            flush=True,
        )

    return ProgressLog(x_data=x_data, y_data=y_data, y_err=y_err, rows=rows, progress_fn=progress)


def save_progress_csv(rows: list[dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["update", "eval_episode_reward", "eval_episode_reward_std", "elapsed_s"])
        writer.writeheader()
        writer.writerows(rows)


def save_training_curve(progress: ProgressLog, output_path: Path, title: str) -> None:
    if not progress.x_data:
        return
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.errorbar(progress.x_data, progress.y_data, yerr=progress.y_err, capsize=3)
    plt.xlabel("Policy updates")
    plt.ylabel("Eval episode reward")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def render_rollout_to_video(
    *,
    reset_fn,
    step_fn,
    inference_fn,
    env,
    output_path: Path,
    n_steps: int,
    camera: str | None,
    seed: int,
    render_every: int,
) -> None:
    import mediapy as media

    rng = jax.random.PRNGKey(seed)
    state = reset_fn(rng)
    rollout = [state.pipeline_state]
    for i in range(n_steps):
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = inference_fn(state.obs, act_rng)
        state = step_fn(state, ctrl)
        if i % render_every == 0:
            rollout.append(state.pipeline_state)

    frames = env.render(rollout, camera=camera)
    fps = 1.0 / (env.dt * render_every)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    media.write_video(str(output_path), frames, fps=fps)


def copy_checkpoint(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def make_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = DEFAULT_RUN_ROOT / f"go2_mjx_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_metadata(run_dir: Path, args: argparse.Namespace, extra: dict[str, Any]) -> None:
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "args": vars(args),
        "extra": extra,
    }
    with (run_dir / "training_meta.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)


def make_stage1_train_fn(args: argparse.Namespace):
    return functools.partial(
        apg.train,
        episode_length=args.stage1_episode_length,
        policy_updates=args.stage1_updates,
        horizon_length=args.horizon_length,
        num_envs=args.num_envs,
        learning_rate=args.stage1_lr,
        num_eval_envs=args.num_eval_envs,
        num_evals=args.num_evals,
        use_float64=args.use_float64,
        normalize_observations=True,
        network_factory=make_network_factory(STAGE1_HIDDEN),
        seed=args.seed,
    )


def make_stage2_train_fn(args: argparse.Namespace):
    return functools.partial(
        apg.train,
        episode_length=args.stage2_episode_length,
        policy_updates=args.stage2_updates,
        horizon_length=args.horizon_length,
        num_envs=args.num_envs,
        learning_rate=args.stage2_lr,
        schedule_decay=args.stage2_schedule_decay,
        num_eval_envs=args.num_eval_envs,
        num_evals=args.num_evals,
        use_float64=args.use_float64,
        normalize_observations=True,
        network_factory=make_network_factory(STAGE2_HIDDEN),
        seed=args.seed,
    )


def run_stage1(args: argparse.Namespace, run_dir: Path, xml_path: Path) -> tuple[Path, Any, Any]:
    print("[stage1] building envs", flush=True)
    env = envs.get_environment(
        "go2_mjx_local_trot",
        xml_path=str(xml_path),
        step_k=args.step_k,
        servo_kp=args.servo_kp,
    )
    eval_env = envs.get_environment(
        "go2_mjx_local_trot",
        xml_path=str(xml_path),
        step_k=args.step_k,
        servo_kp=args.servo_kp,
    )

    progress = make_progress_logger(args.stage1_updates, args.num_evals, "stage1")
    train_fn = make_stage1_train_fn(args)

    print("[stage1] training starts; first JIT compile can take a while", flush=True)
    started = time.time()
    make_inference_fn, params, metrics = train_fn(
        environment=env,
        progress_fn=progress.progress_fn,
        eval_env=eval_env,
    )
    print(f"[stage1] done in {(time.time() - started) / 60.0:.2f} min", flush=True)
    print(f"[stage1] final metrics: {metrics}", flush=True)

    stage_dir = run_dir / "stage1"
    checkpoint_path = run_dir / "checkpoints" / "trotting_2hz_policy"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    brax_model.save_params(str(checkpoint_path), params)
    save_progress_csv(progress.rows, stage_dir / "stage1_metrics.csv")
    save_training_curve(progress, stage_dir / "stage1_training_curve.png", "GO2 local MJX stage1 trot")
    if args.quick:
        print("[stage1] quick mode: skip copying checkpoint to policy_root", flush=True)
    else:
        copy_checkpoint(checkpoint_path, Path(args.policy_root).expanduser().resolve() / "trotting_2hz_policy")

    if args.render:
        demo_env = envs.training.EpisodeWrapper(env, episode_length=args.render_episode_length, action_repeat=1)
        try:
            render_rollout_to_video(
                reset_fn=jax.jit(demo_env.reset),
                step_fn=jax.jit(demo_env.step),
                inference_fn=jax.jit(make_inference_fn(params)),
                env=demo_env,
                output_path=stage_dir / "stage1_trot_rollout.mp4",
                n_steps=args.render_steps,
                camera=args.render_camera,
                seed=args.seed + 1,
                render_every=args.render_every,
            )
        except Exception as exc:
            print(f"[stage1] render failed after checkpoint was saved: {exc}", flush=True)

    return checkpoint_path, make_inference_fn, params


def run_stage2(
    args: argparse.Namespace,
    run_dir: Path,
    xml_path: Path,
    baseline_checkpoint: Path,
) -> Path:
    print(f"[stage2] loading baseline from {baseline_checkpoint}", flush=True)
    baseline_inference_fn, _ = make_inference_from_checkpoint(
        checkpoint_path=baseline_checkpoint,
        obs_size=BASELINE_OBS_SIZE,
        action_size=ACTION_SIZE,
        hidden_layer_sizes=STAGE1_HIDDEN,
    )

    env_kwargs = dict(
        baseline_inference_fn=baseline_inference_fn,
        xml_path=str(xml_path),
        target_vel=args.target_vel,
        step_k=args.step_k,
        servo_kp=args.servo_kp,
        clip_final_action=args.clip_final_action,
    )

    print("[stage2] building envs", flush=True)
    env = envs.get_environment("go2_mjx_local_forward", **env_kwargs)
    eval_env = envs.get_environment("go2_mjx_local_forward", **env_kwargs)

    progress = make_progress_logger(args.stage2_updates, args.num_evals, "stage2")
    train_fn = make_stage2_train_fn(args)

    print("[stage2] training starts; first JIT compile can take a while", flush=True)
    started = time.time()
    make_inference_fn, params, metrics = train_fn(
        environment=env,
        progress_fn=progress.progress_fn,
        eval_env=eval_env,
    )
    print(f"[stage2] done in {(time.time() - started) / 60.0:.2f} min", flush=True)
    print(f"[stage2] final metrics: {metrics}", flush=True)

    stage_dir = run_dir / "stage2"
    checkpoint_path = run_dir / "checkpoints" / "forward_locomotion_policy"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    brax_model.save_params(str(checkpoint_path), params)
    save_progress_csv(progress.rows, stage_dir / "stage2_metrics.csv")
    save_training_curve(progress, stage_dir / "stage2_training_curve.png", "GO2 local MJX stage2 forward residual")
    if args.quick:
        print("[stage2] quick mode: skip copying checkpoint to policy_root", flush=True)
    else:
        copy_checkpoint(checkpoint_path, Path(args.policy_root).expanduser().resolve() / "forward_locomotion_policy")

    if args.render:
        demo_env = envs.training.EpisodeWrapper(env, episode_length=args.render_episode_length, action_repeat=1)
        try:
            render_rollout_to_video(
                reset_fn=jax.jit(demo_env.reset),
                step_fn=jax.jit(demo_env.step),
                inference_fn=jax.jit(make_inference_fn(params)),
                env=demo_env,
                output_path=stage_dir / "stage2_forward_rollout.mp4",
                n_steps=args.render_steps,
                camera=args.render_camera or "track",
                seed=args.seed + 2,
                render_every=args.render_every,
            )
        except Exception as exc:
            print(f"[stage2] render failed after checkpoint was saved: {exc}", flush=True)

    return checkpoint_path


def validate_setup(args: argparse.Namespace, xml_path: Path, baseline_checkpoint: Path | None = None) -> None:
    print("[validate] XML:", xml_path, flush=True)
    stage1 = envs.get_environment(
        "go2_mjx_local_trot",
        xml_path=str(xml_path),
        step_k=args.step_k,
        servo_kp=args.servo_kp,
    )
    state = stage1.reset(jax.random.PRNGKey(args.seed))
    print("[validate] stage1 obs shape:", tuple(state.obs.shape), flush=True)
    state = stage1.step(state, jp.zeros((ACTION_SIZE,)))
    print("[validate] stage1 step reward:", float(state.reward), "done:", float(state.done), flush=True)

    if baseline_checkpoint is None or not baseline_checkpoint.exists():
        print("[validate] stage2 skipped; no baseline checkpoint supplied/found", flush=True)
        return

    baseline_inference_fn, _ = make_inference_from_checkpoint(
        checkpoint_path=baseline_checkpoint,
        obs_size=BASELINE_OBS_SIZE,
        action_size=ACTION_SIZE,
        hidden_layer_sizes=STAGE1_HIDDEN,
    )
    stage2 = envs.get_environment(
        "go2_mjx_local_forward",
        baseline_inference_fn=baseline_inference_fn,
        xml_path=str(xml_path),
        target_vel=args.target_vel,
        step_k=args.step_k,
        servo_kp=args.servo_kp,
        clip_final_action=args.clip_final_action,
    )
    state = stage2.reset(jax.random.PRNGKey(args.seed))
    print("[validate] stage2 obs shape:", tuple(state.obs.shape), flush=True)
    state = stage2.step(state, jp.zeros((ACTION_SIZE,)))
    print("[validate] stage2 step reward:", float(state.reward), "done:", float(state.done), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local GO2 MJX two-stage FoPG/APG training script.")
    parser.add_argument("--stage", choices=["validate", "stage1", "stage2", "both"], default="both")
    parser.add_argument("--xml_path", type=str, default=str(DEFAULT_XML_PATH))
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--policy_root", type=str, default=str(DEFAULT_POLICY_ROOT))
    parser.add_argument("--baseline_checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--step_k", type=int, default=13)
    parser.add_argument("--servo_kp", type=float, default=230.0)
    parser.add_argument("--target_vel", type=float, default=0.75)

    parser.add_argument("--stage1_updates", type=int, default=499)
    parser.add_argument("--stage2_updates", type=int, default=499)
    parser.add_argument("--stage1_episode_length", type=int, default=240)
    parser.add_argument("--stage2_episode_length", type=int, default=1000)
    parser.add_argument("--horizon_length", type=int, default=32)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--num_eval_envs", type=int, default=64)
    parser.add_argument("--num_evals", type=int, default=11)
    parser.add_argument("--stage1_lr", type=float, default=1e-4)
    parser.add_argument("--stage2_lr", type=float, default=1.5e-4)
    parser.add_argument("--stage2_schedule_decay", type=float, default=0.995)

    parser.add_argument("--use_float64", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug_nans", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--matmul_precision", choices=["default", "high", "highest"], default="high")
    parser.add_argument(
        "--clip_final_action",
        action="store_true",
        default=False,
        help="Clip baseline+residual action before scaling. Leave unset for GO2_train.ipynb fidelity.",
    )

    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render_steps", type=int, default=200)
    parser.add_argument("--render_every", type=int, default=3)
    parser.add_argument("--render_episode_length", type=int, default=1000)
    parser.add_argument("--render_camera", type=str, default=None)

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Tiny smoke-test settings. Useful after restoring XML, not for real training.",
    )
    return parser.parse_args()


def apply_quick_settings(args: argparse.Namespace) -> None:
    if not args.quick:
        return
    args.stage1_updates = min(args.stage1_updates, 2)
    args.stage2_updates = min(args.stage2_updates, 2)
    args.stage1_episode_length = min(args.stage1_episode_length, 32)
    args.stage2_episode_length = min(args.stage2_episode_length, 64)
    args.horizon_length = min(args.horizon_length, 8)
    args.num_envs = min(args.num_envs, 1)
    args.num_eval_envs = min(args.num_eval_envs, 1)
    args.num_evals = min(args.num_evals, 2)


def resolve_baseline_checkpoint(args: argparse.Namespace, run_dir: Path) -> Path | None:
    if args.baseline_checkpoint:
        return Path(args.baseline_checkpoint).expanduser().resolve()
    run_checkpoint = run_dir / "checkpoints" / "trotting_2hz_policy"
    if run_checkpoint.exists():
        return run_checkpoint
    policy_checkpoint = Path(args.policy_root).expanduser().resolve() / "trotting_2hz_policy"
    if policy_checkpoint.exists():
        return policy_checkpoint
    existing_checkpoint = SCRIPT_DIR / "go2_policy_export" / "trotting_2hz_policy"
    if existing_checkpoint.exists():
        return existing_checkpoint
    return None


def main() -> int:
    args = parse_args()
    apply_quick_settings(args)
    configure_jax(args)
    register_envs()

    xml_path = Path(args.xml_path).expanduser().resolve()
    require_xml(xml_path)
    run_dir = make_run_dir(args)

    print("=" * 88, flush=True)
    print("GO2 local MJX training", flush=True)
    print(f"stage       : {args.stage}", flush=True)
    print(f"xml_path    : {xml_path}", flush=True)
    print(f"run_dir     : {run_dir}", flush=True)
    print(f"policy_root : {Path(args.policy_root).expanduser().resolve()}", flush=True)
    print(f"JAX devices : {[str(d) for d in jax.devices()]}", flush=True)
    print("=" * 88, flush=True)

    baseline_checkpoint = resolve_baseline_checkpoint(args, run_dir)
    outputs: dict[str, Any] = {}

    if args.stage == "validate":
        validate_setup(args, xml_path, baseline_checkpoint)
        write_metadata(run_dir, args, outputs)
        return 0

    if args.stage in ("stage1", "both"):
        baseline_checkpoint, _, _ = run_stage1(args, run_dir, xml_path)
        outputs["stage1_checkpoint"] = str(baseline_checkpoint)

    if args.stage in ("stage2", "both"):
        if baseline_checkpoint is None or not baseline_checkpoint.exists():
            raise FileNotFoundError(
                "stage2 需要 baseline checkpoint。请先运行 --stage stage1，"
                "或通过 --baseline_checkpoint 指向 trotting_2hz_policy。"
            )
        forward_checkpoint = run_stage2(args, run_dir, xml_path, baseline_checkpoint)
        outputs["stage2_checkpoint"] = str(forward_checkpoint)

    write_metadata(run_dir, args, outputs)
    print("[done] outputs:", json.dumps(outputs, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
