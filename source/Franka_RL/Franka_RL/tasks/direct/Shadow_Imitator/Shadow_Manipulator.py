from __future__ import annotations
import torch
from torch import Tensor
from typing import Dict, List, Tuple
import numpy as np
from .Shadow_Imitator_env import ShandImitatorEnv

from isaaclab.utils.math import sample_uniform, quat_error_magnitude, transform_points, quat_mul, quat_conjugate

class ShadowManipulatorEnv(ShandImitatorEnv):
    def _get_rewards(self):
        states = {}
        target_states = {}

        states["base_state"] = torch.cat([self.wrist_pos, self.wrist_vel], dim=-1)
        states["joints_state"] = torch.cat([self.body_pos, self.body_vel], dim=-1)
        states["q"] = self.dof_pos
        states["dq"] = self.dof_vel
        states["manip_obj_pos"] = self.obj_pos[:, :3]
        states["manip_obj_quat"] = self.obj_pos[:, 3:]
        states["manip_obj_vel"] = self.obj_vel[:, :3]
        states["manip_obj_ang_vel"] = self.obj_vel[:, 3:]

        env_ids = torch.arange(self.num_envs).to(device=self.device)
        target_states["wrist_pos"] = self.target_wrist_pos_seq[env_ids, self.target_jt_j, :3]
        target_states["wrist_quat"] = self.target_wrist_pos_seq[env_ids, self.target_jt_j, 3:]
        target_states["wrist_vel"] = self.target_wrist_vel_seq[env_ids, self.target_jt_j, :3]
        target_states["wrist_ang_vel"] = self.target_wrist_vel_seq[env_ids, self.target_jt_j, 3:]
        target_states["joints_pos"] = self.target_body_pos_seq[env_ids, self.target_jt_j, 1:, :3]
        target_states["joints_vel"] = self.target_body_vel_seq[env_ids, self.target_jt_j, 1:, :3]
        target_states["manip_obj_pos"] = self.target_obj_pos_seq[env_ids, self.target_jt_j, :3]
        target_states["manip_obj_quat"] = self.target_obj_pos_seq[env_ids, self.target_jt_j, 3:]
        target_states["manip_obj_vel"] = self.target_obj_vel_seq[env_ids, self.target_jt_j, :3]
        target_states["manip_obj_ang_vel"] = self.target_obj_vel_seq[env_ids, self.target_jt_j, 3:]

        target_states["power"] = torch.abs(torch.multiply(self.dof_torque, self.dof_vel)).sum(dim=-1)
        wrist_power = torch.abs(torch.sum(self.actions[:, :3] * self.wrist_vel[:, :3], dim=-1))
        wrist_power += torch.abs(torch.sum(self.actions[:, 3:6] * self.wrist_vel[:, 3:], dim=-1))
        target_states["wrist_power"] = wrist_power

        last_step = self._sim_step_counter
        if self.tighten_method == "None":
            scale_factor = 1.0
        elif self.tighten_method == "const":
            scale_factor = self.tighten_factor
        elif self.tighten_method == "linear_decay":
            scale_factor = 1 - (1 - self.tighten_factor) / self.tighten_steps * min(last_step, self.tighten_steps)
        elif self.tighten_method == "exp_decay":
            scale_factor = (np.e * 2) ** (-1 * last_step / self.tighten_steps) * (
                1 - self.tighten_factor
            ) + self.tighten_factor
        elif self.tighten_method == "cos":
            scale_factor = (self.tighten_factor) + np.abs(
                -1 * (1 - self.tighten_factor) * np.cos(last_step / self.tighten_steps * np.pi)
            ) * (2 ** (-1 * last_step / self.tighten_steps))
        else:
            raise NotImplementedError

        (
            self.reward_buf,
            self.reset_buf[:],
            self.reset_time_outs[:],
            self.reset_terminated[:],
            reward_dict,
            self.error_buf,
        ) = compute_manipulation_reward(
            self.reset_buf,
            self.target_jt_j,
            self.running_frame_len,
            self.actions,
            states,
            target_states,
            self.traj_len_seq,
            scale_factor,
            self.dexhand.weight_idx,
        )

        for rew, val in reward_dict.items():
            self._log_reward("Episode_Reward/" + rew, val)

def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))

def sample_int_from_float(x):
    if int(x) == x:
        return int(x)
    return int(x) if np.random.rand() < (x - int(x)) else int(x) + 1

def to_euler_angles(quat):
    assert quat.shape[-1] == 7
    from scipy.spatial.transform import Rotation as R

    r = R.from_quat(quat[3:].cpu().numpy())
    v = quat[:3].cpu().numpy()
    r = r.as_euler("xyz", degrees=False)
    # concat v and r
    return torch.tensor(np.concatenate((v, r), axis=-1), dtype=torch.float32)

@torch.jit.script
def quat_to_angle_axis(q):
    # type: (Tensor) -> Tuple[Tensor, Tensor]
    # computes axis-angle representation from quaternion q
    # q must be normalized
    min_theta = 1e-5
    qx, qy, qz, qw = 1, 2, 3, 0

    sin_theta = torch.sqrt(1 - q[..., qw] * q[..., qw])
    angle = 2 * torch.acos(q[..., qw])
    angle = normalize_angle(angle)
    sin_theta_expand = sin_theta.unsqueeze(-1)
    axis = q[..., qx:qw] / sin_theta_expand

    mask = torch.abs(sin_theta) > min_theta
    default_axis = torch.zeros_like(axis)
    default_axis[..., -1] = 1

    angle = torch.where(mask, angle, torch.zeros_like(angle))
    mask_expand = mask.unsqueeze(-1)
    axis = torch.where(mask_expand, axis, default_axis)
    return angle, axis

@torch.jit.script
def compute_manipulation_reward(
    reset_buf: Tensor,
    progress_buf: Tensor,
    running_progress_buf: Tensor,
    actions: Tensor,
    states: Dict[str, Tensor],
    target_states: Dict[str, Tensor],
    max_length: List[int],
    scale_factor: float,
    dexhand_weight_idx: Dict[str, List[int]],
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

    # type: (Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor], Dict[str, Tensor], Tensor, float, Dict[str, List[int]]) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor], Tensor]

    # end effector pose reward
    current_eef_pos = states["base_state"][:, :3]
    current_eef_quat = states["base_state"][:, 3:7]

    target_eef_pos = target_states["wrist_pos"]
    target_eef_quat = target_states["wrist_quat"]
    diff_eef_pos = target_eef_pos - current_eef_pos
    diff_eef_pos_dist = torch.norm(diff_eef_pos, dim=-1)

    current_eef_vel = states["base_state"][:, 7:10]
    current_eef_ang_vel = states["base_state"][:, 10:13]
    target_eef_vel = target_states["wrist_vel"]
    target_eef_ang_vel = target_states["wrist_ang_vel"]

    diff_eef_vel = target_eef_vel - current_eef_vel
    diff_eef_ang_vel = target_eef_ang_vel - current_eef_ang_vel

    joints_pos = states["joints_state"][:, 1:, :3]
    target_joints_pos = target_states["joints_pos"]
    diff_joints_pos = target_joints_pos - joints_pos
    diff_joints_pos_dist = torch.norm(diff_joints_pos, dim=-1)

    # ? assign different weights to different joints
    # assert diff_joints_pos_dist.shape[1] == 17  # ignore the base joint
    diff_thumb_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["thumb_tip"]]].mean(dim=-1)
    diff_index_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["index_tip"]]].mean(dim=-1)
    diff_middle_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["middle_tip"]]].mean(dim=-1)
    diff_ring_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["ring_tip"]]].mean(dim=-1)
    diff_pinky_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["pinky_tip"]]].mean(dim=-1)
    diff_level_1_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["level_1_joints"]]].mean(dim=-1)
    diff_level_2_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["level_2_joints"]]].mean(dim=-1)

    joints_vel = states["joints_state"][:, 1:, 7:10]
    target_joints_vel = target_states["joints_vel"]
    diff_joints_vel = target_joints_vel - joints_vel

    reward_eef_pos = torch.exp(-40 * diff_eef_pos_dist)
    reward_thumb_tip_pos = torch.exp(-100 * diff_thumb_tip_pos_dist)
    reward_index_tip_pos = torch.exp(-90 * diff_index_tip_pos_dist)
    reward_middle_tip_pos = torch.exp(-80 * diff_middle_tip_pos_dist)
    reward_pinky_tip_pos = torch.exp(-60 * diff_pinky_tip_pos_dist)
    reward_ring_tip_pos = torch.exp(-60 * diff_ring_tip_pos_dist)
    reward_level_1_pos = torch.exp(-50 * diff_level_1_pos_dist)
    reward_level_2_pos = torch.exp(-40 * diff_level_2_pos_dist)

    reward_eef_vel = torch.exp(-1 * diff_eef_vel.abs().mean(dim=-1))
    reward_eef_ang_vel = torch.exp(-1 * diff_eef_ang_vel.abs().mean(dim=-1))
    reward_joints_vel = torch.exp(-1 * diff_joints_vel.abs().mean(dim=-1).mean(-1))
    current_dof_vel = states["dq"]

    diff_eef_rot = quat_mul(target_eef_quat, quat_conjugate(current_eef_quat))
    diff_eef_rot_angle = quat_to_angle_axis(diff_eef_rot)[0]
    reward_eef_rot = torch.exp(-1 * (diff_eef_rot_angle).abs())

    # object pose reward
    current_obj_pos = states["manip_obj_pos"]
    current_obj_quat = states["manip_obj_quat"]

    target_obj_pos = target_states["manip_obj_pos"]
    target_obj_quat = target_states["manip_obj_quat"]
    diff_obj_pos = target_obj_pos - current_obj_pos
    diff_obj_pos_dist = torch.norm(diff_obj_pos, dim=-1)

    reward_obj_pos = torch.exp(-80 * diff_obj_pos_dist)

    diff_obj_rot = quat_mul(target_obj_quat, quat_conjugate(current_obj_quat))
    diff_obj_rot_angle = quat_to_angle_axis(diff_obj_rot)[0]
    reward_obj_rot = torch.exp(-3 * (diff_obj_rot_angle).abs())

    current_obj_vel = states["manip_obj_vel"]
    target_obj_vel = target_states["manip_obj_vel"]
    diff_obj_vel = target_obj_vel - current_obj_vel
    reward_obj_vel = torch.exp(-1 * diff_obj_vel.abs().mean(dim=-1))

    current_obj_ang_vel = states["manip_obj_ang_vel"]
    target_obj_ang_vel = target_states["manip_obj_ang_vel"]
    diff_obj_ang_vel = target_obj_ang_vel - current_obj_ang_vel
    reward_obj_ang_vel = torch.exp(-1 * diff_obj_ang_vel.abs().mean(dim=-1))

    reward_power = torch.exp(-10 * target_states["power"])
    reward_wrist_power = torch.exp(-2 * target_states["wrist_power"])

    finger_tip_force = target_states["tip_force"]
    finger_tip_distance = target_states["tips_distance"]
    contact_range = [0.02, 0.03]
    finger_tip_weight = torch.clamp(
        (contact_range[1] - finger_tip_distance) / (contact_range[1] - contact_range[0]), 0, 1
    )
    finger_tip_force_masked = finger_tip_force * finger_tip_weight[:, :, None]

    reward_finger_tip_force = torch.exp(-1 * (1 / (torch.norm(finger_tip_force_masked, dim=-1).sum(-1) + 1e-5)))

    error_buf = (
        (torch.norm(current_eef_vel, dim=-1) > 100)
        | (torch.norm(current_eef_ang_vel, dim=-1) > 200)
        | (torch.norm(joints_vel, dim=-1).mean(-1) > 100)
        | (torch.abs(current_dof_vel).mean(-1) > 200)
        | (torch.norm(current_obj_vel, dim=-1) > 100)
        | (torch.norm(current_obj_ang_vel, dim=-1) > 200)
    )  # sanity check

    failed_execute = (
        (
            (diff_obj_pos_dist > 0.02 / 0.343 * scale_factor**3)  # TODO
            | (diff_thumb_tip_pos_dist > 0.04 / 0.7 * scale_factor)
            | (diff_index_tip_pos_dist > 0.045 / 0.7 * scale_factor)
            | (diff_middle_tip_pos_dist > 0.05 / 0.7 * scale_factor)
            | (diff_pinky_tip_pos_dist > 0.06 / 0.7 * scale_factor)
            | (diff_ring_tip_pos_dist > 0.06 / 0.7 * scale_factor)
            | (diff_level_1_pos_dist > 0.07 / 0.7 * scale_factor)
            | (diff_level_2_pos_dist > 0.08 / 0.7 * scale_factor)
            | (diff_obj_rot_angle.abs() / np.pi * 180 > 30 / 0.343 * scale_factor**3)  # TODO
            | torch.any((finger_tip_distance < 0.005) & ~(target_states["tip_contact_state"].any(1)), dim=-1)
        )
        & (running_progress_buf >= 8)
    ) | error_buf
    reward_execute = (
        0.1 * reward_eef_pos
        + 0.6 * reward_eef_rot
        + 0.9 * reward_thumb_tip_pos
        + 0.8 * reward_index_tip_pos
        + 0.75 * reward_middle_tip_pos
        + 0.6 * reward_pinky_tip_pos
        + 0.6 * reward_ring_tip_pos
        + 0.5 * reward_level_1_pos
        + 0.3 * reward_level_2_pos
        + 5.0 * reward_obj_pos
        + 1.0 * reward_obj_rot
        + 0.1 * reward_eef_vel
        + 0.05 * reward_eef_ang_vel
        + 0.1 * reward_joints_vel
        + 0.1 * reward_obj_vel
        + 0.1 * reward_obj_ang_vel
        + 1.0 * reward_finger_tip_force
        + 0.5 * reward_power
        + 0.5 * reward_wrist_power
    )

    succeeded = (
        progress_buf + 1 + 3 >= max_length
    ) & ~failed_execute  # reached the end of the trajectory, +3 for max future 3 steps
    reset_buf = torch.where(
        succeeded | failed_execute,
        torch.ones_like(reset_buf),
        reset_buf,
    )
    reward_dict = {
        "reward_eef_pos": reward_eef_pos,
        "reward_eef_rot": reward_eef_rot,
        "reward_eef_vel": reward_eef_vel,
        "reward_eef_ang_vel": reward_eef_ang_vel,
        "reward_joints_vel": reward_joints_vel,
        "reward_obj_pos": reward_obj_pos,
        "reward_obj_rot": reward_obj_rot,
        "reward_obj_vel": reward_obj_vel,
        "reward_obj_ang_vel": reward_obj_ang_vel,
        "reward_joints_pos": (
            reward_thumb_tip_pos
            + reward_index_tip_pos
            + reward_middle_tip_pos
            + reward_pinky_tip_pos
            + reward_ring_tip_pos
            + reward_level_1_pos
            + reward_level_2_pos
        ),
        "reward_power": reward_power,
        "reward_wrist_power": reward_wrist_power,
        "reward_finger_tip_force": reward_finger_tip_force,
    }

    return reward_execute, reset_buf, succeeded, failed_execute, reward_dict, error_buf