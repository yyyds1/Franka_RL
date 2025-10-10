# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import os
import random
import pendulum
import json
import zarr
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform, quat_error_magnitude, transform_points
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import Camera

from .Shadow_Imitator_env_cfg import ShandImitatorEnvCfg
from Franka_RL.robots import DexHandFactory
from Franka_RL.dataset import DataFactory

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

class ShandImitator(DirectRLEnv):
    cfg: ShandImitatorEnvCfg

    def __init__(self, cfg: ShandImitatorEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.obs_future_length = self.cfg.future_frame
        self.action_joint_scale = self.cfg.action_joint_scale
        self.action_pos_scale = self.cfg.action_pos_scale
        self.action_rot_scale = self.cfg.action_rot_scale
        self.action_moving_scale = self.cfg.action_moving_scale
        self.joint_limits = self.dexhand.dof_limit

    def _init_traj(self):

        self.dataset = DataFactory.create_data(data_type=self.cfg.dataset_type, side=self.cfg.side, device=self.device, dexhand=self.dexhand)
        self.traj_num = self.dataset.traj_num
        self.traj_len = self.dataset.traj_len
        self.traj_len_max = self.dataset.max_traj_length

        if self.cfg.human_resample_on_env_reset:
            self.target_jt_i = torch.randint(0, self.traj_num, (self.num_envs, )).to(dtype=torch.int, device=self.device)
        else:
            self.target_jt_i = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
        self.target_jt_j = (torch.rand(self.num_envs).to(dtype=torch.float32, device=self.device) * self.traj_len[self.target_jt_i]).to(torch.int)

        self.target_wrist_pos_seq = self.dataset.wrist_pos[self.target_jt_i]
        self.target_wrist_vel_seq = self.dataset.wrist_vel[self.target_jt_i]
        self.target_joint_pos_seq = self.dataset.joints_pos[self.target_jt_i]
        self.target_body_pos_seq = self.dataset.body_pos[self.target_jt_i]
        self.target_obj_pos_seq = self.dataset.obj_pose[self.target_jt_i]
        self.obj_id_seq = self.dataset.obj_id
        self.obj_pcl = self.dataset.obj_pcl

        # self.target_wrist_pos = self.target_wrist_pos_seq[:, self.target_jt_j]
        # self.target_wrist_vel = self.target_wrist_vel_seq[:, self.target_jt_j]
        # self.target_joint_pos = self.target_joint_pos_seq[:, self.target_jt_j]
        # self.target_body_pos = self.target_body_pos_seq[:, self.target_jt_j]
        # self.target_obj_pos = self.target_obj_pos_seq[:, self.target_jt_j]

        self.target_jt_dt = 1 / self.cfg.human_freq  # type: ignore
        self.target_jt_update_steps = self.target_jt_dt / self.dt  # not necessary integer
        assert self.dt <= self.target_jt_dt
        self.target_jt_update_steps_int = sample_int_from_float(self.target_jt_update_steps)

    def update_target(self, reset_env_ids):
        resample_i = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if reset_env_ids.shape[0] != 0:
            if self.cfg.human_resample_on_env_reset:
                resample_i[reset_env_ids] = True
                self.target_jt_i[reset_env_ids] = torch.randint_like(reset_env_ids, low=0, high=self.traj_num).to(dtype=torch.int, device=self.device)
                self.target_jt_j[reset_env_ids] = (torch.rand(reset_env_ids.shape[0]).to(dtype=torch.float32, device=self.device) * self.traj_len[self.target_jt_i[reset_env_ids]]).to(torch.int)
                self.target_wrist_pos_seq[reset_env_ids] = self.dataset.wrist_pos[self.target_jt_i[reset_env_ids]]
                self.target_wrist_vel_seq[reset_env_ids] = self.dataset.wrist_vel[self.target_jt_i[reset_env_ids]]
                self.target_joint_pos_seq[reset_env_ids] = self.dataset.joints_pos[self.target_jt_i[reset_env_ids]]
                self.target_body_pos_seq[reset_env_ids] = self.dataset.body_pos[self.target_jt_i[reset_env_ids]]
                self.target_obj_pos_seq[reset_env_ids] = self.dataset.obj_pose[self.target_jt_i[reset_env_ids]]
            else:
                self.target_jt_j[reset_env_ids] = (torch.rand(reset_env_ids.shape[0]).to(dtype=torch.float32, device=self.device) * self.traj_len[self.target_jt_i[reset_env_ids]]).to(torch.int)

                # self.target_jt = self.target_jt_seq[self.target_jt_i, self.target_jt_j]
                # self.target_eepose = self.target_eepose_seq[self.target_jt_i, self.target_jt_j]

        if self.common_step_counter % self.target_jt_update_steps_int == 0:
            if self.common_step_counter == 0:
                self.target_jt_j += 1
            else:
                self.target_jt_j += self.move_on


    def _setup_scene(self):
        # create robot
        self.dexhand = DexHandFactory.create_hand(dexhand_type=self.cfg.robot, side=self.cfg.side)
        self.robot_cfg = ArticulationCfg(
            prim_path=f"/World/envs/env_.*/robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=self.dexhand._usd_path,
                activate_contact_sensors=True,
                # visible=False,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    retain_accelerations=True,
                    max_depenetration_velocity=1000.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=True,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=0,
                    sleep_threshold=0.005,
                    stabilization_threshold=0.0005,
                ),
                # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
                joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
                fixed_tendons_props=sim_utils.FixedTendonPropertiesCfg(limit_stiffness=30.0, damping=0.1),
            ),
            init_state=self.dexhand.init_state,
            actuators=self.dexhand.actuators,
            soft_joint_pos_limit_factor=1.0,
        )
        self.robot = Articulation(self.robot_cfg)

        # load dataset
        self.dt = self.cfg.decimation * self.cfg.sim.dt
        self.future_frame = 5
        self.move_on = 1
        self._init_traj()

        # create object
        obj_cfg_list = []
        for usd_path in self.dataset.obj_usd:
            obj_name = os.path.basename(usd_path).split('.')[0]
            obj_cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/object",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=usd_path,
                    scale=(1.00, 1.00, 1.00),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=False,
                        disable_gravity=True,
                        enable_gyroscopic_forces=True,
                        solver_position_iteration_count=8,
                        solver_velocity_iteration_count=0,
                        sleep_threshold=0.005,
                        stabilization_threshold=0.0025,
                        max_depenetration_velocity=1000.0,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(density=1.0),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, 0.0, 0.0),
                    rot=(0.0, 0.0, 0.0, 0.0),
                ),
                debug_vis=False,
            )
            self.obj_cfg_list.append(obj_cfg)

        self.object = RigidObject([obj_cfg_list[idx] for idx in self.obj_id_seq])

        # create sensors
        self.contact_sensor = {}
        for contact_body in self.dexhand.contact_body_names:
            sensor_cfg = ContactSensorCfg(
                prim_path="/World/envs/env_.*/Robot/" + contact_body,
                history_length=3,
                update_period=0.005,
                track_air_time=True,
                debug_vis=False,
                filter_prim_paths_expr=["/World/envs/env_.*/object"],
            )
            self.contact_sensor[contact_body] = ContactSensor(sensor_cfg)
            self.scene.sensors[contact_body] = self.contact_sensor[contact_body]
        # Create camera
        # self.camera = Camera(self.cfg.camera)
        # self.camera.set_world_poses_from_view(eyes=[2.0, 2.0, 2.0], targets=[0.0, 0.0, 0.0])

        # add ground plane
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs  # type: ignore
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing  # type: ignore
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)  # type: ignore
        # clone, filter, and replicate
        # self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        self.scene.rigid_objects["object"] = self.object
        # add articultion to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


        # Markers
        frame_marker_cfg = FRAME_MARKER_CFG.copy()
        frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        self.ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
        self.goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))


    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # limit the action in joint limitations
        # self.action[:, :3] -- force applied to root
        # self.action[:, 3:6] -- torque applied to root
        # self.action[:, 6:] -- dexhand joint target
        self.last_actions = self.actions
        self.actions[:, :3] = actions[:, :3] * self.dt * self.action_pos_scale * self.action_moving_scale + self.last_actions[:, :3] * (1 - self.action_moving_scale)
        self.actions[:, :3] = actions[:, 3:6] * self.dt * self.action_rot_scale * self.action_moving_scale + self.last_actions[:, 3:6] * (1 - self.action_moving_scale)
        self.actions[:, 6:] = ((torch.tanh(actions[:, 6:]) + 1) / 2 * (self.joint_limits[:, 1] - self.joint_limits[:, 0]) + self.joint_limits[:, 0]) * self.action_moving_scale + self.last_actions[:, 6:] * (1 - self.action_moving_scale)
        # self.actions[:, 6:] = torch.clamp(torch.tanh(actions[:, 6:] * self.action_scale) * torch.pi / 6 + self.dof_pos, min = self.joint_limits[:, 0], max=self.joint_limits[:, 1])
        # self.actions = torch.tanh(actions * self.action_scale) * self.joint_gears
        pass

    def _apply_action(self) -> None:
        self.robot.set_external_force_and_torque(forces=self.actions[:, :3].unsqueeze(1), torques=self.actions[:, 3:6].unsqueeze(1), body_ids=[self.robot.body_names.index(self.dexhand.wrist_name)])
        self.robot.set_joint_position_target(self.actions[:, 6:], joint_ids=range(0, self.dexhand.n_dofs))
        # self.robot.set_joint_position_target(self.actions, joint_ids=range(0, 7))
        # self.robot.set_jointset_joint_effort_target(self.actions, joint_ids=range(0, 7))

    def _compute_intermediate_values(self):
        self.dof_pos, self.dof_vel = self.robot.data.joint_pos, self.robot.data.joint_vel
        self.dof_pos = self.dof_pos[:, :7]
        self.dof_vel = self.dof_vel[:, :7]
        # self.eepose = self.robot.data.body_state_w[:, self.robot.body_names.index("panda_hand"), :7] - F.pad(
        #     self.scene.env_origins, (0, 4)
        # )
        # self.eepose_diff = self.target_eepose - self.eepose
        # self.eepose_error = self.reward_pose(self.eepose, self.target_eepose, tanh_weight=0.0, pos_norm=2)[0]

        self.wrist_pos = self.robot.data.body_state_w[:, self.robot.body_names.index(self.dexhand.wrist_name), :7] - F.pad(self.scene.env_origins, (0, 4))
        self.wrist_vel = self.robot.data.body_state_w[:, self.robot.body_names.index(self.dexhand.wrist_name), 7:]
        self.body_pos = self.robot.data.body_state_w[:, :, :7] - F.pad(self.scene.env_origins, (0, 4))
        self.body_vel = self.robot.data.body_state_w[:, :, 7:]

        self.obj_pos = self.object.root_state_w[:, :7]
        self.obj_vel = self.object.root_state_w[:, 7:]
        self.obj_com_pos = transform_points(self.object.data.body_com_pos_b, self.obj_pos[:, :3], self.obj_pos[:, 3:]).squeeze()
        self.obj_curr_pcl = transform_points(self.obj_pcl[self.obj_id_seq], self.obj_pos[:, :3], self.obj_pos[:, 3:])
        

    def _get_observations(self) -> dict:
        # proprioception state
        proprioception = {}
        proprioception["joint_pos"] = self.dof_pos # num_dof
        proprioception["joint_vel"] = self.dof_vel # num_dof
        proprioception["cos_joint_pos"] = torch.cos(self.dof_pos) # num_dof
        proprioception["sin_joint_pos"] = torch.sin(self.dof_pos) # num_dof
        proprioception["wrist_pos"] = torch.cat([torch.zeros_like(self.wrist_pos[:, :3]), self.wrist_pos[:, 3:]], dim=-1).unsqueeze(1) # ignore wrist position 7
        proprioception["obj_trans"] = self.obj_pos[:, :3] - self.wrist_pos[:, :3] # 3
        proprioception["obj_vel"] = self.obj_vel # 6
        tip_force = torch.stack([self.contact_sensor.data.net_forces_w[k] for k in self.dexhand.contact_body_names], axis=1)
        proprioception["tip_force"] = torch.cat([tip_force, torch.norm(tip_force, dim=-1, keepdim=True)], dim=-1) # add force magnitude 15 + 5
        proprioception["obj_com"] = self.obj_com_pos # 3
        proprioception["obj_gravity"] = self.object.data.default_mass * self.sim.cfg.gravity # 3

        # target state
        next_target_state = {}
        cur_idx = self.target_jt_j + 1
        cur_idx = torch.stack([cur_idx + t for t in range(self.obs_future_length)], dim=-1)
        cur_idx = torch.clamp(cur_idx, torch.zeros_like(self.traj_len), self.traj_len - 1)
        nE, nT = self.target_wrist_pos_seq.shape[:2]
        nF = self.obs_future_length

        def indicing(data, idx):
            assert data.shape[0] == nE and data.shape[1] == nT
            remaining_shape = data.shape[2:]
            expanded_idx = idx
            for _ in remaining_shape:
                expanded_idx = expanded_idx.unsqueeze(-1)
            expanded_idx = expanded_idx.expand(-1, -1, *remaining_shape)
            return torch.gather(data, 1, expanded_idx)

        target_wrist_pos = indicing(self.target_wrist_pos_seq[:, :3], cur_idx)
        cur_wrist_pos = self.wrist_pos[:, :3]
        next_target_state["delta_wrist_pos"] = (target_wrist_pos - cur_wrist_pos[:, None]).reshape(nE, -1) # 3

        target_wrist_vel = indicing(self.target_wrist_vel_seq[:, :3], cur_idx)
        cur_wrist_vel = self.wrist_vel[:, :3]
        next_target_state["wrist_vel"] = target_wrist_vel.reshape(nE, -1) # 3
        next_target_state["delta_wrist_vel"] = (target_wrist_vel - cur_wrist_vel[:, None]).reshape(nE, -1) # 3

        target_wrist_rot = indicing(self.target_wrist_pos_seq[:, 3:], cur_idx)
        cur_wrist_rot = self.wrist_pos[:, 3:]

        next_target_state["wrist_quat"] = aa_to_quat(target_wrist_rot.reshape(nE * nF, -1))[:, [1, 2, 3, 0]]
        next_target_state["delta_wrist_quat"] = quat_mul(
            cur_wrist_rot[:, None].repeat(1, nF, 1).reshape(nE * nF, -1),
            quat_conjugate(next_target_state["wrist_quat"]),
        ).reshape(nE, -1)
        next_target_state["wrist_quat"] = next_target_state["wrist_quat"].reshape(nE, -1)

        target_wrist_ang_vel = indicing(self.demo_data["wrist_angular_velocity"], cur_idx)
        cur_wrist_ang_vel = self.states["base_state"][:, 10:13]
        next_target_state["wrist_ang_vel"] = target_wrist_ang_vel.reshape(nE, -1)
        next_target_state["delta_wrist_ang_vel"] = (target_wrist_ang_vel - cur_wrist_ang_vel[:, None]).reshape(nE, -1)

        target_joints_pos = indicing(self.demo_data["mano_joints"], cur_idx).reshape(nE, nF, -1, 3)
        cur_joint_pos = self.states["joints_state"][:, 1:, :3]  # skip the base joint
        next_target_state["delta_joints_pos"] = (target_joints_pos - cur_joint_pos[:, None]).reshape(self.num_envs, -1)

        target_joints_vel = indicing(self.demo_data["mano_joints_velocity"], cur_idx).reshape(nE, nF, -1, 3)
        cur_joint_vel = self.states["joints_state"][:, 1:, 7:10]  # skip the base joint
        next_target_state["joints_vel"] = target_joints_vel.reshape(self.num_envs, -1)
        next_target_state["delta_joints_vel"] = (target_joints_vel - cur_joint_vel[:, None]).reshape(self.num_envs, -1)

        target_obj_transf = indicing(self.demo_data["obj_trajectory"], cur_idx)
        target_obj_transf = target_obj_transf.reshape(nE * nF, 4, 4)
        next_target_state["delta_manip_obj_pos"] = (
            target_obj_transf[:, :3, 3].reshape(nE, nF, -1) - self.states["manip_obj_pos"][:, None]
        ).reshape(nE, -1)

        target_obj_vel = indicing(self.demo_data["obj_velocity"], cur_idx)
        cur_obj_vel = self.states["manip_obj_vel"]
        next_target_state["manip_obj_vel"] = target_obj_vel.reshape(nE, -1)
        next_target_state["delta_manip_obj_vel"] = (target_obj_vel - cur_obj_vel[:, None]).reshape(nE, -1)

        next_target_state["manip_obj_quat"] = rotmat_to_quat(target_obj_transf[:, :3, :3])[:, [1, 2, 3, 0]]
        next_target_state["delta_manip_obj_quat"] = quat_mul(
            self.states["manip_obj_quat"][:, None].repeat(1, nF, 1).reshape(nE * nF, -1),
            quat_conjugate(next_target_state["manip_obj_quat"]),
        ).reshape(nE, -1)
        next_target_state["manip_obj_quat"] = next_target_state["manip_obj_quat"].reshape(nE, -1)

        target_obj_ang_vel = indicing(self.demo_data["obj_angular_velocity"], cur_idx)
        cur_obj_ang_vel = self.states["manip_obj_ang_vel"]
        next_target_state["manip_obj_ang_vel"] = target_obj_ang_vel.reshape(nE, -1)
        next_target_state["delta_manip_obj_ang_vel"] = (target_obj_ang_vel - cur_obj_ang_vel[:, None]).reshape(nE, -1)

        next_target_state["obj_to_joints"] = torch.norm(
            self.states["manip_obj_pos"][:, None] - self.states["joints_state"][:, :, :3], dim=-1
        ).reshape(self.num_envs, -1)

        next_target_state["gt_tips_distance"] = indicing(self.demo_data["tips_distance"], cur_idx).reshape(nE, -1)


        # joint = self.dof_pos.unsqueeze(1)
        # joint_vel = self.dof_vel.unsqueeze(1)
        # eepose = self.eepose.unsqueeze(1)
        # eepose_diff = self.eepose_diff.unsqueeze(1)
        # target_eepose = self.target_eepose_seq[self.target_jt_i.unsqueeze(1), self.target_jt_j.unsqueeze(1) + torch.arange(self.future_frame).to(device=self.device)].reshape(self.num_envs, 1, self.future_frame * 7)
        # obs = torch.cat(
        #     (
        #         joint,
        #         joint_vel, 
        #         eepose,
        #         eepose_diff, 
        #         target_eepose,
        #     ),
        #     dim=-1,
        # )
        # # observations = {"policy": {"self_obs": obs.squeeze()}}
        # observations = {"policy": obs.squeeze()}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        eepose_reward = torch.exp( 10 * self.reward_pose(self.eepose, self.target_eepose, tanh_weight=0.0, pos_norm=2)[0])

        joint_reward = torch.exp(- torch.norm(self.target_jt - self.dof_pos, p=2, dim=-1))

        died, _ = self._get_dones()
        alive_reward = torch.where(died, -100, 0)

        action_diff = self.actions - self.last_actions
        smooth_reward =  torch.exp(- torch.norm(action_diff, p=2, dim=-1))
        
        # Log Reward
        self._log_reward("Episode_Reward/eepose_reward", eepose_reward)
        self._log_reward("Episode_Reward/joint_reward", joint_reward)
        self._log_reward("Episode_Reward/alive_reward", alive_reward)
        self._log_reward("Episode_Reward/smooth_reward", smooth_reward)

        total_reward = (
            100 * 
            eepose_reward *
            joint_reward +
            alive_reward
            # smooth_reward
        )
        self._log_reward("Episode_Reward/total_reward", total_reward)

        return total_reward

    
    def position_command_error(self, des_pos, curr_pos, std=0.1, p=1):
        """Penalize tracking of the position error using L1-norm.

        The function computes the position error between the desired position (from the command) and the
        current position of the asset's body (in world frame). The position error is computed as the L1-norm
        of the difference between the desired and current positions.
        """
        # obtain the desired and current positions
        distance = torch.norm(curr_pos - des_pos, p=p, dim=-1)
        if len(distance.size()) == 2:
            distance = distance.mean(dim=-1)
        error_tanh = 1 - torch.tanh(distance / std)
        return distance, error_tanh

    def orientation_command_error(self, des_quat, curr_quat) -> torch.Tensor:
        """Penalize tracking orientation error using shortest path.

        The function computes the orientation error between the desired orientation (from the command) and the
        current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
        path between the desired and current orientations.
        """
        magnitude = quat_error_magnitude(curr_quat, des_quat)
        if len(magnitude.size()) == 2:
            magnitude = magnitude.mean(dim=-1)
        return magnitude

    def reward_pose(self, des_pose, curr_pose, tanh_weight=0.1, pos_norm=1):
        pos_error, pos_tanh = self.position_command_error(des_pose[..., :3], curr_pose[..., :3], p=pos_norm)
        ori_error = self.orientation_command_error(des_pose[..., 3:7], curr_pose[..., 3:7])
        # self.pos_error = pos_error
        # self.ori_error = ori_error
        return (pos_error * -0.2 + pos_tanh * tanh_weight + ori_error * -0.1), pos_error, pos_tanh, ori_error

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        time_out = self.target_jt_j >= self.traj_len[self.target_jt_i]

        died = - self.eepose_error > 0.15
        return died, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_pos[:, :7] = self.target_jt[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        # this can only be turned on if robot joint is corrected
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self._compute_intermediate_values()
    
    def _log_reward(self, reward_name, reward_data):

        # add "log" key to extras
        if "log" not in self.extras:
            self.extras["log"] = {}
        self.extras["log"][reward_name] = reward_data
    
    def step(self, action: torch.Tensor):
        action = action.to(self.device)
        # add action noise
        if self.cfg.action_noise_model:
            action = self._action_noise_model.apply(action)

        # process actions
        self._pre_physics_step(action)

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        # is_rendering = True

        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self._apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # Visualize Markers
        self.ee_marker.visualize(self.eepose[:, 0:3] + self.scene.env_origins, self.eepose[:, 3:7])
        self.goal_marker.visualize(self.target_eepose[:, 0:3] + self.scene.env_origins, self.target_eepose[:, 3:7])

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)

        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        # -- update target joint with reset
        self.update_target(reset_env_ids)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)

        # post-step: step interval event
        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)

        # update observations
        self.obs_buf = self._get_observations()

        # add observation noise
        # note: we apply no noise to the state space (since it is used for critic networks)
        if self.cfg.observation_noise_model:
            self.obs_buf["policy"] = self._observation_noise_model.apply(self.obs_buf["policy"])

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_pole_pos: float,
    rew_scale_cart_vel: float,
    rew_scale_pole_vel: float,
    pole_pos: torch.Tensor,
    pole_vel: torch.Tensor,
    cart_pos: torch.Tensor,
    cart_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    rew_pole_pos = rew_scale_pole_pos * torch.sum(torch.square(pole_pos).unsqueeze(dim=1), dim=-1)
    rew_cart_vel = rew_scale_cart_vel * torch.sum(torch.abs(cart_vel).unsqueeze(dim=1), dim=-1)
    rew_pole_vel = rew_scale_pole_vel * torch.sum(torch.abs(pole_vel).unsqueeze(dim=1), dim=-1)
    total_reward = rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel
    return total_reward