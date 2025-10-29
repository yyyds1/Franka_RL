# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Go2 quadruped locomotion environment using Direct RL workflow with Command Helper.

这是go2_env.py的改进版本，使用了DirectCommandHelper来实现自动命令管理。
"""

from __future__ import annotations

import torch
import math
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, quat_apply, sample_uniform

from .go2_env_cfg import Go2EnvCfg
from Franka_RL.utils.command_helper import DirectCommandHelper, CommandConfig, CommandPresets


class Go2Env2(DirectRLEnv):
    """
    Go2 quadruped locomotion environment with automatic command management.
    
    相比原版go2_env.py的改进：
    - ✅ 使用DirectCommandHelper自动管理命令
    - ✅ 支持中途自动重采样（每8-12秒）
    - ✅ 支持静止环境（10%概率）
    - ✅ 可选的heading控制
    - ✅ 代码更简洁，逻辑更清晰
    """
    
    cfg: Go2EnvCfg
    
    def __init__(self, cfg: Go2EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        self.action_scale = self.cfg.action_scale
        self.dt = self.cfg.decimation * self.cfg.sim.dt
        self.default_joint_pos = None
        
        # ✅ 使用DirectCommandHelper替代手动实现
        self._init_command_helper()
        
        # 奖励权重配置
        self.reward_weights = {
            "track_lin_vel_xy_exp": 1.5,
            "track_ang_vel_z_exp": 1.5,
            "flat_orientation": -2.5,
            "base_height_reward": -5.0,
            "feet_air_time": 0.2,
            "feet_contact_forces": -0.01,
            "hip_deviation": -0.4,
            "joint_deviation": -0.04,
            "joint_power": -2e-5,
            "dof_acc_l2": -2.5e-7,
            "action_rate_l2": -0.02,
            "action_smoothness": -0.02,
            "termination_penalty": -2.0,
        }
        
        self.obs_scales = self.cfg.obs_scales
        
        self._joint_dof_idx = None
        self._base_idx = None
        self._feet_indices = None
        
        self.last_actions = torch.zeros(self.num_envs, 12, dtype=torch.float32, device=self.device)
        self.last_dof_vel = torch.zeros(self.num_envs, 12, dtype=torch.float32, device=self.device)
        self._latest_policy_obs_seq: torch.Tensor | None = None
        
        self.raw_actions = torch.zeros(self.num_envs, 12, dtype=torch.float32, device=self.device)
        self.last_raw_actions = torch.zeros(self.num_envs, 12, dtype=torch.float32, device=self.device)
        
    def _init_command_helper(self):
        """
        初始化命令辅助系统。
        
        这里展示了三种配置方式，根据需要选择：
        """
        # 方式1：使用预定义配置（推荐）
        # command_cfg = CommandPresets.basic()       # 最简单：无高级功能
        command_cfg = CommandPresets.standard()    # 标准：10%静止环境
        # command_cfg = CommandPresets.advanced()    # 高级：heading+静止
        # command_cfg = CommandPresets.training()    # 训练：频繁重采样
        
        # 方式2：手动配置（自定义）
        # command_cfg = CommandConfig(
        #     lin_vel_x_range=(-2.0, 2.0),
        #     lin_vel_y_range=(-2.0, 2.0),
        #     ang_vel_z_range=(-1.5, 1.5),
        #     resampling_time_range=(8.0, 12.0),
        #     enable_heading_control=False,  # 不启用heading
        #     enable_standing_envs=True,     # 启用静止环境
        #     rel_standing_envs=0.1,         # 10%静止
        #     enable_metrics=False,          # 不追踪误差
        # )
        
        # 方式3：从配置文件读取（与原go2_env兼容）
        # command_cfg = CommandConfig(
        #     lin_vel_x_range=self.cfg.commands_cfg["lin_vel_x_range"],
        #     lin_vel_y_range=self.cfg.commands_cfg["lin_vel_y_range"],
        #     ang_vel_z_range=self.cfg.commands_cfg["ang_vel_z_range"],
        #     heading_range=self.cfg.commands_cfg["heading_range"],
        #     enable_standing_envs=True,
        #     rel_standing_envs=0.1,
        # )
        
        # 创建Helper实例
        self.command_helper = DirectCommandHelper(
            num_envs=self.num_envs,
            device=self.device,
            cfg=command_cfg,
        )
    
    def _setup_scene(self):
        """Set up the simulation scene."""
        self.robot = Articulation(self.cfg.robot)
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self.scene.clone_environments(copy_from_source=False)
        
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self._joint_dof_idx = None
        self._base_idx = None
        self._feet_indices = None
    
    def _initialize_indices(self):
        """Initialize robot joint/body indices and default joint positions."""
        if self._joint_dof_idx is not None:
            return

        self._joint_dof_idx, _ = self.robot.find_joints(".*")
        self._base_idx, _ = self.robot.find_bodies("base")
        self._base_idx = self._base_idx[0]

        fl_idx, _ = self.robot.find_bodies("FL_foot")
        fr_idx, _ = self.robot.find_bodies("FR_foot")
        rl_idx, _ = self.robot.find_bodies("RL_foot")
        rr_idx, _ = self.robot.find_bodies("RR_foot")

        self._feet_indices = [fl_idx[0], fr_idx[0], rl_idx[0], rr_idx[0]]

        if self.default_joint_pos is None:
            from Franka_RL.robots import QuadrupedRobotFactory
            go2_robot = QuadrupedRobotFactory.create_robot("unitree_go2")
            joint_pos_dict = go2_robot.init_state.joint_pos
            joint_pos_list = [joint_pos_dict[name] for name in go2_robot.dof_names]
            self.default_joint_pos = torch.tensor(joint_pos_list, dtype=torch.float32, device=self.device)
    
    def step(self, action: torch.Tensor):
        """
        Execute one time-step of the environment's dynamics.
        
        ✅ 关键改进：在这里自动更新命令系统
        """
        # ✅ 1. 更新命令系统（自动重采样）
        # 如果配置了heading控制，需要传入robot对象
        if self.command_helper.cfg.enable_heading_control:
            self.command_helper.update(self.step_dt, self.robot)
        else:
            self.command_helper.update(self.step_dt)
        
        # 2. 继续原有的step逻辑
        return super().step(action)
    
    def _pre_physics_step(self, actions: torch.Tensor):
        """Pre-process actions before the physics step."""
        if self._joint_dof_idx is None:
            self._initialize_indices()

        self.last_raw_actions = self.raw_actions.clone()
        self.raw_actions = actions.clone()
        
        self.last_actions = self.actions.clone() if hasattr(self, 'actions') else self.dof_pos
        
        joint_pos_target = self.dof_pos + actions * self.action_scale
        self.actions = joint_pos_target

    def _apply_action(self):
        """Apply actions to the robot."""
        self.robot.set_joint_position_target(self.actions, joint_ids=self._joint_dof_idx)
        
    def _compute_intermediate_values(self):
        """Compute intermediate values for rewards and observations."""
        if self._joint_dof_idx is None:
            self._initialize_indices()

        self.dof_pos = self.robot.data.joint_pos[:, :12]
        self.dof_vel = self.robot.data.joint_vel[:, :12]

        base_state = self.robot.data.root_state_w
        self.base_pos = base_state[:, :3]
        self.base_quat = base_state[:, 3:7]
        self.base_lin_vel = base_state[:, 7:10]
        self.base_ang_vel = base_state[:, 10:13]

        gravity_vec = torch.tensor([[0.0, 0.0, -1.0]], device=self.device).repeat(self.num_envs, 1)
        self.projected_gravity = quat_apply_inverse(self.base_quat, gravity_vec)
        self.base_lin_vel_local = quat_apply_inverse(self.base_quat, self.base_lin_vel)
        self.base_ang_vel_local = quat_apply_inverse(self.base_quat, self.base_ang_vel)

        forward_vec = quat_apply(self.base_quat, torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1))
        self.heading = torch.atan2(forward_vec[:, 1], forward_vec[:, 0])
        
        # ✅ 从Helper获取命令
        commands_3d = self.command_helper.get_commands()  # [num_envs, 3]
        self.commands = self.command_helper.get_commands_with_heading()  # [num_envs, 4]
        
        self.heading_error = torch.abs(self.heading - self.commands[:, 3])
        self.lin_vel_error = self.base_lin_vel_local[:, :2] - commands_3d[:, :2]
        self.ang_vel_error = self.base_ang_vel_local[:, 2] - commands_3d[:, 2]

        body_states = self.robot.data.body_state_w
        feet_pos_list = []
        for foot_idx in self._feet_indices:
            feet_pos_list.append(body_states[:, foot_idx, :3])
        self.feet_pos = torch.stack(feet_pos_list, dim=1)
        self.feet_contact = (self.feet_pos[:, :, 2] < 0.02).float()

        self.base_height = self.base_pos[:, 2]
        self.dof_acc = (self.dof_vel - self.last_dof_vel) / self.dt
        self.last_dof_vel = self.dof_vel.clone()
        
    def _get_observations(self) -> dict:
        """Compute observations."""
        self._compute_intermediate_values()

        obs_list = []
        base_lin_vel_obs = self.base_lin_vel_local * self.obs_scales["lin_vel"]
        obs_list.append(base_lin_vel_obs)

        base_ang_vel_obs = self.base_ang_vel_local * self.obs_scales["ang_vel"]
        obs_list.append(base_ang_vel_obs)

        commands_obs = self.commands[:, :3] * torch.tensor(self.obs_scales["commands"], device=self.device)
        obs_list.append(commands_obs)

        dof_pos_normalized = (self.dof_pos - self.default_joint_pos.unsqueeze(0)) * self.obs_scales["dof_pos"]
        obs_list.append(dof_pos_normalized)

        dof_vel_scaled = self.dof_vel * self.obs_scales["dof_vel"]
        obs_list.append(dof_vel_scaled)

        obs_list.append(self.last_actions)
        
        obs = torch.cat(obs_list, dim=-1)
        self._latest_policy_obs_seq = obs.unsqueeze(1)

        observations = {"policy": obs}
        self.extras["observations"] = observations
        return observations
        
    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards (与原版完全相同)."""
        self._compute_intermediate_values()

        # 速度跟踪奖励
        lin_vel_error_squared = torch.sum(torch.square(self.lin_vel_error), dim=1)
        lin_vel_reward = torch.exp(-lin_vel_error_squared / 0.5) * self.reward_weights["track_lin_vel_xy_exp"]
        
        ang_vel_error_squared = torch.square(self.ang_vel_error)
        ang_vel_reward = torch.exp(-ang_vel_error_squared / 0.5) * self.reward_weights["track_ang_vel_z_exp"]

        # 姿态稳定性
        gravity_target = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1)
        gravity_error = self.projected_gravity - gravity_target
        flat_orientation_reward = torch.sum(torch.square(gravity_error), dim=1) * self.reward_weights["flat_orientation"]

        target_height = 0.32
        height_error_squared = torch.square(self.base_height - target_height)
        height_reward = height_error_squared * self.reward_weights["base_height_reward"]

        # 脚部接触
        cmd_norm = torch.norm(self.commands[:, :2], dim=1)
        is_moving = (cmd_norm > 0.1).float()
        
        feet_air_time = self._contact_sensor.data.last_air_time[:, self._feet_indices]
        first_contact = self._contact_sensor.compute_first_contact(self.dt)[:, self._feet_indices]
        
        threshold = 0.5
        feet_air_time_reward = torch.sum((feet_air_time - threshold) * first_contact, dim=1)
        feet_air_time_reward *= is_moving * self.reward_weights["feet_air_time"]
        
        contact_reward = self.feet_contact.sum(dim=1) * self.reward_weights["feet_contact_forces"]

        # 关节偏差
        hip_indices = [0, 1, 2, 3]
        hip_deviation = torch.sum(torch.abs(
            self.dof_pos[:, hip_indices] - self.default_joint_pos[hip_indices].unsqueeze(0)
        ), dim=1) * self.reward_weights["hip_deviation"]
        
        thigh_calf_indices = [4, 5, 6, 7, 8, 9, 10, 11]
        joint_deviation = torch.sum(torch.abs(
            self.dof_pos[:, thigh_calf_indices] - self.default_joint_pos[thigh_calf_indices].unsqueeze(0)
        ), dim=1) * self.reward_weights["joint_deviation"]

        # 功率惩罚
        applied_torques = self.robot.data.applied_torque[:, :12]
        power = applied_torques * self.dof_vel
        power_reward = torch.sum(torch.abs(power), dim=1) * self.reward_weights["joint_power"]

        # 加速度惩罚
        dof_acc_reward = torch.sum(torch.square(self.dof_acc), dim=1) * self.reward_weights["dof_acc_l2"]

        # 动作平滑性
        action_diff = self.raw_actions - self.last_raw_actions
        action_rate_reward = torch.sum(torch.square(action_diff), dim=1) * self.reward_weights["action_rate_l2"]
        action_smoothness_reward = torch.sum(torch.abs(action_diff), dim=1) * self.reward_weights["action_smoothness"]

        # 终止惩罚
        died, _ = self._get_dones()
        termination_reward = died.float() * self.reward_weights["termination_penalty"]

        # 总奖励
        total_reward = (
            lin_vel_reward + 
            ang_vel_reward + 
            flat_orientation_reward + 
            height_reward + 
            feet_air_time_reward + 
            contact_reward + 
            hip_deviation + 
            joint_deviation + 
            power_reward + 
            dof_acc_reward + 
            action_rate_reward +
            action_smoothness_reward +
            termination_reward
        )

        # 记录奖励分项
        self.extras["log"] = {
            "rewards/lin_vel": lin_vel_reward.mean(),
            "rewards/ang_vel": ang_vel_reward.mean(),
            "rewards/flat_orientation": flat_orientation_reward.mean(),
            "rewards/height": height_reward.mean(),
            "rewards/feet_air_time": feet_air_time_reward.mean(),
            "rewards/contact": contact_reward.mean(),
            "rewards/hip_deviation": hip_deviation.mean(),
            "rewards/joint_deviation": joint_deviation.mean(),
            "rewards/power": power_reward.mean(),
            "rewards/dof_acc": dof_acc_reward.mean(),
            "rewards/action_rate": action_rate_reward.mean(),
            "rewards/action_smoothness": action_smoothness_reward.mean(),
            "rewards/termination": termination_reward.mean(),
            "rewards/total": total_reward.mean(),
        }
        
        # ✅ 可选：记录命令误差（如果启用了metrics）
        if self.command_helper.cfg.enable_metrics:
            metrics = self.command_helper.get_metrics()
            self.extras["log"]["commands/error_vel_xy"] = metrics["error_vel_xy"].mean()
            self.extras["log"]["commands/error_vel_yaw"] = metrics["error_vel_yaw"].mean()

        return total_reward
        
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute termination conditions."""
        if not hasattr(self, '_intermediate_computed_this_step'):
            self._compute_intermediate_values()
        tilted = torch.abs(self.projected_gravity[:, 2]) < torch.cos(
            torch.tensor(self.cfg.termination_cfg["roll_pitch_max"], device=self.device)
        )

        base_contact = self._check_illegal_contact_by_name(["base"], threshold=1.0)
        hip_contact = self._check_illegal_contact_by_name(["FL_hip", "FR_hip", "RL_hip", "RR_hip"], threshold=1.0)
        calf_contact = self._check_illegal_contact_by_name(["FL_calf", "FR_calf", "RL_calf", "RR_calf"], threshold=1.0)

        terminated = tilted | base_contact | hip_contact | calf_contact
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return terminated, time_out
    
    def _check_illegal_contact(self, body_indices: list, threshold: float = 1.0) -> torch.Tensor:
        """检查指定身体部位是否有非法接触。"""
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        selected_forces = net_contact_forces[:, :, body_indices, :]
        contact_force_magnitude = torch.norm(selected_forces, dim=-1)
        max_contact_force = torch.max(contact_force_magnitude, dim=1)[0]
        illegal_contact = torch.any(max_contact_force > threshold, dim=1)
        
        return illegal_contact
    
    def _check_illegal_contact_by_name(self, body_names: list[str], threshold: float = 1.0) -> torch.Tensor:
        """通过body名称检查非法接触。"""
        body_indices = []
        for name in body_names:
            try:
                idx = self._contact_sensor.body_names.index(name)
                body_indices.append(idx)
            except ValueError:
                pass
        
        if not body_indices:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        return self._check_illegal_contact(body_indices, threshold)
    
    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset specified environments."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES

        self.robot.reset(env_ids)

        pos_range = self.cfg.init_state_cfg["pos_range"]
        init_pos = self.robot.data.default_root_state[env_ids, :3].clone()
        init_pos[:, 0] += sample_uniform(pos_range["x"][0], pos_range["x"][1], (len(env_ids),), self.device)
        init_pos[:, 1] += sample_uniform(pos_range["y"][0], pos_range["y"][1], (len(env_ids),), self.device)
        init_pos[:, 2] += pos_range["z"][0]

        rot_range = self.cfg.init_state_cfg["rot_range"]
        yaw = sample_uniform(rot_range["yaw"][0], rot_range["yaw"][1], (len(env_ids),), self.device)
        init_quat = torch.zeros(len(env_ids), 4, device=self.device)
        init_quat[:, 0] = torch.cos(yaw / 2)
        init_quat[:, 3] = torch.sin(yaw / 2)

        vel_range = self.cfg.init_state_cfg["vel_range"]
        init_vel = torch.zeros(len(env_ids), 6, device=self.device)
        init_vel[:, 0] = sample_uniform(vel_range["x"][0], vel_range["x"][1], (len(env_ids),), self.device)
        init_vel[:, 1] = sample_uniform(vel_range["y"][0], vel_range["y"][1], (len(env_ids),), self.device)

        if self.default_joint_pos is None:
            from Franka_RL.robots import QuadrupedRobotFactory
            go2_robot = QuadrupedRobotFactory.create_robot("unitree_go2")
            joint_pos_dict = go2_robot.init_state.joint_pos
            joint_pos_list = [joint_pos_dict[name] for name in go2_robot.dof_names]
            self.default_joint_pos = torch.tensor(joint_pos_list, dtype=torch.float32, device=self.device)

        joint_pos = self.default_joint_pos.unsqueeze(0).repeat(len(env_ids), 1)
        joint_pos += sample_uniform(-0.1, 0.1, joint_pos.shape, self.device)
        joint_vel = torch.zeros(len(env_ids), 12, device=self.device)

        root_state = torch.cat([init_pos, init_quat, init_vel], dim=1)
        self.robot.write_root_state_to_sim(root_state, env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # ✅ 使用Helper重置命令（替代手动_resample_commands）
        self.command_helper.reset(env_ids)

        self.last_actions[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.raw_actions[env_ids] = 0.0
        self.last_raw_actions[env_ids] = 0.0
