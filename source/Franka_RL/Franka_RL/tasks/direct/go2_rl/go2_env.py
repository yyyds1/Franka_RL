# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 quadruped locomotion environment using Direct RL workflow."""

from __future__ import annotations

import torch
import math
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.math import quat_apply_inverse, quat_apply, sample_uniform

from .go2_env_cfg import Go2EnvCfg


class Go2Env(DirectRLEnv):
    """Go2 quadruped locomotion environment using the Direct RL workflow.

    This environment implements:
    - Velocity tracking for locomotion
    - Terrain adaptation
    - Energy-efficient gaits
    - Contact-based feedback

    Key differences from the manager-based approach:
    - All logic is implemented in this class
    - Direct access to robot state
    - Custom reward computation
    - Flexible observation composition
    """
    
    cfg: Go2EnvCfg
    
    def __init__(self, cfg: Go2EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
    # ============================================================
    # 1. Read parameters from configuration
    # ============================================================
        self.action_scale = self.cfg.action_scale
        self.dt = self.cfg.decimation * self.cfg.sim.dt
        
    # Delayed initialization (set in _initialize_indices)
        self.default_joint_pos = None
        
    # ============================================================
    # 2. Initialize command generator
    # ============================================================
        self._init_commands()
        
    # ============================================================
    # 3. Initialize reward weights and observation scales
    # ============================================================
        self.reward_weights = {
            "track_lin_vel_xy_exp": self.cfg.lin_vel_reward_scale,
            "track_ang_vel_z_exp": self.cfg.ang_vel_reward_scale,
            "flat_orientation": self.cfg.flat_orientation_reward_scale,
            "base_height_reward": self.cfg.base_height_reward_scale if hasattr(self.cfg, 'base_height_reward_scale') else -5.0,
            "feet_air_time": self.cfg.feet_air_time_reward_scale,
            "feet_contact_forces": self.cfg.undersired_contact_reward_scale,
            "dof_torques_l2": self.cfg.joint_torque_reward_scale,
            "dof_acc_l2": self.cfg.joint_acc_reward_scale,
            "action_rate_l2": self.cfg.action_rate_reward_scale,
            # "lin_vel_z_l2": -2.0,  
            # "ang_vel_xy_l2": -0.05,  
            "termination_penalty": -2.0,
        }
        
        self.obs_scales = self.cfg.obs_scales
        
    # ============================================================
    # 4. Joint and body indices (initialized after _setup_scene)
    # These will be initialized in _setup_scene
        self._joint_dof_idx = None
        self._base_idx = None
        self._feet_indices = None
        
    # ============================================================
    # 5. Initialize history buffers (used by Transformer)
    # ============================================================
        self.last_actions = torch.zeros(
            self.num_envs, 12, dtype=torch.float32, device=self.device
        )
        self.last_dof_vel = torch.zeros(
            self.num_envs, 12, dtype=torch.float32, device=self.device
        )
        self._latest_policy_obs_seq: torch.Tensor | None = None
        
    def _init_commands(self):
        """Initialize the velocity command generator."""
        # Command buffer: [lin_vel_x, lin_vel_y, ang_vel_z, heading]
        self.commands = torch.zeros(
            self.num_envs, 4, dtype=torch.float32, device=self.device
        )

        # Command ranges
        self.command_ranges = {
            "lin_vel_x": self.cfg.commands_cfg["lin_vel_x_range"],
            "lin_vel_y": self.cfg.commands_cfg["lin_vel_y_range"],
            "ang_vel_z": self.cfg.commands_cfg["ang_vel_z_range"],
            "heading": self.cfg.commands_cfg["heading_range"],
        }
        
    def _setup_scene(self):
        """Set up the simulation scene."""
        # ============================================================
        # 1. Create robot
        # ============================================================
        self.robot = Articulation(self.cfg.robot)
        
    # ============================================================
    # 2. Add terrain
    # ============================================================
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        
        # ============================================================
        # 4. Add robot to the scene
        # ============================================================
        self.scene.articulations["robot"] = self.robot

        # ============================================================
        # 5. Clone environments - must be called after adding all assets
        # ============================================================
        self.scene.clone_environments(copy_from_source=False)
        
        # ============================================================
        # 6. Filter collisions (needed for CPU simulation)
        # ============================================================
        if self.device == "cpu":
            self.scene.filter_collisions(
                global_prim_paths=[self.cfg.terrain.prim_path]
            )
        
        # ============================================================
        # 7. Add lighting
        # ============================================================
        light_cfg = sim_utils.DomeLightCfg(
            intensity=2000.0, color=(0.75, 0.75, 0.75)
        )
        light_cfg.func("/World/Light", light_cfg)

        # ============================================================
        # Note: index initialization is postponed until _post_physics_step
        # because the scene is not fully initialized at this point.
        # These will be initialized during the first reset.
        # ============================================================
        self._joint_dof_idx = None
        self._base_idx = None
        self._feet_indices = None

        # ============================================================
        # 7. Create visualization markers (optional, disabled for now)
        # ============================================================
        # TODO: enable visualization markers when needed
        # if sim_utils.SimulationContext.instance().get_render_mode() != "headless":
        #     marker_cfg = FRAME_MARKER_CFG.copy()
        #     marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        #     self.base_marker = VisualizationMarkers(
        #         marker_cfg.replace(prim_path="/Visuals/base_marker")
        #     )
    
    def _initialize_indices(self):
        """Initialize robot joint/body indices and the default joint positions.

        This method is called after the scene is fully initialized (during the
        first reset).
        """
        if self._joint_dof_idx is not None:
            return  # already initialized

        # Joint indices (all 12 joints)
        self._joint_dof_idx, _ = self.robot.find_joints(".*")

        # Base index
        self._base_idx, _ = self.robot.find_bodies("base")
        self._base_idx = self._base_idx[0]

        # Foot (end-effector) indices
        fl_idx, _ = self.robot.find_bodies("FL_foot")
        fr_idx, _ = self.robot.find_bodies("FR_foot")
        rl_idx, _ = self.robot.find_bodies("RL_foot")
        rr_idx, _ = self.robot.find_bodies("RR_foot")

        self._feet_indices = [
            fl_idx[0],
            fr_idx[0],
            rl_idx[0],
            rr_idx[0],
        ]

        # Get default joint positions from the robot factory
        if self.default_joint_pos is None:
            from Franka_RL.robots import QuadrupedRobotFactory
            go2_robot = QuadrupedRobotFactory.create_robot("unitree_go2")

            # joint_pos is a dict; convert to a list ordered by dof_names
            joint_pos_dict = go2_robot.init_state.joint_pos
            joint_pos_list = [joint_pos_dict[name] for name in go2_robot.dof_names]

            self.default_joint_pos = torch.tensor(
                joint_pos_list,
                dtype=torch.float32,
                device=self.device,
            )

        print("\nIndices initialized:")
        print(f"   - Joint DOFs: {len(self._joint_dof_idx)} joints")
        print(f"   - Base index: {self._base_idx}")
        print(f"   - Feet indices: {self._feet_indices}")
        print(f"   - Default joint pos: {self.default_joint_pos}")
        print(f"   - Num envs: {self.num_envs}")
        print(f"   - Device: {self.device}")
            
    def _pre_physics_step(self, actions: torch.Tensor):
        """Pre-process actions - called before the physics step.

        Design decisions:
        1. Use PD position control instead of torque control
        2. Actions are joint position deltas
        3. Clamp targets within joint limits
        """
        # Ensure indices are initialized
        if self._joint_dof_idx is None:
            self._initialize_indices()

        # Save previous actions
        self.last_actions = self.actions.clone() if hasattr(self, 'actions') else actions

        # Action processing: delta control + joint clamping
        # actions: [num_envs, 12], roughly in [-1, 1]

        # Option 1: delta control (recommended - more stable)
        joint_pos_target = (
            self.dof_pos +                    # current position
            actions * self.action_scale       # delta (scaled)
        )

        # Option 2: absolute position control
        # joint_pos_target = (
        #     self.default_joint_pos.unsqueeze(0) +  # default pose
        #     actions * self.action_scale            # offset
        # )

        # Clamp within joint limits (uncomment to enable clamping)
        # self.actions = torch.clamp(
        #     joint_pos_target,
        #     self.joint_limits[:, 0],  # lower limit
        #     self.joint_limits[:, 1],  # upper limit
        # )
        self.actions = joint_pos_target

    def _apply_action(self):
        """Apply actions to the robot."""
        # Set joint position targets (PD controller computes torques)
        self.robot.set_joint_position_target(
            self.actions,
            joint_ids=self._joint_dof_idx,
        )
        
    def _compute_intermediate_values(self):
        """Compute intermediate values - the core of the Direct architecture.

        All required state quantities are computed and cached here to avoid
        duplicated work in reward/observation computations.
        """
        # Ensure indices are initialized
        if self._joint_dof_idx is None:
            self._initialize_indices()

        # ============================================================
        # 1. Basic states
        # ============================================================
        # Joint states
        self.dof_pos = self.robot.data.joint_pos[:, :12]  # [num_envs, 12]
        self.dof_vel = self.robot.data.joint_vel[:, :12]  # [num_envs, 12]

        # Base state (world frame)
        base_state = self.robot.data.root_state_w
        self.base_pos = base_state[:, :3]       # [num_envs, 3]
        self.base_quat = base_state[:, 3:7]     # [num_envs, 4] (w, x, y, z)
        self.base_lin_vel = base_state[:, 7:10] # [num_envs, 3]
        self.base_ang_vel = base_state[:, 10:13] # [num_envs, 3]

        # ============================================================
        # 2. Coordinate transforms - world -> robot local frame
        # ============================================================
        # Important: RL usually operates in the robot local frame

        # Project gravity into the robot frame
        gravity_vec = torch.tensor([[0.0, 0.0, -1.0]], device=self.device).repeat(self.num_envs, 1)  # [num_envs, 3]
        self.projected_gravity = quat_apply_inverse(
            self.base_quat,  # [num_envs, 4]
            gravity_vec      # [num_envs, 3]
        )  # [num_envs, 3]

        # Linear velocity in local frame
        self.base_lin_vel_local = quat_apply_inverse(
            self.base_quat, self.base_lin_vel
        )

        # Angular velocity in local frame
        self.base_ang_vel_local = quat_apply_inverse(
            self.base_quat, self.base_ang_vel
        )

        # ============================================================
        # 3. Compute heading and errors
        # ============================================================
        # Current heading (yaw)
        forward_vec = quat_apply(self.base_quat, torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1))
        self.heading = torch.atan2(forward_vec[:, 1], forward_vec[:, 0])

        # Heading error
        self.heading_error = torch.abs(self.heading - self.commands[:, 3])

        # Velocity errors
        self.lin_vel_error = self.base_lin_vel_local[:, :2] - self.commands[:, :2]
        self.ang_vel_error = self.base_ang_vel_local[:, 2] - self.commands[:, 2]

        # ============================================================
        # 4. Contact information
        # ============================================================
        # Foot contact forces (should come from contact sensors). Here we
        # simplify and approximate contacts using foot positions.
        body_states = self.robot.data.body_state_w  # [num_envs, num_bodies, 13]

        # Correct indexing: fetch each foot body separately
        feet_pos_list = []
        for foot_idx in self._feet_indices:
            feet_pos_list.append(body_states[:, foot_idx, :3])  # [num_envs, 3]
        self.feet_pos = torch.stack(feet_pos_list, dim=1)  # [num_envs, 4, 3]

        # Simple contact detection: foot height < threshold
        self.feet_contact = (self.feet_pos[:, :, 2] < 0.02).float()  # [num_envs, 4]

        # ============================================================
        # 5. Other useful quantities
        # ============================================================
        # Base height
        self.base_height = self.base_pos[:, 2]

        # Joint accelerations (numerical differentiation)
        self.dof_acc = (self.dof_vel - self.last_dof_vel) / self.dt
        self.last_dof_vel = self.dof_vel.clone()

        # Action rate of change
        self.action_rate = (self.actions - self.last_actions) / self.dt
        
    def _get_observations(self) -> dict:
        """Compute observations - the hand-written core logic.

        Observation design principles:
        1. Include task-relevant information
        2. Preserve Markov property
        3. Apply appropriate normalization/scaling
        4. Consider sequence requirements for Transformers
        """
        # Ensure intermediate values are up to date
        self._compute_intermediate_values()

        # ============================================================
        # Base observations
        # ============================================================
        obs_list = []

        # 1. Base state (local frame)
        base_lin_vel_obs = self.base_lin_vel_local * self.obs_scales["lin_vel"]  # [num_envs, 3]
        obs_list.append(base_lin_vel_obs)  # 3

        base_ang_vel_obs = self.base_ang_vel_local * self.obs_scales["ang_vel"]  # [num_envs, 3]
        obs_list.append(base_ang_vel_obs)  # 3

        #projected_grav_obs = self.projected_gravity * self.obs_scales["gravity"]  # [num_envs, 3]
        #obs_list.append(projected_grav_obs)  # 3

        # 2. Commands
        commands_obs = self.commands[:, :3] * torch.tensor(
            self.obs_scales["commands"], device=self.device
        )  # [num_envs, 3]
        obs_list.append(commands_obs)  # 3

        # 3. Joint states (relative to default positions)
        dof_pos_normalized = (self.dof_pos - self.default_joint_pos.unsqueeze(0)) * self.obs_scales["dof_pos"]  # [num_envs, 12]
        obs_list.append(dof_pos_normalized)  # 12

        dof_vel_scaled = self.dof_vel * self.obs_scales["dof_vel"]  # [num_envs, 12]
        obs_list.append(dof_vel_scaled)  # 12

        # 4. Action history
        obs_list.append(self.last_actions)  # [num_envs, 12]
        
        # ============================================================
        # Total observation dimension: 3+3+3+3+12+12+12 = 48
        # ============================================================
        # Concatenate all observations
        obs = torch.cat(obs_list, dim=-1)  # [num_envs, 48]

        # If using a Transformer, expand into a sequence dimension
        self._latest_policy_obs_seq = obs.unsqueeze(1)  # [num_envs, 1, 48]

        observations = {"policy": obs}
        self.extras["observations"] = observations
        return observations
        
    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards - the hand-crafted core logic.

        Reward design principles:
        1. Sparse vs dense: dense rewards are used here
        2. Reward shaping: use smooth functions like exp/tanh
        3. Balance multiple objectives via weights
        """
        # Ensure intermediate values are up to date
        self._compute_intermediate_values()

        # ============================================================
        # 1. Task rewards - velocity tracking
        # ============================================================
        # Linear velocity tracking (xy plane)
        lin_vel_error_norm = torch.norm(self.lin_vel_error, dim=1)
        lin_vel_reward = torch.exp(-lin_vel_error_norm / 0.25) * self.reward_weights["track_lin_vel_xy_exp"]

        # Angular velocity tracking (z)
        ang_vel_reward = torch.exp(-torch.abs(self.ang_vel_error) / 0.25) * self.reward_weights["track_ang_vel_z_exp"]

        # ============================================================
        # 2. Posture rewards - stay upright
        # ============================================================
        # Upright posture (gravity projection should point to -z)
        up_reward = torch.square(self.projected_gravity[:, 2] + 1.0) * self.reward_weights["flat_orientation"]

        # Base height
        target_height = 0.32  # target standing height for Go2
        height_error = torch.abs(self.base_height - target_height)
        height_reward = torch.exp(-height_error / 0.1) * self.reward_weights["base_height_reward"]

        # ============================================================
        # 3. Gait rewards
        # ============================================================
        # Foot air-time reward (encourage lifting legs) - simplified
        feet_air_time_reward = (1.0 - self.feet_contact).sum(dim=1) * self.reward_weights["feet_air_time"]

        # Contact penalty (encourage soft contacts) - simplified
        contact_reward = -self.feet_contact.sum(dim=1) * self.reward_weights["feet_contact_forces"]

        # ============================================================
        # 4. Energy penalties
        # ============================================================
        # Approximate torques with velocity * action
        torques_approx = self.dof_vel * self.actions
        torques_reward = -torch.sum(torch.square(torques_approx), dim=1) * self.reward_weights["dof_torques_l2"]

        # Joint acceleration L2
        dof_acc_reward = -torch.sum(torch.square(self.dof_acc), dim=1) * self.reward_weights["dof_acc_l2"]

        # Action rate L2 (smoothness)
        action_rate_reward = -torch.sum(torch.square(self.action_rate), dim=1) * self.reward_weights["action_rate_l2"]

        # ============================================================
        # 5. 额外奖励/惩罚项
        #  joint_deviation_l1
        joint_deviation_l1 = -torch.sum(torch.abs(self.dof_pos - self.default_joint_pos.unsqueeze(0)), dim=1) * 0.04  # go2_low_base_cfg: weight=-0.04
        # hip_deviation
        hip_deviation = -torch.sum(torch.abs(self.dof_pos[:, [0, 3, 6, 9]] - self.default_joint_pos.unsqueeze(0)[:, :4]), dim=1) * 0.4  # go2_low_base_cfg: weight=-0.4

        # 注释掉go2_env原有的lin_vel_z_reward和ang_vel_xy_reward（go2_low_base_cfg没有）
        # lin_vel_z_reward = -torch.square(self.base_lin_vel_local[:, 2]) * self.reward_weights["lin_vel_z_l2"]
        # ang_vel_xy_reward = -torch.sum(torch.square(self.base_ang_vel_local[:, :2]), dim=1) * self.reward_weights["ang_vel_xy_l2"]

        # 6. Termination penalty
        died, _ = self._get_dones()
        termination_reward = died.float() * self.reward_weights["termination_penalty"]

        # ============================================================
        # Total reward
        # ============================================================
        total_reward = (
            lin_vel_reward
            + ang_vel_reward
            + up_reward
            + height_reward
            + feet_air_time_reward
            + contact_reward
            + torques_reward
            + dof_acc_reward
            + action_rate_reward
            + joint_deviation_l1
            + hip_deviation
            + termination_reward
        )

        # Log individual reward terms for debugging and analysis
        self.extras["log"] = {
            "rewards/lin_vel": lin_vel_reward.mean(),
            "rewards/ang_vel": ang_vel_reward.mean(),
            "rewards/up": up_reward.mean(),
            "rewards/height": height_reward.mean(),
            "rewards/feet_air_time": feet_air_time_reward.mean(),
            "rewards/total": total_reward.mean(),
        }

        return total_reward
        
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute termination conditions.

        Returns:
            terminated: task failure (death)
            truncated: episode timeout (max steps reached)
        """
        # ============================================================
        # Termination checks
        # ============================================================
        # 1. Base too low
        too_low = self.base_height < self.cfg.termination_cfg["base_height_min"]

        # 2. Base too high (excessive jump)
        too_high = self.base_height > self.cfg.termination_cfg["base_height_max"]

        # 3. Excessive roll/pitch (use gravity projection)
        tilted = torch.abs(self.projected_gravity[:, 2]) < torch.cos(
            torch.tensor(self.cfg.termination_cfg["roll_pitch_max"], device=self.device)
        )

        # Combine all termination conditions
        terminated = too_low | too_high | tilted

        # ============================================================
        # Timeout condition
        # ============================================================
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return terminated, time_out
        
    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset specified environments.

        Args:
            env_ids: environment IDs to reset, None resets all environments
        """
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES

        # ============================================================
        # 1. Reset robot physical state
        # ============================================================
        self.robot.reset(env_ids)

        # ============================================================
        # 2. Randomize initial states
        # ============================================================
        # Randomize initial position
        pos_range = self.cfg.init_state_cfg["pos_range"]
        init_pos = self.robot.data.default_root_state[env_ids, :3].clone()
        init_pos[:, 0] += sample_uniform(
            pos_range["x"][0], pos_range["x"][1], (len(env_ids),), self.device
        )
        init_pos[:, 1] += sample_uniform(
            pos_range["y"][0], pos_range["y"][1], (len(env_ids),), self.device
        )
        init_pos[:, 2] += pos_range["z"][0]  # usually don't randomize z

        # Randomize initial orientation
        rot_range = self.cfg.init_state_cfg["rot_range"]
        yaw = sample_uniform(
            rot_range["yaw"][0], rot_range["yaw"][1], (len(env_ids),), self.device
        )
        init_quat = torch.zeros(len(env_ids), 4, device=self.device)
        init_quat[:, 0] = torch.cos(yaw / 2)  # w
        init_quat[:, 3] = torch.sin(yaw / 2)  # z

        # Randomize initial velocity
        vel_range = self.cfg.init_state_cfg["vel_range"]
        init_vel = torch.zeros(len(env_ids), 6, device=self.device)
        init_vel[:, 0] = sample_uniform(
            vel_range["x"][0], vel_range["x"][1], (len(env_ids),), self.device
        )
        init_vel[:, 1] = sample_uniform(
            vel_range["y"][0], vel_range["y"][1], (len(env_ids),), self.device
        )

        # ============================================================
        # 3. Set initial joint positions
        # ============================================================
        # Ensure default_joint_pos is initialized
        if self.default_joint_pos is None:
            from Franka_RL.robots import QuadrupedRobotFactory
            go2_robot = QuadrupedRobotFactory.create_robot("unitree_go2")

            # joint_pos is a dict; convert to a list ordered by dof_names
            joint_pos_dict = go2_robot.init_state.joint_pos
            joint_pos_list = [joint_pos_dict[name] for name in go2_robot.dof_names]

            self.default_joint_pos = torch.tensor(
                joint_pos_list,
                dtype=torch.float32,
                device=self.device,
            )

        joint_pos = self.default_joint_pos.unsqueeze(0).repeat(len(env_ids), 1)

        # Optionally add small random perturbations
        joint_pos += sample_uniform(-0.1, 0.1, joint_pos.shape, self.device)

        joint_vel = torch.zeros(len(env_ids), 12, device=self.device)

        # ============================================================
        # 4. Write states into the simulator
        # ============================================================
        # Write root state
        root_state = torch.cat([init_pos, init_quat, init_vel], dim=1)
        self.robot.write_root_state_to_sim(root_state, env_ids)

        # Write joint states
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # ============================================================
        # 5. Resample commands
        # ============================================================
        self._resample_commands(env_ids)

        # ============================================================
        # 6. Reset internal buffers/state
        # ============================================================
        self.last_actions[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        
    def _resample_commands(self, env_ids: torch.Tensor):
        """Resample velocity commands."""
        n = len(env_ids)

        # linear velocity x
        self.commands[env_ids, 0] = sample_uniform(
            self.command_ranges["lin_vel_x"][0],
            self.command_ranges["lin_vel_x"][1],
            (n,), self.device,
        )

        # linear velocity y
        self.commands[env_ids, 1] = sample_uniform(
            self.command_ranges["lin_vel_y"][0],
            self.command_ranges["lin_vel_y"][1],
            (n,), self.device,
        )

        # angular velocity z
        self.commands[env_ids, 2] = sample_uniform(
            self.command_ranges["ang_vel_z"][0],
            self.command_ranges["ang_vel_z"][1],
            (n,), self.device,
        )

        # heading
        self.commands[env_ids, 3] = sample_uniform(
            self.command_ranges["heading"][0],
            self.command_ranges["heading"][1],
            (n,), self.device,
        )
