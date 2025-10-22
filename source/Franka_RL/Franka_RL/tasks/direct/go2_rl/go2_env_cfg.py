# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

# Import robot factory
from Franka_RL.robots import QuadrupedRobotFactory

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip



def get_robot_cfg() -> ArticulationCfg:
    """Get Go2 robot configuration from factory."""
    go2_robot = QuadrupedRobotFactory.create_robot("unitree_go2")
    
    robot_cfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=go2_robot._usd_path,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=go2_robot.init_state,
        actuators=go2_robot.actuators,
    )
    return robot_cfg


@configclass
class Go2EnvCfg(DirectRLEnvCfg):
    """Configuration for the Go2 locomotion environment."""

    
    observation_space = 45
    action_space = 12
    state_space = 0
    
    # env
    episode_length_s = 20.0
    decimation = 4
    action_scale = 0.25  
    num_envs = 4096
    env_spacing = 2.5
    
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=num_envs, 
        env_spacing=env_spacing, 
        replicate_physics=True
    )

  
    robot: ArticulationCfg = "Franka_RL.tasks.direct.go2_rl.go2_env_cfg:get_robot_cfg"
    
    # reward scales
    lin_vel_reward_scale = 1.0
    ang_vel_reward_scale = 0.5
    joint_torque_reward_scale = -0.0002
    joint_acc_reward_scale = -2.5e-7
    action_rate_reward_scale = -0.01
    feet_air_time_reward_scale = 0.5
    undersired_contact_reward_scale = -1.0
    flat_orientation_reward_scale = -5.0

    # normalization
    obs_scales = {
        "lin_vel": 2.0,
        "ang_vel": 0.25,
        "dof_pos": 1.0,
        "dof_vel": 0.05,
        "gravity": 1.0,
        "commands": [2.0, 2.0, 0.25],  #  [vx_scale, vy_scale, wz_scale]
    }

    # noise
    noise_scales = {
        "lin_vel": 0.1,
        "ang_vel": 0.2,
        "dof_pos": 0.01,
        "dof_vel": 1.5,
        "gravity": 0.05,
    }

   
    commands_cfg = {
        "lin_vel_x_range": [-1.0, 1.0],
        "lin_vel_y_range": [-1.0, 1.0],
        "ang_vel_z_range": [-1.0, 1.0],
        "heading_range": [-3.14, 3.14],
    }
    
    
    init_state_cfg = {
        "pos_range": {
            "x": [-0.5, 0.5],
            "y": [-0.5, 0.5],
            "z": [0.0, 0.0],
        },
        "rot_range": {
            "roll": [0.0, 0.0],
            "pitch": [0.0, 0.0],
            "yaw": [-3.14, 3.14],
        },
        "vel_range": {
            "x": [-0.5, 0.5],
            "y": [-0.5, 0.5],
            "z": [0.0, 0.0],
        },
    }
    
    termination_cfg = {
        "base_height_min": 0.20,
        "base_height_max": 0.50,
        "roll_pitch_max": 0.7,  # ~40度
    }
    
    # termination 
    termination_height = 0.3
    
    def __post_init__(self):
        """Post initialization."""
    
        if isinstance(self.robot, str):
            
            self.robot = get_robot_cfg()
        
       
        self.scene.robot = self.robot
