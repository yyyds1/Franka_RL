"""Script to train agent with DAgger."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from copy import deepcopy

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an agent with DAgger.")
parser.add_argument("--epoch", type=int, default=None, help="DAgger Policy initialization training epoches.")
parser.add_argument("--iteration", type=int, default=None, help="DAgger Policy training iterations.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="DAgger_cfg_entry_point", help="Name of the DAgger agent configuration entry point."
)

# append RSL-RL cli arguments
cli_args.add_DAgger_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
import numpy as np
from datetime import datetime

import omni

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)

from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import Franka_RL.tasks  # noqa: F401
from Franka_RL.dataset import DataFactory
from Franka_RL.runners import MyDAggerTrainer

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

import tempfile

import numpy as np
import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy

from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with DAgger agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    
    rng = np.random.default_rng(0)
    env = make_vec_env(
        args_cli.task,
        rng=rng,
        env_make_kwargs={
            "cfg": env_cfg,
        }
    )
    
    with tempfile.TemporaryDirectory(prefix="dagger_example_") as tmpdir:
        print(tmpdir)
        dagger_trainer = MyDAggerTrainer(
            venv=env,
            scratch_dir=tmpdir,
            rng=rng,
            train_cfg=agent_cfg,
        )
        dagger_trainer.train(8_000)

    reward, _ = evaluate_policy(dagger_trainer.policy, env, 10)
    print("Reward:", reward)