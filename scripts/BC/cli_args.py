from __future__ import annotations

import argparse
import random
from typing import TYPE_CHECKING

def add_BC_args(parser: argparse.ArgumentParser):
    """Add BC arguments to the parser.

    Args:
        parser: The parser to add the arguments to.
    """
    # create a new argument group
    arg_group = parser.add_argument_group("BC", description="Arguments for BC agent.")
    # -- experiment arguments
    arg_group.add_argument(
        "--experiment_name", type=str, default="BC", help="Name of the experiment folder where logs will be stored."
    )
    arg_group.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
    # -- load arguments
    arg_group.add_argument("--resume", action="store_true", default=False, help="Whether to resume from a checkpoint.")
    arg_group.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from.")
    arg_group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")
    # -- logger arguments
    arg_group.add_argument(
        "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="Logger module to use."
    )
    arg_group.add_argument(
        "--log_project_name", type=str, default=None, help="Name of the logging project when using wandb or neptune."
    )

def update_BC_cfg(agent_cfg: dict, args_cli: argparse.Namespace):
    """Update configuration for BC agent based on inputs.

    Args:
        agent_cfg: The configuration for BC agent.
        args_cli: The command line arguments.

    Returns:
        The updated configuration for BC agent based on inputs.
    """
    # override the default configuration with CLI arguments
    if hasattr(args_cli, "seed") and args_cli.seed is not None:
        # randomly sample a seed if seed = -1
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        agent_cfg["seed"] = args_cli.seed
    if args_cli.epoch is not None:
        agent_cfg["epoch"] = args_cli.epoch
    if args_cli.logger is not None:
        agent_cfg["logger"] = args_cli.logger
    # set the project name for wandb and neptune
    if agent_cfg["logger"] in {"wandb", "neptune"} and args_cli.log_project_name:
        agent_cfg["wandb_project"] = args_cli.log_project_name
        agent_cfg["neptune_project"] = args_cli.log_project_name

    return agent_cfg
