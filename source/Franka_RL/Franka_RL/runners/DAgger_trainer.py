import abc
import logging
import os
import pathlib
import uuid
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch as th
from stable_baselines3.common import policies, utils, vec_env
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn
from torch.utils import data as th_data

from imitation.algorithms import base, bc
from imitation.algorithms.dagger import DAggerTrainer, _save_dagger_demo
from imitation.data import rollout, serialize, types
from imitation.util import logger as imit_logger
from imitation.util import util


class MyDAggerTrainer(DAggerTrainer):
    """Simpler subclass of DAggerTrainer for training with synthetic feedback."""

    def __init__(
        self,
        *,
        venv: vec_env.VecEnv,
        scratch_dir: types.AnyPath,
        rng: np.random.Generator,
        train_cfg: dict,
        expert_trajs: Optional[Sequence[types.Trajectory]] = None,
        **dagger_trainer_kwargs,
    ):
        """Builds SimpleDAggerTrainer.

        Args:
            venv: Vectorized training environment. Note that when the robot
                action is randomly injected (in accordance with `beta_schedule`
                argument), every individual environment will get a robot action
                simultaneously for that timestep.
            scratch_dir: Directory to use to store intermediate training
                information (e.g. for resuming training).
            rng: Random state to use for the random number generator.
            train_cfg: Main config of DAgger training.
            expert_trajs: Optional starting dataset that is inserted into the round 0
                dataset.
            dagger_trainer_kwargs: Other keyword arguments passed to the
                superclass initializer `DAggerTrainer.__init__`.

        Raises:
            ValueError: The observation or action space does not match between
                `venv` and `expert_policy`.
        """
        bc_trainer = bc.BC

        super().__init__(
            venv=venv,
            scratch_dir=scratch_dir,
            rng=rng,
            **dagger_trainer_kwargs,
        )
        self.expert_policy = expert_policy

        # TODO(shwang):
        #   Might welcome Transitions and DataLoaders as sources of expert data
        #   in the future too, but this will require some refactoring, so for
        #   now we just have `expert_trajs`.
        if expert_trajs is not None:
            # Save each initial expert trajectory into the "round 0" demonstration
            # data directory.
            for traj_index, traj in enumerate(expert_trajs):
                _save_dagger_demo(
                    traj,
                    traj_index,
                    self._demo_dir_path_for_round(),
                    self.rng,
                    prefix="initial_data",
                )

    def train(
        self,
        total_timesteps: int,
        *,
        rollout_round_min_episodes: int = 3,
        rollout_round_min_timesteps: int = 500,
        bc_train_kwargs: Optional[dict] = None,
    ) -> None:
        """Train the DAgger agent.

        The agent is trained in "rounds" where each round consists of a dataset
        aggregation step followed by BC update step.

        During a dataset aggregation step, `self.expert_policy` is used to perform
        rollouts in the environment but there is a `1 - beta` chance (beta is
        determined from the round number and `self.beta_schedule`) that the DAgger
        agent's action is used instead. Regardless of whether the DAgger agent's action
        is used during the rollout, the expert action and corresponding observation are
        always appended to the dataset. The number of environment steps in the
        dataset aggregation stage is determined by the `rollout_round_min*` arguments.

        During a BC update step, `BC.train()` is called to update the DAgger agent on
        all data collected so far.

        Args:
            total_timesteps: The number of timesteps to train inside the environment.
                In practice this is a lower bound, because the number of timesteps is
                rounded up to finish the minimum number of episodes or timesteps in the
                last DAgger training round, and the environment timesteps are executed
                in multiples of `self.venv.num_envs`.
            rollout_round_min_episodes: The number of episodes the must be completed
                completed before a dataset aggregation step ends.
            rollout_round_min_timesteps: The number of environment timesteps that must
                be completed before a dataset aggregation step ends. Also, that any
                round will always train for at least `self.batch_size` timesteps,
                because otherwise BC could fail to receive any batches.
            bc_train_kwargs: Keyword arguments for calling `BC.train()`. If
                the `log_rollouts_venv` key is not provided, then it is set to
                `self.venv` by default. If neither of the `n_epochs` and `n_batches`
                keys are provided, then `n_epochs` is set to `self.DEFAULT_N_EPOCHS`.
        """
        total_timestep_count = 0
        round_num = 0

        while total_timestep_count < total_timesteps:
            collector = self.create_trajectory_collector()
            round_episode_count = 0
            round_timestep_count = 0

            sample_until = rollout.make_sample_until(
                min_timesteps=max(rollout_round_min_timesteps, self.batch_size),
                min_episodes=rollout_round_min_episodes,
            )

            trajectories = rollout.generate_trajectories(
                policy=self.expert_policy,
                venv=collector,
                sample_until=sample_until,
                deterministic_policy=True,
                rng=collector.rng,
            )

            for traj in trajectories:
                self._logger.record_mean(
                    "dagger/mean_episode_reward",
                    np.sum(traj.rews),
                )
                round_timestep_count += len(traj)
                total_timestep_count += len(traj)

            round_episode_count += len(trajectories)

            self._logger.record("dagger/total_timesteps", total_timestep_count)
            self._logger.record("dagger/round_num", round_num)
            self._logger.record("dagger/round_episode_count", round_episode_count)
            self._logger.record("dagger/round_timestep_count", round_timestep_count)

            # `logger.dump` is called inside BC.train within the following fn call:
            self.extend_and_update(bc_train_kwargs)
            round_num += 1
