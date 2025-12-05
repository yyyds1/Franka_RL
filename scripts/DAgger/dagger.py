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

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with DAgger agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_DAgger_cfg(agent_cfg, args_cli)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg["experiment_name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    
    


img_dim = [64,64,3]
action_dim = 1
steps = 1000
batch_size = 32
nb_epoch = 100

def get_teacher_action(ob):
    steer = ob.angle*10/np.pi
    steer -= ob.trackPos*0.10
    return np.array([steer])

def img_reshape(input_img):
    _img = np.transpose(input_img, (1, 2, 0))
    _img = np.flipud(_img)
    _img = np.reshape(_img, (1, img_dim[0], img_dim[1], img_dim[2]))
    return _img

images_all = np.zeros((0, img_dim[0], img_dim[1], img_dim[2]))
actions_all = np.zeros((0,action_dim))
rewards_all = np.zeros((0,))

img_list = []
action_list = []
reward_list = []

env = TorcsEnv(vision=True, throttle=False)
ob = env.reset(relaunch=True)

print('Collecting data...')
for i in range(steps):
    if i == 0:
        act = np.array([0.0])
    else:
        act = get_teacher_action(ob)

    if i%100 == 0:
        print(i)
    ob, reward, done, _ = env.step(act)
    img_list.append(ob.img)
    action_list.append(act)
    reward_list.append(np.array([reward]))

env.end()

print('Packing data into arrays...')
for img, act, rew in zip(img_list, action_list, reward_list):
    images_all = np.concatenate([images_all, img_reshape(img)], axis=0)
    actions_all = np.concatenate([actions_all, np.reshape(act, [1,action_dim])], axis=0)
    rewards_all = np.concatenate([rewards_all, rew], axis=0)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam

#model from https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py
model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=img_dim))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(action_dim))
model.add(Activation('tanh'))

model.compile(loss='mean_squared_error',
              optimizer=Adam(lr=1e-4),
              metrics=['mean_squared_error'])

model.fit(images_all, actions_all,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              shuffle=True)

output_file = open('results.txt', 'w')

#aggregate and retrain
dagger_itr = 5
for itr in range(dagger_itr):
    ob_list = []

    env = TorcsEnv(vision=True, throttle=False)
    ob = env.reset(relaunch=True)
    reward_sum = 0.0

    for i in range(steps):
        act = model.predict(img_reshape(ob.img))
        ob, reward, done, _ = env.step(act)
        if done is True:
            break
        else:
            ob_list.append(ob)
        reward_sum += reward
        print(i, reward, reward_sum, done, str(act[0]))
    print('Episode done ', itr, i, reward_sum)
    output_file.write('Number of Steps: %02d\t Reward: %0.04f\n'%(i, reward_sum))
    env.end()

    if i==(steps-1):
        break

    for ob in ob_list:
        images_all = np.concatenate([images_all, img_reshape(ob.img)], axis=0)
        actions_all = np.concatenate([actions_all, np.reshape(get_teacher_action(ob), [1,action_dim])], axis=0)

    model.fit(images_all, actions_all,
                  batch_size=batch_size,
                  nb_epoch=nb_epoch,
                  shuffle=True)