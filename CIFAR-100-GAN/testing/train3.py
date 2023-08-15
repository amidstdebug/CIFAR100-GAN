import gym

import pickle
import plotly
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import tensorflow as tf
from tensorflow.keras.models import load_model

from utils import seed_everything, Experience, ReplayBuffer
from collections import deque

from model.dqn import DQN
from model.ddqn import DoubleDQN as DDQN
from model.sarsa import SARSA
from model.dqn import DQN
from model.ddqn import DoubleDQN as DDQN
from model.sarsa import SARSA

import gym
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import base64, io
import wandb
seed_everything(1)
plotly.offline.init_notebook_mode()

import numpy as np
from collections import deque, namedtuple

# For visualization
from gym.wrappers.monitoring import video_recorder
from IPython.display import HTML
from IPython import display 
import glob


import wandb
wandb.login()
wandb.init()

env = gym.make('LunarLander-v2')
env.seed(1)

# Import Code for DQN
dqn_model = DQN(
    env = env,
    lr =  0.0005,
    gamma = 0.99,
    epsilon = 1,
    epsilon_decay = 0.95,
    target_update_interval = 20,
    log_wandb = True
)


def train(config=None):
    with wandb.init(config=config):
        config = wandb.config
        env = gym.make("LunarLander-v2")
        model = DQN(
            env=env,
            lr=config.lr,
            gamma=config.gamma,
            epsilon=config.epsilon,
            epsilon_decay=config.epsilon_decay,
            target_update_interval=config.update_target_net_interval,
            log_wandb=True,
            tuning_condition=True
        )
        model.train(config.episodes, mean_stopping=True)

sweep_id = "3fzhy39d"
project_name = "DQN-Tuning"
num_runs = 50


wandb.agent(sweep_id, train, count=num_runs, entity="onsen", project=project_name)