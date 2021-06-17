import random
import pickle

import gym
from gym_abalone.envs import abalone_env
import torch
import numpy as np

from rainbow import DQNAgent

AGENT_FILE_PATH: str = "rainbow-agent.pth"
LOAD_FROM_FILE: bool = True
SAVE_TO_FILE: bool = False


def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

env = gym.make("abalone-v0")

seed = 777

np.random.seed(seed)
random.seed(seed)
seed_torch(seed)
env.seed(seed)

num_frames = 2000
memory_size = 1000
batch_size = 128
target_update = 100

if not LOAD_FROM_FILE:
    agent = DQNAgent(env, memory_size, batch_size, target_update)
    agent.train(num_frames)

    if SAVE_TO_FILE:
        with open(AGENT_FILE_PATH, "wb") as f:
            pickle.dump(agent, f)   
else:
    with open(AGENT_FILE_PATH, "rb") as f:
        agent = torch.load(f, map_location='cpu')
        agent.reset_torch_device()


agent.test()