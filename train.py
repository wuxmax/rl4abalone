import gym
import torch
import numpy as np

from gym_abalone.envs import abalone_env
from gym_abalone.envs.abalone_env import AbaloneEnv

from agents.rainbow.agent import RainbowAgent
from utils import set_seeds


AGENT_FILE_PATH: str = "trained-agents/rainbow-agent_test.pth"
LOAD_FROM_FILE: bool = False
SAVE_TO_FILE: bool = True
MAX_TURNS: int = 400


# env = gym.make("abalone-v0")
env = AbaloneEnv(max_turns=MAX_TURNS)

SEED = 12
set_seeds(SEED, env)

num_turns_total = 214800  # 150000 = ~7h, 214800 ~10h
memory_size = 10000
batch_size = 32
target_update = 10

if not LOAD_FROM_FILE:
    agent = RainbowAgent(env, memory_size, batch_size, target_update, hidden_dim=128)
else:
    with open(AGENT_FILE_PATH, "rb") as f:
        agent = torch.load(f, map_location='cpu')  # cpu is correct?
    print("Agent successfully loaded.")
    agent.reset_torch_device()

agent.train(num_turns_total, plotting_interval=MAX_TURNS)

if SAVE_TO_FILE:
    with open(AGENT_FILE_PATH, "wb") as f:
        torch.save(agent, f)
    print("Agent successfully saved.")
