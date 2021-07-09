import gym
import torch
import numpy as np

from gym_abalone.envs import abalone_env
from gym_abalone.envs.abalone_env import AbaloneEnv

from agents.rainbow.agent import RainbowAgent
from utils import set_seeds


AGENT_FILE_PATH: str = "trained-agents/rainbow-agent_test.pth"
LOAD_FROM_FILE: bool = False
MAX_TURNS: int = 400


# env = gym.make("abalone-v0")
env = AbaloneEnv(max_turns=MAX_TURNS)

SEED = 12
set_seeds(SEED, env)

num_turns_total = 100  # 150000 = ~7h, 214800 ~10h
save_interval = 1
warmup_period = 0

if not LOAD_FROM_FILE:
    agent = RainbowAgent(env, warmup_period=warmup_period, save_interval=save_interval, save_path=AGENT_FILE_PATH)
else:
    with open(AGENT_FILE_PATH, "rb") as f:
        agent = torch.load(f, map_location='cpu')  # cpu is correct?
    print("Agent successfully loaded.")
    agent.reset_torch_device()

agent.train(num_turns_total, plotting_interval=MAX_TURNS)

