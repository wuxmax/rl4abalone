import random

import gym
from gym_abalone.envs import abalone_env
import torch
import numpy as np

from rainbow import DQNAgent


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

memory_size = 10000
batch_size = 128
target_update = 100

agent = DQNAgent(env, memory_size, batch_size, target_update)

N_EPISODES = 1
for episode in range(1, N_EPISODES + 1):
    state = env.reset()

    done = False
    while not done:
        # ==== YOUR AGENT HERE ====

        action = agent.select_action(state)

        print(f"ACTION: {action}")

        # =========================
        state, reward, done, info = env.step(action)  # action

        print(f"{info['turn']: <4} | {info['player_name']} | {str(info['move_type']): >16} | reward={reward: >4} ")
        env.render(fps=0.5)

    print(f"Episode {info['turn']: <4} finished after {env.game.turns_count} turns \n")

env.close()
