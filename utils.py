import random
import torch

import numpy as np


def set_seeds(seed, env):
    np.random.seed(seed)
    random.seed(seed)
    env.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def cvst(state: np.ndarray, current_player: int = 0) -> np.ndarray:
    """Convert gym_abalone state into 121x3 representation"""
    black = state.flatten().copy()
    black[black < 1] = 0
    white = state.flatten()
    white[white > 0] = -1
    white[white == 0] = 1
    white[white < 1] = 0
    current_player = np.zeros(121, dtype="int64") if current_player is 0 else np.ones(121, dtype="int64")
    return np.concatenate((black, white, current_player), axis=0)


def cvact(action: int):
    """Convert action index into position->position action"""
    # return selected_action
    # actions are converted according to this: https://github.com/towzeur/gym-abalone#actions
    return action // 61, action % 61


def next_player(player: str):
    return "white" if player is "black" else "black"

