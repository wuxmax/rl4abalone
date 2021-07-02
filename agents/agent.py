from typing import Tuple

import gym
import numpy as np


class Agent:
    def __init__(
            self,
            env: gym.Env
    ):
        self.env = env

    def select_action(self, state: np.ndarray):
        raise NotImplementedError

    def step(self, action: Tuple[int, int], turn: int):
        raise NotImplementedError
