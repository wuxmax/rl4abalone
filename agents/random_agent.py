from typing import Dict, List, Tuple

import gym
import numpy as np

from agents.agent import Agent
from utils import cvst

rng = np.random.default_rng(0)


class RandomAgent(Agent):
    def __init__(
            self,
            env: gym.Env,
            priorities: List = None
    ):
        self.env = env
        self.priorities = priorities

    def select_action(self, state: np.ndarray) -> Tuple:
        """Select an action from the input state."""
        player = self.env.game.current_player
        use_priority = self.priorities is not None
        possible_moves = self.env.game.get_possible_moves(player, group_by_type=use_priority)

        if self.priorities:
            for move_type in self.priorities:
                if possible_moves[move_type]:
                    return rng.choice(possible_moves[move_type])

        return rng.choice(possible_moves)

    def step(self, action: Tuple) -> Tuple[np.ndarray, np.float64, bool, Dict]:
        """Take an action and return the response of the env, where the state is already in 121x3 representation"""
        next_state, reward, done, info = self.env.step(action)
        next_state = cvst(next_state, self.env.current_player)
        return next_state, reward, done, info
