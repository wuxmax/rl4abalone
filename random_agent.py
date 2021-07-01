import gym

import numpy as np

from typing import Dict, List, Tuple


class RandomAgent:

    def __init__(
            self,
            env: gym.Env,
            priorities: List = None
    ):
        self.env = env
        self.priorities = priorities

    def cvst(self, state: np.ndarray, turn: int) -> np.ndarray:
        """Convert gym_abalone state into 121x3 representation"""
        black = state.flatten().copy()
        black[black < 1] = 0
        white = state.flatten()
        white[white > 0] = -1
        white[white == 0] = 1
        white[white < 1] = 0
        current_player = np.zeros(121, dtype="int64") if turn % 2 == 0 else np.ones(121, dtype="int64")
        return np.concatenate((black, white, current_player), axis=0)

    def select_action(self, state: np.ndarray) -> Tuple:
        """Select an action from the input state."""
        player = self.env.game.current_player
        possible_moves = self.env.game.get_possible_moves(player, group_by_type=True)

        if self.priorities:
            for move_type in self.priorities:
                if possible_moves[move_type]:
                    i_random = np.random.randint(len(possible_moves[move_type]))
                    pos0, pos1 = possible_moves[move_type][i_random]
                    return pos0, pos1

        i_random = np.random.randint(len(possible_moves))
        pos0, pos1 = possible_moves[i_random]

        return pos0, pos1

    def step(self, action: Tuple, turn: int) -> Tuple[np.ndarray, np.float64, bool, Dict]:
        """Take an action and return the response of the env, where the state is already in 121x3 representation"""
        next_state, reward, done, info = self.env.step(action)
        next_state = self.cvst(next_state, turn+1)
        return next_state, reward, done, info