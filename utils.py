from typing import List, Dict

import random
import torch

import pandas as pd
import numpy as np
from gym_abalone.envs.abalone_env import Reward


def set_seeds(seed, env, cudnn_deterministic: bool = False):
    np.random.seed(seed)
    random.seed(seed)
    env.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = not cudnn_deterministic
        torch.backends.cudnn.deterministic = cudnn_deterministic


def cvst(state: np.ndarray, current_player: int = 0) -> np.ndarray:
    """Convert gym_abalone state into 121x3 representation"""
    black = state.flatten().copy()
    black[black < 1] = 0
    white = state.flatten()
    white[white > 0] = -1
    white[white == 0] = 1
    white[white < 1] = 0
    current_player_layer = np.zeros(121, dtype="int64") if current_player == 0 else np.ones(121, dtype="int64")
    return np.concatenate((black, white, current_player_layer), axis=0)


def cvact(action: int) -> (int, int):
    """Convert action index into position->position action"""
    # return selected_action
    # actions are converted according to this: https://github.com/towzeur/gym-abalone#actions
    return action // 61, action % 61


def get_atom_distribution_borders() -> (float, float):
    """Import correct Rewards and construct v_max and v_min accordingly"""
    """(used for the distributional network implementation)"""
    v_max = float(Reward.method_1(None, "winner") + 6 * Reward.method_1(None, "ejected"))
    return v_max, -v_max


def print_action_prob_info(action_probs: np.ndarray, action_mask: np.ndarray):
    action_probs_masked = action_probs * action_mask
    num_actions_total = action_mask.shape[0]
    mean_action_probs = action_probs.mean()
    num_legal_actions = np.count_nonzero(action_mask)
    num_illegal_actions = np.count_nonzero(action_mask == 0)
    is_maximum_masked = action_mask[action_probs.argmax()] == 0
    mean_legal_probs = action_probs_masked.sum() / np.count_nonzero(action_probs_masked)
    action_probs_masked_inverse = (1 - action_mask) * action_probs
    mean_illegal_probs = action_probs_masked_inverse.sum() / np.count_nonzero(action_probs_masked_inverse)
    print(f" Num actions total: {num_actions_total} | Num illegal actions: {num_illegal_actions} |"
          f" Num legal actions. {num_legal_actions} | Is maximum masked? {is_maximum_masked}")
    print(f" Mean actions probs: {mean_action_probs} |"
          f" Mean legal probs: {mean_legal_probs} | Mean illegal probs: {mean_illegal_probs}")
    print(f" Max action prob: {action_probs.max()} | Max legal prob {action_probs_masked.max()} |"
          f" Max illegal prob {action_probs_masked_inverse.max()}")
    action_probs_masked_min = action_probs[np.nonzero(action_probs_masked)].min()
    action_probs_masked_inverse_min = action_probs[np.nonzero(action_probs_masked_inverse)].min()
    print(f" Min action prob: {action_probs.min()} | Min legal prob {action_probs_masked_min} |"
          f" Min illegal prob {action_probs_masked_inverse_min}")


def print_latex_result_table(df_results: pd.DataFrame):
    column_name_mapping = {
        'agent_white_name': "White agent",
        'agent_black_name': "Black agent",
        'winner': "Winner",
        'agent_white_score': "Score (white)",
        'agent_black_score': "Score (black)",
        'num_turns': "# Turns (total)",
        'agent_white_score_per_turn': "Score / # Turns (white)",
        'agent_black_score_per_turn': "Score / # Turns (black)",
        'agent_white_unique_turn_ratio': "# Unique turns / # Turns (white)",
        'agent_black_unique_turn_ratio': "# Unique turns / # Turns (black)",
        'agent_white_ejects': "# Ejects (white)",
        'agent_black_ejects': "# Ejects (black)"
    }

    df_results = df_results[column_name_mapping.keys()]
    df_results = df_results.rename(mapper=column_name_mapping, axis='columns')

    print("--- Latex result table ---")
    print(df_results.to_latex(float_format="{:0.3f}".format, index=False))




