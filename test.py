import os.path
from typing import Dict, List, Tuple

import pandas as pd
import torch
import numpy as np
from halo import Halo

from gym_abalone.envs.abalone_env import AbaloneEnv
from agents.rainbow.agent import RainbowAgent
from agents.rainbow.config import RainbowConfig
from agents.random_agent import RandomAgent
from agents.agent import Agent
from utils import cvst, set_seeds

# patch pickel error
import sys
from agents.rainbow import config
sys.modules['config'] = config

# AGENT_FILE_PATHS: List = ["trained-agents/ra_noisy-net_std-init-5_1000000_.pth", "random"]
AGENT_FILE_PATHS: List = ["trained-agents/ra_noisy-net_std-init-5_1000000_.pth", "trained-agents/ra_noisy-net_std-init-5_1000000_.pth"]
# AGENT_FILE_PATHS: List = [None]
# GAMES_PER_BENCHMARK = 100
GAMES_PER_BENCHMARK = 10
MAX_TURNS: int = 400
ENABLE_GUI: bool = False
RESULTS_FILE = "results.xlsx"
RANDOM_SEED = 777

spinner = Halo(spinner='dots')


def load_agent(path: str, env: AbaloneEnv):
    if not path:
        memory_size = 1000
        batch_size = 128
        target_update = 100
        config = RainbowConfig()
        agent = RainbowAgent(env, memory_size, batch_size, target_update, feature_conf=config)

    elif path == "random":
        agent = RandomAgent(env, priorities=['winner', 'ejected', 'inline_push', 'inline_move', 'sidestep_move'])

    else:
        with open(path, "rb") as f:
            agent = torch.load(f, map_location=torch.device('cpu'))
            agent.reset_torch_device()
            agent.env = env

    return agent


def print_game_info(info: Dict, reward: int, score_white: int, score_black: int):
    if str(info['move_type']) == "ejected":
        print(f"\n{info['turn']: <4} | {info['player_name']} | {str(info['move_type']): >16} "
              f"| reward={reward: >4}")

    elif str(info['move_type']) == "winner":
        # score update depending on defeat
        score_winner = score_white if info['player_name'] == 'white' else score_black
        score_looser = score_white - 1 if info['player_name'] == 'black' else score_black - 1

        print(f"\n{info['player_name']} won in {info['turn']: <4} turns with a total score of {score_winner}!\n"
              f"The looser scored {score_looser}!")


def test_step(agent: Agent, state: np.ndarray, score_white: int, score_black: int, enable_gui: bool,
              last_actions_white: List, last_actions_black: List, ejects_white: int, ejects_black: int):

    action = agent.select_action(state)
    next_state, reward, done, info = agent.step(action)

    if info["player"] == 0:
        score_white += reward
        score_black -= reward
        last_actions_white.append(action)
        if info['move_type'] == 'ejected':
            ejects_white += 1
    else:
        score_black += reward
        score_white -= reward
        last_actions_black.append(action)
        if info['move_type'] == 'ejected':
            ejects_black += 1

    # print_game_info(info=info, reward=reward, score_white=score_white, score_black=score_black)

    if enable_gui:
        agent.env.render(fps=1)

    return next_state, score_white, score_black, done, last_actions_white, last_actions_black, ejects_white, ejects_black


# def self_play(agent_file_path: str, max_turns: int = 400, enable_gui: bool = False, episodes: int = 1):
#     """Test the agent."""
#     env = AbaloneEnv(max_turns=max_turns)
#     set_seeds(RANDOM_SEED, env)
#
#     agent = load_agent(agent_file_path, env)
#     agent.is_test = True
#
#     last_actions_white = []
#     last_actions_black = []
#
#     for episode in range(episodes):
#         state = cvst(env.reset(random_player=False))
#         score_black = 0
#         score_white = 0
#         done = False
#         turns_white = 0
#         turns_black = 0
#
#         spinner.start(text=f"Playing episode {episode + 1}/{episodes}")
#
#         while not done:
#             state, score_white, score_black, done, last_actions_white, last_actions_black =\
#                 test_step(agent=agent, state=state, score_white=score_white, score_black=score_black,
#                           enable_gui=enable_gui, last_actions_white=last_actions_white,
#                           last_actions_black=last_actions_black)
#
#         print(f"In the last game white has made {len(set(last_actions_white))} (ratio:"
#               f"{len(set(last_actions_white)) / turns_white}) and black"
#               f"has made {len(set(last_actions_black))} (ratio:"
#               f"{len(set(last_actions_black)) / turns_black}) unique turns")
#
#         spinner.stop()
#
#     env.close()


def agent_vs_agent(white_agent_file_path: str, black_agent_file_path: str, max_turns: int = 400,
                   enable_gui: bool = False, episodes: int = 1):
    env = AbaloneEnv(max_turns=max_turns)
    set_seeds(RANDOM_SEED, env)

    agent_white_name = os.path.basename(white_agent_file_path)
    agent_black_name = os.path.basename(black_agent_file_path)

    print(f"Loading white agent: '{agent_white_name}' ...")
    agent_white = load_agent(white_agent_file_path, env)
    print(f"Done loading!")
    print(f"Loading white agent: '{agent_black_name}' ...")
    agent_black = load_agent(black_agent_file_path, env)
    print(f"Done loading!")

    agent_white.is_test = True
    agent_black.is_test = True

    results = []
    for episode in range(episodes):
        turns_white = 0
        turns_black = 0
        score_black = 0
        score_white = 0
        last_actions_white = []
        last_actions_black = []
        ejects_white = 0
        ejects_black = 0

        state = cvst(env.reset(random_player=False))
        done = False

        spinner.start(text=f"Playing episode {episode + 1}/{episodes}")

        while not done:
            if env.current_player == 0:
                turn_player = agent_white
                turns_white += 1
            else:
                turn_player = agent_black
                turns_black += 1

            state, score_white, score_black, done, last_actions_white, last_actions_black, ejects_white, ejects_black =\
                test_step(agent=turn_player, state=state, enable_gui=enable_gui, score_white=score_white, score_black=score_black,
                          last_actions_white=last_actions_white, last_actions_black=last_actions_black,
                          ejects_white=ejects_white, ejects_black=ejects_black)

        winner = 'draw'
        if score_white > score_black:
            winner = 'white'
        if score_black > score_white:
            winner = 'black'

        game_name = agent_white_name + agent_black_name + str(episode + 1)

        result = {
            'game_name': game_name,
            'agent_white_name': agent_white_name,
            'agent_black_name': agent_black_name,
            'agent_white_score': score_white,
            'agent_black_score': score_black,
            'winner': winner,
            'num_turns': turns_white + turns_black,
            'agent_white_score_per_turn': score_white / turns_white,
            'agent_black_score_per_turn': score_black / turns_black,
            'agent_white_unique_turn_ratio': len(set(last_actions_white)) / turns_white,
            'agent_black_unique_turn_ratio': len(set(last_actions_black)) / turns_black,
            'agent_white_ejects': ejects_white,
            'agent_black_ejects': ejects_black,
            'agent_white_ejects_per_turn': ejects_white / turns_white,
            'agent_black_ejects_per_turn': ejects_black / turns_black,
        }
        results.append(result)

        print(f"--- Result of Game {episode + 1}: ---")
        print(result)
        print("-----")

        spinner.stop()

    del agent_white, agent_black
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    env.close()
    return results


def benchmark_agents(agent_path_list_1: List[str], agent_path_list_2: List[str] = None,
                     num_games: int = 100, max_turns: int = 400, results_file: str = None, enable_gui: bool = False):

    results = []

    if agent_path_list_2:
        agent_matchups = zip(agent_path_list_1, agent_path_list_2)
    else:
        agent_matchups = [(agent, agent_path_list_1[idx + 1]) for idx, agent in enumerate(agent_path_list_1[:-1])]

    for agent_path_1, agent_path_2 in agent_matchups:
        new_results = agent_vs_agent(white_agent_file_path=agent_path_1,
                                     black_agent_file_path=agent_path_2,
                                     max_turns=max_turns, episodes=num_games, enable_gui=enable_gui)

        results += new_results
        df = pd.DataFrame.from_records(results)
        df.to_excel(results_file)


if __name__ == "__main__":
    # self_play(AGENT_FILE_PATHS[0], MAX_TURNS, ENABLE_GUI, EPISODES)
    # agent_vs_agent(AGENT_FILE_PATHS[0], AGENT_FILE_PATHS[1], MAX_TURNS, ENABLE_GUI, GAMES_PER_BENCHMARK)
    benchmark_agents(AGENT_FILE_PATHS, None, GAMES_PER_BENCHMARK, MAX_TURNS, RESULTS_FILE, ENABLE_GUI)
