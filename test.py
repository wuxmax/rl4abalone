from typing import Dict, List

import pandas as pd
import torch
import numpy as np
from halo import Halo

from gym_abalone.envs.abalone_env import AbaloneEnv
from agents.rainbow.agent import RainbowAgent
from agents.rainbow.config import RainbowConfig
from agents.random_agent import RandomAgent
from agents.agent import Agent
from utils import cvst, set_seeds, track_actions

AGENT_FILE_PATHS: List = ["trained-agents/rainbow-agent_12_16.pth", "random"]
# AGENT_FILE_PATHS: List = [None]
# GAMES_PER_BENCHMARK = 100
GAMES_PER_BENCHMARK = 1
MAX_TURNS: int = 400
ENABLE_GUI: bool = True
RESULTS_FILE = "results.csv"
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


def print_game_info(info: Dict, reward: int, score_white: int, score_black: int, unique_actions: List):
    if unique_actions:
        print(f"In the last 100 turns {info['player_name']} has made {unique_actions[0]} unique turns")

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
              last_actions_white: List, last_actions_black: List):

    action = agent.select_action(state)
    next_state, reward, done, info = agent.step(action)

    if info["player"] == 0:
        score_white += reward
        score_black -= reward
        print("whites turn, last actions:")
        print(last_actions_white)
        last_actions_white, unique_actions = track_actions(action=action, last_actions=last_actions_white,
                                                           unique_actions=[])
    else:
        score_black += reward
        score_white -= reward
        print("blacks turn, last actions:")
        print(last_actions_black)
        last_actions_black, unique_actions = track_actions(action=action, last_actions=last_actions_black,
                                                           unique_actions=[])

    print("got following unique_actions:")
    print(unique_actions)

    print_game_info(info=info, reward=reward, score_white=score_white, score_black=score_black,
                    unique_actions=unique_actions)

    if enable_gui:
        agent.env.render(fps=1)

    return next_state, score_white, score_black, done, last_actions_white, last_actions_black


def self_play(agent_file_path: str, max_turns: int = 400, enable_gui: bool = False, episodes: int = 1):
    """Test the agent."""
    env = AbaloneEnv(max_turns=max_turns)
    set_seeds(RANDOM_SEED, env)

    agent = load_agent(agent_file_path, env)
    agent.is_test = True

    last_actions_white = []
    last_actions_black = []

    for episode in range(episodes):
        state = cvst(env.reset(random_player=False))
        score_black = 0
        score_white = 0
        done = False

        spinner.start(text=f"Playing episode {episode + 1}/{episodes}")

        while not done:
            state, score_white, score_black, done, last_actions_white, last_actions_black =\
                test_step(agent=agent, state=state, score_white=score_white, score_black=score_black,
                          enable_gui=enable_gui, last_actions_white=last_actions_white,
                          last_actions_black=last_actions_black)

        spinner.stop()

    env.close()


def agent_vs_agent(white_agent_file_path: str, black_agent_file_path: str, max_turns: int = 400,
                   enable_gui: bool = False, episodes: int = 1):
    env = AbaloneEnv(max_turns=max_turns)
    set_seeds(RANDOM_SEED, env)

    print(f"loading .../{white_agent_file_path.split('/')[-1]}")
    agent1 = load_agent(white_agent_file_path, env)
    print(f"loading .../{black_agent_file_path.split('/')[-1]}")
    agent2 = load_agent(black_agent_file_path, env)
    agent1.is_test = True
    agent2.is_test = True

    score = [0, 0]
    last_actions_white = []
    last_actions_black = []

    for episode in range(episodes):
        state = cvst(env.reset(random_player=False))
        score_black = 0
        score_white = 0
        done = False

        spinner.start(text=f"Playing episode {episode + 1}/{episodes}")

        while not done:
            turn_player = agent1 if env.current_player % 2 == 0 else agent2
            state, score_white, score_black, done, last_actions_white, last_actions_black =\
                test_step(agent=turn_player, state=state, score_white=score_white, score_black=score_black,
                          enable_gui=enable_gui, last_actions_white=last_actions_white,
                          last_actions_black=last_actions_black)

        if score_white > score_black:
            score[0] += 1
        elif score_black > score_white:
            score[1] += 1
        spinner.stop()

    del agent1, agent2
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    env.close()
    return score[0], score[1]


def benchmark_agents(agent_path_list: List, num_games: int = 100, max_turns: int = 400, enable_gui: bool = False,
                     results_file: str = None):
    env = AbaloneEnv(max_turns=max_turns)
    set_seeds(RANDOM_SEED, env)
    shortened_agent_path_list = [agent_path.split('/')[-1] for agent_path in agent_path_list]
    scores = []

    for idx, agent_path_1 in enumerate(agent_path_list):
        score = []
        for idx_, agent_path_2 in enumerate(agent_path_list):
            if idx == idx_:
                score.append("-")
            else:
                score_agent_1, score_agent_2 = agent_vs_agent(white_agent_file_path=agent_path_1,
                                                              black_agent_file_path=agent_path_2,
                                                              max_turns=max_turns, enable_gui=enable_gui,
                                                              episodes=num_games)
                score.append((score_agent_1, score_agent_2))
        scores.append(score)

    df = pd.DataFrame(scores, columns=shortened_agent_path_list, index=shortened_agent_path_list)
    df.to_excel(results_file)



if __name__ == "__main__":
    # self_play(AGENT_FILE_PATHS[0], MAX_TURNS, ENABLE_GUI, EPISODES)
    agent_vs_agent(AGENT_FILE_PATHS[0], AGENT_FILE_PATHS[1], MAX_TURNS, ENABLE_GUI, GAMES_PER_BENCHMARK)
    # benchmark_agents(AGENT_FILE_PATHS, GAMES_PER_BENCHMARK, MAX_TURNS, ENABLE_GUI, RESULTS_FILE)
