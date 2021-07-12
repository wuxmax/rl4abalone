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
from utils import set_seeds, cvst

# AGENT_FILE_PATHS: List = ["rainbow-agent.pth"]
AGENT_FILE_PATHS: List = [None]
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


def test_step(agent: Agent, state: np.ndarray, turn: int, score_white: int, score_black: int, enable_gui: bool,
              env: AbaloneEnv):
    action = agent.select_action(state)
    next_state, reward, done, info = agent.step(action, turn)

    if turn % 2 == 0:
        score_white += reward
    else:
        score_black += reward

    print_game_info(info=info, reward=reward, score_white=score_white, score_black=score_black)

    if enable_gui:
        env.render(fps=1)

    return next_state, score_white, score_black, turn + 1, done


def self_play(agent_file_path: str, max_turns: int = 400, enable_gui: bool = False, episodes: int = 1):
    """Test the agent."""
    env = AbaloneEnv(max_turns=max_turns)
    set_seeds(RANDOM_SEED, env)

    agent = load_agent(agent_file_path, env)
    agent.is_test = True

    for episode in range(episodes):
        state = cvst(env.reset(random_player=False), 0)
        score_black = 0
        score_white = 0
        turn = 0
        done = False

        spinner.start(text=f"Playing episode {episode + 1}/{episodes}")

        while not done:
            state, score_white, score_black, turn, done = test_step(agent=agent, state=state, turn=turn,
                                                                    score_white=score_white, score_black=score_black,
                                                                    enable_gui=enable_gui, env=env)

        spinner.stop()

    env.close()


def agent_vs_agent(white_agent_file_path: str, black_agent_file_path: str, max_turns: int = 400,
                   enable_gui: bool = False, episodes: int = 1):
    env = AbaloneEnv(max_turns=max_turns)
    set_seeds(RANDOM_SEED, env)

    agent1 = load_agent(white_agent_file_path, env)
    agent2 = load_agent(black_agent_file_path, env)
    agent1.is_test = True
    agent2.is_test = True
    score = [0, 0]

    for episode in range(episodes):
        state = cvst(env.reset(random_player=False), 0)
        score_black = 0
        score_white = 0
        turn = 0
        done = False

        spinner.start(text=f"Playing episode {episode + 1}/{episodes}")

        while not done:
            turn_player = agent1 if turn % 2 == 0 else agent2
            state, score_white, score_black, turn, done = test_step(agent=turn_player, state=state, turn=turn,
                                                                    score_white=score_white, score_black=score_black,
                                                                    enable_gui=enable_gui, env=env)

        score[(env.current_player + 1) % 2] += 1
        spinner.stop()

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
            if idx < idx_:
                score.append(scores[idx][idx_][::-1])
            elif idx == idx_:
                score.append("-")
            else:
                score.append(agent_vs_agent(white_agent_file_path=agent_path_1, black_agent_file_path=agent_path_2,
                                       max_turns=max_turns, enable_gui=enable_gui, episodes=num_games))
        scores.append(score)

    df = pd.DataFrame(scores, colums=shortened_agent_path_list, index=shortened_agent_path_list)
    df.to_excel(results_file)



if __name__ == "__main__":
    # self_play(AGENT_FILE_PATHS[0], MAX_TURNS, ENABLE_GUI, EPISODES)
    # agent_vs_agent(AGENT_FILE_PATHS[0], AGENT_FILE_PATHS[1], MAX_TURNS, ENABLE_GUI, EPISODES)
    benchmark_agents(AGENT_FILE_PATHS, GAMES_PER_BENCHMARK, MAX_TURNS, ENABLE_GUI, RESULTS_FILE)
