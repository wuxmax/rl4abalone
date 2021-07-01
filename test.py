import random
import torch
import numpy as np
from halo import Halo

from gym_abalone.envs.abalone_env import AbaloneEnv

from rainbow_module.agent import RainbowAgent
from rainbow_module.config import RainbowConfig
from rainbow import DQNAgent
from utils import set_seeds

# AGENT_FILE_PATH_1: str = "rainbow-agent.pth"  # used for self play
AGENT_FILE_PATH_1: str = None  # used for self play
AGENT_FILE_PATH_2: str = "rainbow-agent.pth"
MAX_TURNS: int = 9999
ENABLE_GUI: bool = False
EPISODES: int = 1

spinner = Halo(spinner='dots')


def self_play(agent_file_path: str, max_turns: int = 400, enable_gui: bool = False, episodes: int = 1):
    """Test the agent."""
    env = AbaloneEnv(max_turns=max_turns)
    set_seeds(777, env)

    if agent_file_path:
        with open(agent_file_path, "rb") as f:
            agent = torch.load(f, map_location=torch.device('cpu'))
            agent.reset_torch_device()
    else:
        # num_frames = 2000
        memory_size = 1000
        batch_size = 128
        target_update = 100
        config = RainbowConfig()
        agent = RainbowAgent(env, memory_size, batch_size, target_update, feature_conf=config)
        # agent = DQNAgent(env, memory_size, batch_size, target_update)

    agent.env = env
    agent.is_test = False

    for episode in range(episodes):
        state = agent.cvst(env.reset(random_player=False), 0)
        score_black = 0
        score_white = 0
        turn = 0
        done = False

        spinner.start(text=f"Playing episode {episode + 1}/{episodes}")

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = agent.step(action, turn)

            if turn % 2 == 0:
                score_white += reward
            else:
                score_black += reward

            if str(info['move_type']) == "ejected":
                print(f"\n{info['turn']: <4} | {info['player_name']} | {str(info['move_type']): >16} "
                      f"| reward={reward: >4}")
            elif str(info['move_type']) == "winner":
                # score update depending on defeat
                score_black -= 1 if info['player_name'] == 'white' else 0
                score_white -= 1 if info['player_name'] == 'black' else 0
                print(
                    f"\n{info['player_name']} won in {info['turn']: <4} turns with a total score of "
                    f"{score_black if info['player_name'] == 'black' else score_white}!\n"
                    f"The looser scored with {score_black if info['player_name'] == 'white' else score_white}!")

            if enable_gui:
                env.render(fps=1)

            turn += 1
            state = next_state

        spinner.stop()

    env.close()


def agent_vs_agent(white_agent_file_path: str, black_agent_file_path: str, max_turns: int = 400,
                   enable_gui: bool = False, episodes: int = 1):
    with open(white_agent_file_path, "rb") as f:
        agent1 = torch.load(f, map_location=torch.device('cpu'))
        agent1.reset_torch_device()

    with open(black_agent_file_path, "rb") as f:
        agent2 = torch.load(f, map_location=torch.device('cpu'))
        agent2.reset_torch_device()

    env = AbaloneEnv(max_turns=max_turns)
    agent1.env = env
    agent2.env = env
    agent1.is_test = False
    agent2.is_test = False
    reset_seeds(777, env)

    for episode in range(episodes):
        state = agent1.cvst(env.reset(random_player=False), 0)
        score_black = 0
        score_white = 0
        turn = 0
        done = False

        while not done:
            turn_player = agent1 if turn % 2 == 0 else agent2
            action = turn_player.select_action(state)
            next_state, reward, done, info = turn_player.step(action, turn)

            if turn % 2 == 0:
                score_white += reward
            else:
                score_black += reward

            if str(info['move_type']) == "ejected":
                print(f"\n{info['turn']: <4} | {info['player_name']} | {str(info['move_type']): >16} "
                      f"| reward={reward: >4}")
            elif str(info['move_type']) == "winner":
                # score update depending on defeat
                score_black -= 1 if info['player_name'] == 'white' else 0
                score_white -= 1 if info['player_name'] == 'black' else 0
                print(
                    f"\n{info['player_name']} won in {info['turn']: <4} turns with a total score of "
                    f"{score_black if info['player_name'] == 'black' else score_white}!\n"
                    f"The looser scored with {score_black if info['player_name'] == 'white' else score_white}!")

            if enable_gui:
                env.render(fps=1)

            turn += 1
            state = next_state

    env.close()


if __name__ == "__main__":
    self_play(AGENT_FILE_PATH_1, MAX_TURNS, ENABLE_GUI, EPISODES)
    # agent_vs_agent(AGENT_FILE_PATH_1, AGENT_FILE_PATH_2, MAX_TURNS, ENABLE_GUI, EPISODES)
