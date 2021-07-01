import random
import time
import gym
import torch
import numpy as np

AGENT_FILE_PATH: str = "rainbow-agent.pth"

def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

env = gym.make("abalone-v0")

seed = random.randint(1, 20000)

np.random.seed(seed)
random.seed(seed)
seed_torch(seed)
env.seed(seed)

global agent
global state
global done
global score
global turn

with open(AGENT_FILE_PATH, "rb") as f:
        agent = torch.load(f, map_location='cpu')
        agent.reset_torch_device()

agent.is_test = True
state = agent.cvst(env.reset(random_player=False), 0)
done = False
score = 0
turn = 0

def agent_game(dt, gui):
        # pos0, pos1 = Agents.choice_random(gui.game)
        pos0, pos1 = agent.train()
        print(pos0, pos1)
        # highlight the starting pos
        gui.action(pos0)
        move_type = gui.action(pos1)
        print(f"{gui.game.turns_count - 1: <4} {move_type: >14}")

while not done:
    if turn % 2 == 0:
        action = agent.select_action(state)
        next_state, reward, done, info = agent.step(action, turn)

        score += reward

        turn += 1
        state = next_state

        print(f"{info['turn']: <4} | {info['player_name']} | {str(info['move_type']): >16} | reward={reward: >4}")
        agent.env.render(fps=0.5)
    else:
        print("You have 15 seconds left for your turn")
        time.sleep(5)
        print("You have 10 seconds left for your turn")
        time.sleep(5)
        print("You have 5 seconds left for your turn")
        time.sleep(5)

        turn += 1
        state = agent.cvst(env.observation)

print(f"Testing completed. Score: {score}")
agent.env.close()