import gym


class Agent:
    def __init__(
            self,
            env: gym.Env
    ):
        raise NotImplementedError

    def select_action(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError
