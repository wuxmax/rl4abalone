
class RainbowConfig:
    # noisy_net: bool
    # distributional_net: bool

    def __init__(
        self,
        noisy_net: bool = True,
        distributional_net: bool = True,
    ):
        self.noisy_net = noisy_net
        self.distributional_net = distributional_net


# reset noise
# reset parameters