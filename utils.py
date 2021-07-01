import random
import torch

import numpy as np


def set_seeds(seed, env):
    np.random.seed(seed)
    random.seed(seed)
    env.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
