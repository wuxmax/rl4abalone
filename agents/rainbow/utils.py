import sys
from typing import List

import numpy as np
import torch
from matplotlib import pyplot as plt

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from IPython.core.display import clear_output


def _plot(
        frame_idx: int,
        scores: List[float],
        losses: List[torch.Tensor],
        unique_actions_white: List,
        unique_actions_black: List
):
    """Plot the training progresses."""
    if IN_COLAB:
        clear_output(True)
    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.title('turn %s. mean score (last 10 runs): %s' % (frame_idx, np.mean(scores[-100:])))
    plt.plot(scores)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.subplot(133)
    plt.title('unique actions in 100 actions')
    plt.plot(unique_actions_white, color="grey")
    plt.plot(unique_actions_black, color="black")
    # axes = plt.gca()
    # axes.set_xlim([max(0, len(losses) - 300), len(losses)])
    # if len(losses) > 100:
    #     axes.set_ylim([-0.1, 1])
    plt.show()
