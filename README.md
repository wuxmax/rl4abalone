# Reinforcement Learning for Abalone

In this project reinforcement learning methods (namely Rainbow) are implemented for the game of [Abalone](https://en.wikipedia.org/wiki/Abalone_(board_game)).

It is very much based on the OpenAI Gym environment [gym-abalone](https://github.com/towzeur/gym-abalone) and on the [Rainbow is all you need](https://github.com/Curt-Park/rainbow-is-all-you-need) tutorial/implementation.

## Setup
Like always, it is preferred to use an virtual environment.

To install `gym-abalone`, run:
```
git clone git@github.com:towzeur/gym-abalone.git
cd gym-abalone
pip install -e .
```
TODO: Requirements for Rainbow

## Use
To test a trained rainbow agent, modify the `AGENT_FILE_PATH` variable accordingly and run `python test_rainbow-agent.py`
