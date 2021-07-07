from typing import Dict, List, Tuple

import numpy as np
import gym
import torch
from torch import min, optim
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from .config import RainbowConfig
from .buffer import ReplayBuffer, PrioritizedReplayBuffer
from .network import DQN
from .utils import _plot
from utils import cvst, cvact, next_player
from agents.agent import Agent


class RainbowAgent(Agent):
    """Rainbow Agent interacting with environment.

    Attribute:
        env (gym.Env): openAI Gym environment
        memory (PrioritizedReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        target_update (int): period for target model's hard update
        gamma (float): discount factor
        dqn (Network): model to train and select actions
        dqn_target (Network): target model to update
        optimizer (torch.optim): optimizer for training dqn
        transition (list): transition information including
                           state, action, reward, next_state, done
        v_min (float): min value of support
        v_max (float): max value of support
        atom_size (int): the unit number of support
        support (torch.Tensor): support for categorical dqn
        use_n_step (bool): whether to use n_step memory
        n_step (int): step number to calculate n-step td error
        memory_n (ReplayBuffer): n-step replay buffer
    """

    def __init__(
            self,
            env: gym.Env,
            memory_size: int,
            batch_size: int,
            target_update: int,
            gamma: float = 0.99,
            hidden_dim: int = 512,
            # PER parameters
            alpha: float = 0.5,
            beta: float = 0.4,
            prior_eps: float = 1e-6,
            # Epsilon-greedy parameters
            epsilon_decay: float = 1 / 2000,  # taken from RIAYN | 250K frames in RP
            max_epsilon: float = 1.0,
            min_epsilon: float = 0.01,
            # Categorical DQN parameters
            v_min: float = -10,
            v_max: float = 10,
            atom_size: int = 51,
            # N-step Learning
            n_step: int = 3,
            # Toggle rainbow features
            feature_conf: RainbowConfig = RainbowConfig()
    ):
        """Initialization.

        Args:
            env (gym.Env): openAI Gym environment
            memory_size (int): length of memory
            batch_size (int): batch size for sampling
            target_update (int): period for target model's hard update
            lr (float): learning rate
            gamma (float): discount factor
            alpha (float): determines how much prioritization is used
            beta (float): determines how much importance sampling is used
            prior_eps (float): guarantees every transition can be sampled
            v_min (float): min value of support
            v_max (float): max value of support
            atom_size (int): the unit number of support
            n_step (int): step number to calculate n-step td error
        """
        super().__init__(env)

        self.feature_conf = feature_conf
        self.trigger_states = ["winner"]

        obs_dim = 121 * 3
        action_dim = 61 * 61
        # obs_dim = env.observation_space.shape[0]
        # action_dim = env.action_space.n

        self.batch_size = batch_size
        self.target_update = target_update
        self.gamma = gamma

        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(self.device)

        # Epsilon-greedy (if no NoisyNet)
        if not feature_conf.noisy_net:
            self.epsilon = max_epsilon
            self.epsilon_decay = epsilon_decay
            self.max_epsilon = max_epsilon
            self.min_epsilon = min_epsilon

        # PER
        # memory for 1-step Learning
        self.beta = beta
        self.prior_eps = prior_eps
        self.memory = PrioritizedReplayBuffer(
            obs_dim, memory_size, batch_size, alpha=alpha
        )

        # memory for N-step Learning
        self.use_n_step = True if n_step > 1 else False
        if self.use_n_step:
            self.n_step = n_step
            self.memory_n = ReplayBuffer(
                obs_dim, memory_size, batch_size, n_step=n_step, gamma=gamma
            )

        # Categorical DQN parameters
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support = torch.linspace(
            self.v_min, self.v_max, self.atom_size
        ).to(self.device)

        # networks: dqn, dqn_target
        self.dqn = DQN(
            obs_dim, action_dim, hidden_dim, self.atom_size, self.support,
            self.feature_conf).to(self.device)
        self.dqn_target = DQN(
            obs_dim, action_dim, hidden_dim, self.atom_size, self.support,
            self.feature_conf).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters())

        # transition to store in memory
        self.transition = list()

        # mode: train / test
        self.is_test = False

    def _get_dqn_action(self, state: np.ndarray):
        action_probs = self.dqn(torch.FloatTensor(state).to(self.device)).detach().cpu().numpy()
        if min(action_probs) < 0:
            action_probs = action_probs + min(action_probs)
        action_probs_masked = action_probs * self.env.get_action_mask()
        return action_probs_masked.argmax()

    # expects an 'get_action_mask' function as provided by gym_abalone
    # see:  https://github.com/towzeur/gym-abalone/blob/048e443101c29bbc20ff6646beeca92c5776b1e7/
    #       gym_abalone/envs/abalone_env.py#L149
    def select_action(self, state: np.ndarray) -> Tuple[int, int]:
        """Select an action from the input state."""
        # if no NoisyNet: epsilon greedy policy
        if not self.is_test and not self.feature_conf.noisy_net and self.epsilon > np.random.random():
            # if no NoisyNet: epsilon greedy policy
            possible_actions = np.flatnonzero(self.env.get_action_mask())
            selected_action = np.random.choice(possible_actions)
        else:
            selected_action = self._get_dqn_action(state)

        if not self.is_test:
            self.transition = [state, selected_action]

        return cvact(selected_action)

    def step(self, action: Tuple[int, int], turn: int) -> Tuple[np.ndarray, np.float64, bool, Dict]:
        """Take an action and return the response of the env, where the state is already in 121x3 representation"""
        next_state, reward, done, info = self.env.step(action)
        next_state = cvst(next_state, turn + 1)

        if not self.is_test:
            self.add_custom_transition(self.transition + [reward, next_state, done])

        return next_state, reward, done, info

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        # PER needs beta to calculate weights
        samples = self.memory.sample_batch(self.beta)
        weights = torch.FloatTensor(
            samples["weights"].reshape(-1, 1)
        ).to(self.device)
        indices = samples["indices"]

        # 1-step Learning loss
        elementwise_loss = self._compute_dqn_loss(samples, self.gamma)

        # PER: importance sampling before average
        loss = torch.mean(elementwise_loss * weights)

        # N-step Learning loss
        # we are gonna combine 1-step loss and n-step loss so as to
        # prevent high-variance. The original rainbow employs n-step loss only.
        if self.use_n_step:
            gamma = self.gamma ** self.n_step
            samples = self.memory_n.sample_batch_from_idxs(indices)
            elementwise_loss_n_loss = self._compute_dqn_loss(samples, gamma)
            elementwise_loss += elementwise_loss_n_loss

            # PER: importance sampling before average
            loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.dqn.parameters(), 10.0)
        self.optimizer.step()

        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)

        # NoisyNet: reset noise
        if self.feature_conf.noisy_net:
            self.dqn.reset_noise()
            self.dqn_target.reset_noise()

        return loss.item()

    def _decrease_epsilon(self):
        self.epsilon = max(
            self.min_epsilon, self.epsilon - (
                    self.max_epsilon - self.min_epsilon
            ) * self.epsilon_decay
        )

    def train(self, num_turns_total: int, plotting_interval: int = 200):
        """Train the agent."""
        self.is_test = False

        state = cvst(self.env.reset(random_player=False), 0)
        update_cnt = 0
        losses = []
        scores = []
        score_black = 0
        score_white = 0
        turn_game = 0
        last_opposing_player_transition = list()

        for turn_total_idx in tqdm(range(1, num_turns_total + 1)):
            action = self.select_action(state)

            print("first assert")
            print(f"turn_game + 1: {turn_game + 1}, self.env.turns: {self.env.turns}")
            assert turn_game + 1 == self.env.turns

            next_state, reward, done, info = self.step(action, turn_game)

            if turn_game % 2 == 0:
                score_white += reward
            else:
                score_black += reward

            score_black, score_white = self.handle_trigger_states(score_black=score_black, score_white=score_white,
                                                                  reward=reward, info=info,
                                                                  trigger_states=self.trigger_states,
                                                                  last_opposing_player_transition=
                                                                  last_opposing_player_transition)

            turn_game += 1

            print("second assert")
            print(f"turn_game + 1: {turn_game + 1}, self.env.turns: {self.env.turns}")
            assert turn_game + 1 == self.env.turns

            last_opposing_player_transition = self.transition
            state = next_state

            # PER: increase beta
            fraction = min(turn_total_idx / num_turns_total, 1.0)
            self.beta = self.beta + fraction * (1.0 - self.beta)

            # if no NoisyNet: linearly decrease epsilon
            if not self.feature_conf.noisy_net:
                self._decrease_epsilon()

            # if episode ends
            if done:
                state = cvst(self.env.reset(random_player=False), 0)
                turn_game = 0
                scores.append(max(score_black, score_white))
                score_black = 0
                score_white = 0

            # if training is ready
            if len(self.memory) >= self.batch_size:
                loss = self.update_model()
                losses.append(loss)
                update_cnt += 1

                # if hard update is needed
                if update_cnt % self.target_update == 0:
                    self._target_hard_update()

            # plotting
            if turn_total_idx % plotting_interval == 0:
                _plot(turn_total_idx, scores, losses)

        self.env.close()

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray], gamma: float) -> torch.Tensor:
        """Return categorical dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            # Double DQN
            next_action = self.dqn(next_state).argmax(1)
            next_dist = self.dqn_target.dist(next_state)
            next_dist = next_dist[range(self.batch_size), next_action]

            t_z = reward + (1 - done) * gamma * self.support.to(self.device)
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.atom_size, self.batch_size
                ).long()
                    .unsqueeze(1)
                    .expand(self.batch_size, self.atom_size)
                    .to(self.device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.dqn.dist(state)
        log_p = torch.log(dist[range(self.batch_size), action])
        elementwise_loss = -(proj_dist * log_p).sum(1)

        return elementwise_loss

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())

    def reset_torch_device(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"Resetting torch devices to {self.device}...")
        self.support.to(self.device)
        self.dqn.to(self.device)
        self.dqn_target.to(self.device)

        if torch.cuda.is_available():
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
        print(f"Done!")

    def add_custom_transition(self, transition, reward=None):
        if not self.is_test:
            if reward:
                self.transition = transition[:2] + [reward] + transition[3:]
            else:
                self.transition = transition

            # N-step transition
            if self.use_n_step:
                one_step_transition = self.memory_n.store(*self.transition)
            # 1-step transition
            else:
                one_step_transition = self.transition

            # add a single step transition
            if one_step_transition:
                self.memory.store(*one_step_transition)

    def handle_trigger_states(self, score_black: int, score_white: int, reward: int, info: Dict,
                              trigger_states: List, last_opposing_player_transition):
        for move in trigger_states:
            if str(info['move_type']) == move:
                # score update depending on negative reward
                score_black -= reward if info['player_name'] == 'white' else 0
                score_white -= reward if info['player_name'] == 'black' else 0

                # saving the negative reward from defeat into replay buffer
                self.add_custom_transition(last_opposing_player_transition, reward=-reward)

                # print information about the triggering move
                if str(info['move_type']) == "winner":
                    print(f"\n{info['player_name']} won in {info['turn']: <4} turns with a total score of "
                          f"{score_black if info['player_name'] == 'black' else score_white}!\n The looser scored with "
                          f"{score_black if info['player_name'] == 'white' else score_white}!")
                else:
                    print(f"\n{info['turn']: <4} | {info['player_name']} | {str(info['move_type']): >16}"
                          f" | reward={reward: >4}")

        return score_black, score_white
