import math

import torch
from torch import nn
import torch.nn.functional as F

from .config import RainbowConfig


class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet.

    Attributes:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        std_init (float): initial std value
        weight_mu (nn.Parameter): mean value weight parameter
        weight_sigma (nn.Parameter): std value weight parameter
        bias_mu (nn.Parameter): mean value bias parameter
        bias_sigma (nn.Parameter): std value bias parameter
    """
    def __init__(
            self,
            in_features: int,
            out_features: int,
            std_init: float = 0.5,
    ):
        """Initialization."""
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        self.register_buffer(
            "weight_epsilon", torch.Tensor(out_features, in_features)
        )

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Reset trainable network parameters (factorized gaussian noise)."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )

    def reset_noise(self):
        """Make new noise."""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        # self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation.

        We don't use separate statements on train / eval mode.
        It doesn't show remarkable difference of performance.
        """
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )

    # @staticmethod
    def scale_noise(self, size: int) -> torch.Tensor:
        """Set scale to make noise (factorized gaussian noise)."""
        # get device (not fully exhaustive, my lead to errors)
        device = next(self.parameters()).device

        x = torch.empty(size, device=device).normal_()
        # x = torch.randn(size)

        return x.sign().mul(x.abs().sqrt())


class DQN(nn.Module):
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            hidden_dim: int,
            std_init: float,
            atom_size: int,
            support: torch.Tensor,
            feature_conf: RainbowConfig
    ):
        """Initialization."""
        super(DQN, self).__init__()

        self.feature_conf = feature_conf
        if self.feature_conf.noisy_net:
            deep_linear_class = NoisyLinear
            deep_linear_kwargs = {'std_init': std_init}
        else:
            deep_linear_class = nn.Linear
            deep_linear_kwargs = {}

        # set device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.support = support.to(self.device)
        self.out_dim = out_dim
        self.atom_size = atom_size

        # set common feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
        )

        if self.feature_conf.distributional_net:
            # set advantage layer
            self.advantage_hidden_layer = deep_linear_class(hidden_dim, hidden_dim, **deep_linear_kwargs)
            self.advantage_layer = deep_linear_class(hidden_dim, out_dim * atom_size, **deep_linear_kwargs)

            # set value layer
            self.value_hidden_layer = deep_linear_class(hidden_dim, hidden_dim, **deep_linear_kwargs)
            self.value_layer = deep_linear_class(hidden_dim, atom_size, **deep_linear_kwargs)

        else:
            # set advantage layer
            self.advantage_layer = nn.Sequential(deep_linear_class(hidden_dim, hidden_dim, **deep_linear_kwargs),
                                                 nn.ReLU(),
                                                 deep_linear_class(hidden_dim, out_dim, **deep_linear_kwargs))

            # set value layer
            self.value_layer = nn.Sequential(deep_linear_class(hidden_dim, hidden_dim, **deep_linear_kwargs),
                                             nn.ReLU(),
                                             deep_linear_class(hidden_dim, 1, **deep_linear_kwargs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""

        if self.feature_conf.distributional_net:
            dist = self.dist(x)
            q = torch.sum(dist * self.support.to(self.device), dim=2)
        else:
            feature = self.feature_layer(x)
            value = self.value_layer(feature)
            advantage = self.advantage_layer(feature)
            q = value + advantage - advantage.mean(dim=-1, keepdim=True)

        return q

    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """Get distribution for atoms."""
        feature = self.feature_layer(x)
        adv_hid = F.relu(self.advantage_hidden_layer(feature))
        val_hid = F.relu(self.value_hidden_layer(feature))

        advantage = self.advantage_layer(adv_hid).view(
            -1, self.out_dim, self.atom_size
        )
        value = self.value_layer(val_hid).view(-1, 1, self.atom_size)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)

        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans

        return dist

    def reset_noise(self):
        """Reset all noisy layers."""
        self.advantage_hidden_layer.reset_noise()
        self.advantage_layer.reset_noise()
        self.value_hidden_layer.reset_noise()
        self.value_layer.reset_noise()

    def to(self, device):
        self.device = device
        self.support.to(device)  # not sure if this is necessary
        return super(DQN, self).to(device)


