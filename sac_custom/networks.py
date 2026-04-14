import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class GaussianPolicy(nn.Module):
    """
    Gaussian policy network for SAC.
    Maps observations to a Gaussian distribution over actions (with tanh squashing).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Shared network
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Separate output layers for mean and log_std
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=1.0)
                nn.init.constant_(layer.bias, 0.0)

        # Mean layer: small init (0.01) to start with near-zero actions
        nn.init.xavier_uniform_(self.mean_layer.weight, gain=0.01)
        nn.init.constant_(self.mean_layer.bias, 0.0)

        # Log std layer: init to zero (std=1) after clipping will be in range
        nn.init.xavier_uniform_(self.log_std_layer.weight, gain=0.01)
        nn.init.constant_(self.log_std_layer.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to compute mean and log_std.

        Args:
            obs: Observation tensor of shape (batch, obs_dim).

        Returns:
            mean: Mean of the Gaussian, shape (batch, action_dim).
            log_std: Log standard deviation, clipped, shape (batch, action_dim).
        """
        x = self.net(obs)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        with_logprob: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Sample actions from the policy.

        Args:
            obs: Observation tensor (batch, obs_dim).
            deterministic: If True, return mean action (no sampling).
            with_logprob: If True and not deterministic, return log probability.

        Returns:
            action: Sampled action, shape (batch, action_dim).
            log_prob: Log probability of the action, shape (batch, 1) or None.
        """
        mean, log_std = self.forward(obs)

        if deterministic:
            return torch.tanh(mean), None

        std = log_std.exp()
        eps = torch.randn_like(std)
        u = mean + std * eps  # reparameterization
        action = torch.tanh(u)

        if not with_logprob:
            return action, None

        # Compute log probability of u
        log_prob_u = (
            -0.5 * ((u - mean) / std).pow(2)
            - 0.5 * np.log(2 * np.pi)
            - log_std
        ).sum(dim=-1, keepdim=True)  # (batch, 1)

        # Correction for tanh squashing
        log_prob = log_prob_u - torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

        return action, log_prob

    def log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of given actions under the current policy.

        Args:
            obs: Observation tensor (batch, obs_dim).
            action: Action tensor (batch, action_dim) already after tanh.

        Returns:
            log_prob: Log probability, shape (batch, 1).
        """
        # Inverse tanh: u = arctanh(action)
        # Clamp to avoid numerical issues
        u = torch.atanh(torch.clamp(action, -1.0 + 1e-6, 1.0 - 1e-6))
        mean, log_std = self.forward(obs)
        std = log_std.exp()

        # Log probability of u
        log_prob_u = (
            -0.5 * ((u - mean) / std).pow(2)
            - 0.5 * np.log(2 * np.pi)
            - log_std
        ).sum(dim=-1, keepdim=True)

        # Correction for tanh
        log_prob = log_prob_u - torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        return log_prob


class QNetwork(nn.Module):
    """
    Q-network for SAC (critic).
    Maps (obs, action) to a single Q-value.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=1.0)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute Q-value.

        Args:
            obs: Observation tensor (batch, obs_dim).
            action: Action tensor (batch, action_dim).

        Returns:
            Q-value tensor of shape (batch, 1).
        """
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)
