# sac_custom/agent.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Optional, Union

from utils.config import Config
from .networks import GaussianPolicy, QNetwork


class SACAgent:
    """
    Soft Actor-Critic (SAC) agent.
    """

    def __init__(self, obs_dim: int, action_dim: int, config: Config):
        """
        Initialize SAC agent.

        Args:
            obs_dim: Observation dimension.
            action_dim: Action dimension.
            config: Configuration object with training and device parameters.
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config
        self.device = torch.device(config.device)

        # Hyperparameters
        self.gamma = config.training.gamma
        self.tau = config.training.tau
        self.lr = config.training.lr

        # Automatic entropy tuning
        self.automatic_alpha_tuning = config.training.alpha == 'auto'
        if self.automatic_alpha_tuning:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr)
            self.target_entropy = config.training.target_entropy
            if self.target_entropy == 'auto':
                self.target_entropy = -float(action_dim)
        else:
            self.alpha = float(config.training.alpha)
            self.log_alpha = None
            self.alpha_optimizer = None

        # Create networks
        self.actor = GaussianPolicy(obs_dim, action_dim).to(self.device)
        self.q1 = QNetwork(obs_dim, action_dim).to(self.device)
        self.q2 = QNetwork(obs_dim, action_dim).to(self.device)
        self.target_q1 = QNetwork(obs_dim, action_dim).to(self.device)
        self.target_q2 = QNetwork(obs_dim, action_dim).to(self.device)

        # Initialize target networks with same weights as Q networks
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.q_optimizer = optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=self.lr)

    @property
    def alpha(self) -> torch.Tensor:
        """Get current alpha value."""
        if self.automatic_alpha_tuning:
            return self.log_alpha.exp().detach()
        else:
            return torch.tensor(self._alpha, device=self.device)

    @alpha.setter
    def alpha(self, value: float):
        """Set alpha for non-automatic case."""
        if not self.automatic_alpha_tuning:
            self._alpha = value

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Select action given observation.

        Args:
            obs: Observation numpy array (obs_dim,).
            deterministic: Whether to return deterministic action.

        Returns:
            Action numpy array (action_dim,).
        """
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _ = self.actor.sample(obs_tensor, deterministic=deterministic, with_logprob=False)
        return action.cpu().numpy().flatten()

    def predict(self, obs, deterministic=True):
        """Alias for select_action with deterministic=True, for compatibility with evaluation."""
        return self.select_action(obs, deterministic=deterministic)

    def update(self, batch: Tuple[torch.Tensor, ...]) -> Dict[str, float]:
        """
        Perform one update step using a batch of transitions.

        Args:
            batch: Tuple of (obs, action, reward, next_obs, done) tensors.
                  All tensors are on the correct device and have shape (batch_size, dim).

        Returns:
            Dictionary of loss values for logging.
        """
        obs, action, reward, next_obs, done = batch

        # ---------------------------- Target Q computation ----------------------------
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_obs, deterministic=False, with_logprob=True)
            target_q1 = self.target_q1(next_obs, next_action)
            target_q2 = self.target_q2(next_obs, next_action)
            min_target_q = torch.min(target_q1, target_q2)
            # target = r + γ (1-d) (min_target_q - α * log π(a'|s'))
            target = reward + self.gamma * (1 - done) * (min_target_q - self.alpha * next_log_prob)

        # ---------------------------- Q network update ----------------------------
        current_q1 = self.q1(obs, action)
        current_q2 = self.q2(obs, action)

        q1_loss = F.mse_loss(current_q1, target)
        q2_loss = F.mse_loss(current_q2, target)
        q_loss = q1_loss + q2_loss

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # ---------------------------- Policy update ----------------------------
        # Sample new actions from current policy (reparameterized)
        new_action, log_prob = self.actor.sample(obs, deterministic=False, with_logprob=True)

        q1_new = self.q1(obs, new_action)
        q2_new = self.q2(obs, new_action)
        min_q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha * log_prob - min_q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ---------------------------- Temperature update ----------------------------
        alpha_loss = None
        if self.automatic_alpha_tuning:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        # ---------------------------- Target networks update ----------------------------
        self._soft_update(self.target_q1, self.q1, self.tau)
        self._soft_update(self.target_q2, self.q2, self.tau)

        # Logging
        log_dict = {
            'q_loss': q_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.alpha.item(),
        }
        if alpha_loss is not None:
            log_dict['alpha_loss'] = alpha_loss.item()

        return log_dict

    def _soft_update(self, target: nn.Module, source: nn.Module, tau: float):
        """Soft update target network parameters."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def save_checkpoint(self, path: str):
        """Save agent checkpoint."""
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'q1_state_dict': self.q1.state_dict(),
            'q2_state_dict': self.q2.state_dict(),
            'target_q1_state_dict': self.target_q1.state_dict(),
            'target_q2_state_dict': self.target_q2.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'q_optimizer_state_dict': self.q_optimizer.state_dict(),
            'config': self.config,
        }
        if self.automatic_alpha_tuning:
            checkpoint['log_alpha'] = self.log_alpha.detach().cpu().numpy()
            checkpoint['alpha_optimizer_state_dict'] = self.alpha_optimizer.state_dict()
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load agent checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.q1.load_state_dict(checkpoint['q1_state_dict'])
        self.q2.load_state_dict(checkpoint['q2_state_dict'])
        self.target_q1.load_state_dict(checkpoint['target_q1_state_dict'])
        self.target_q2.load_state_dict(checkpoint['target_q2_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.q_optimizer.load_state_dict(checkpoint['q_optimizer_state_dict'])
        if self.automatic_alpha_tuning and 'log_alpha' in checkpoint:
            self.log_alpha.data = torch.tensor(checkpoint['log_alpha'], device=self.device)
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])