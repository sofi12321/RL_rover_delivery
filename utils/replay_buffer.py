import numpy as np
import torch
from typing import Tuple, Union


class ReplayBuffer:
    """
    Circular replay buffer for storing transitions (obs, action, reward, next_obs, done).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        capacity: int,
        device: Union[str, torch.device] = "cpu",
    ):
        """
        Initialize the replay buffer.

        Args:
            obs_dim: Dimension of observations.
            action_dim: Dimension of actions.
            capacity: Maximum number of transitions to store.
            device: Torch device to which sampled batches will be moved.
        """
        self.capacity = capacity
        self.device = torch.device(device)

        # Preallocate numpy arrays
        self.obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action_buf = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward_buf = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done_buf = np.zeros((capacity, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def push(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: Union[float, np.ndarray],
        next_obs: np.ndarray,
        done: Union[bool, float, np.ndarray],
    ) -> None:
        """
        Add a transition to the buffer.

        Args:
            obs: Current observation (shape: [obs_dim]).
            action: Action taken (shape: [action_dim]).
            reward: Reward received (scalar).
            next_obs: Next observation (shape: [obs_dim]).
            done: Whether the episode terminated (bool or float 0/1).
        """
        # Convert inputs to appropriate numpy types
        obs = np.asarray(obs, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        reward = float(reward)
        next_obs = np.asarray(next_obs, dtype=np.float32)
        done = float(done)  # bool will be converted to 1.0/0.0

        # Store at current pointer
        self.obs_buf[self.ptr] = obs
        self.action_buf[self.ptr] = action
        self.reward_buf[self.ptr] = reward
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done

        # Advance pointer
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Tuple of tensors (obs, action, reward, next_obs, done).
            Each tensor has shape (batch_size, dim) except reward and done which have
            shape (batch_size, 1). All tensors are on the specified device.
        """
        if self.size < batch_size:
            raise ValueError(
                f"Not enough transitions in buffer (have {self.size}, need {batch_size})"
            )

        # Random indices without replacement
        indices = np.random.choice(self.size, batch_size, replace=False)

        # Extract batches
        obs = torch.from_numpy(self.obs_buf[indices]).to(self.device)
        action = torch.from_numpy(self.action_buf[indices]).to(self.device)
        reward = torch.from_numpy(self.reward_buf[indices]).to(self.device)
        next_obs = torch.from_numpy(self.next_obs_buf[indices]).to(self.device)
        done = torch.from_numpy(self.done_buf[indices]).to(self.device)

        return obs, action, reward, next_obs, done

    def __len__(self) -> int:
        """Return the current number of stored transitions."""
        return self.size
