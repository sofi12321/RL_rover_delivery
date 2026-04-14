# baseline/baseline_agent.py

import numpy as np
import gymnasium as gym
from typing import Union

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from utils.config import Config
from env.rover_env import RoversEnv


class BaselineSACAgent:
    """
    Adapter for stable-baselines3 SAC implementation.
    Provides a consistent interface with our custom SAC agent.
    """

    def __init__(self, env: RoversEnv, config: Config):
        """
        Initialize the baseline SAC agent.

        Args:
            env: The RoversEnv instance (non-vectorized).
            config: Configuration object.
        """
        self.config = config
        self.env = env

        # Create a vectorized environment (DummyVecEnv expects a list of callables)
        def make_env():
            return env

        self.vec_env = make_vec_env(make_env, n_envs=1)

        # Extract training parameters
        training_cfg = config.training

        # Handle entropy coefficient (alpha)
        if training_cfg.alpha == 'auto':
            ent_coef = 'auto'
        else:
            ent_coef = float(training_cfg.alpha)

        # Handle target entropy
        if training_cfg.target_entropy == 'auto':
            target_entropy = 'auto'
        else:
            target_entropy = float(training_cfg.target_entropy)

        # Create SAC model
        self.model = SAC(
            policy='MlpPolicy',
            env=self.vec_env,
            learning_rate=training_cfg.lr,
            buffer_size=int(training_cfg.buffer_size),
            batch_size=int(training_cfg.batch_size),
            gamma=training_cfg.gamma,
            tau=training_cfg.tau,
            ent_coef=ent_coef,
            target_entropy=target_entropy,
            # net_arch=[256, 256], # default
            verbose=0,
            device=config.device,
        )

    def learn(self, total_timesteps: int, callback=None) -> None:
        """
        Train the agent for a given number of timesteps.

        Args:
            total_timesteps: Number of environment steps to train for.
        """
        self.model.learn(total_timesteps=total_timesteps, callback=callback)

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Predict action for a single observation.

        Args:
            obs: Observation array of shape (obs_dim,).
            deterministic: Whether to return deterministic action.

        Returns:
            Action array of shape (action_dim,).
        """
        # Add batch dimension
        obs_batch = obs.reshape(1, -1)
        action, _ = self.model.predict(obs_batch, deterministic=deterministic)
        return action.flatten()

    def save(self, path: str) -> None:
        """
        Save the model to disk.

        Args:
            path: Path where the model will be saved.
        """
        self.model.save(path)

    def load(self, path: str) -> None:
        """
        Load a saved model.

        Args:
            path: Path to the saved model.
        """
        # When loading, we need to provide the environment again
        self.model = SAC.load(path, env=self.vec_env)