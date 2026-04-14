# training/train_custom.py

import os
import csv
import torch
import numpy as np
from typing import Optional, Callable, Dict, Any

from env.rover_env import RoversEnv
from sac_custom.agent import SACAgent
from utils.config import Config
from utils.replay_buffer import ReplayBuffer
from evaluation.evaluate import evaluate_agent  # будет реализовано позже


def train_custom(
    env: RoversEnv,
    agent: SACAgent,
    config: Config,
    log_dir: str,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
) -> None:
    """
    Train a custom SAC agent.

    Args:
        env: The environment instance.
        agent: The SACAgent instance.
        config: Configuration object.
        log_dir: Directory to save logs and checkpoints.
        callback: Optional callback function called after each evaluation.
                  Signature: callback(locals, globals)
    """
    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    models_dir = os.path.join(log_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Extract dimensions
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Create replay buffer
    replay_buffer = ReplayBuffer(
        obs_dim=obs_dim,
        action_dim=action_dim,
        capacity=config.training.buffer_size,
        device=config.device,
    )

    # Training parameters
    total_timesteps = config.training.total_timesteps
    batch_size = config.training.batch_size
    eval_freq = config.training.eval_freq
    eval_episodes = config.training.eval_episodes

    # Logging setup
    csv_path = os.path.join(log_dir, "progress.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "step",
                "avg_reward",
                "success_rate",
                "avg_length",
                "q_loss",
                "actor_loss",
                "alpha",
            ]
        )

    # Initialize episode tracking
    obs, _ = env.reset()
    episode_reward = 0.0
    episode_length = 0
    episode_num = 0

    # For logging losses (we'll store last values)
    last_q_loss = 0.0
    last_actor_loss = 0.0
    last_alpha = agent.alpha.item()

    # Main training loop
    for timestep in range(total_timesteps):
        # Select action
        action = agent.select_action(obs, deterministic=False)

        # Step environment
        next_obs, reward, done, truncated, info = env.step(action)

        # Store transition (done is True only for terminal states, truncated is not terminal)
        replay_buffer.push(obs, action, reward, next_obs, done)

        # Update episode stats
        episode_reward += reward
        episode_length += 1

        # Check if episode ended
        if done or truncated:
            # Log episode stats (optional, not in CSV)
            # Reset environment
            obs, _ = env.reset()
            episode_reward = 0.0
            episode_length = 0
            episode_num += 1
        else:
            obs = next_obs

        # Update agent if enough samples
        if len(replay_buffer) >= batch_size:
            batch = replay_buffer.sample(batch_size)
            log_dict = agent.update(batch)
            last_q_loss = log_dict.get("q_loss", last_q_loss)
            last_actor_loss = log_dict.get("actor_loss", last_actor_loss)
            last_alpha = log_dict.get("alpha", last_alpha)

        # Evaluation and logging
        if (timestep + 1) % eval_freq == 0 or timestep == total_timesteps - 1:
            # Create a fresh environment for evaluation
            eval_env = RoversEnv(config)

            # Evaluate agent
            eval_stats = evaluate_agent(
                env=eval_env,
                agent=agent,
                num_episodes=eval_episodes,
                deterministic=True,
            )

	    
            print(f"Step {timestep+1}: avg_reward={eval_stats['avg_reward']:.2f}, "
                      f"success_rate={eval_stats['success_rate']:.2f}")

            # Save checkpoint
            checkpoint_path = os.path.join(models_dir, f"step_{timestep+1}.pt")
            agent.save_checkpoint(checkpoint_path)

            # Log to CSV
            with open(csv_path, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(
                    [
                        timestep + 1,
                        eval_stats.get("avg_reward", 0.0),
                        eval_stats.get("success_rate", 0.0),
                        eval_stats.get("avg_length", 0.0),
                        last_q_loss,
                        last_actor_loss,
                        last_alpha,
                    ]
                )

            # Callback
            if callback is not None:
                callback(locals(), globals())

    # Save final model
    final_path = os.path.join(models_dir, "final.pt")
    agent.save_checkpoint(final_path)
    print(f"Training finished. Final model saved to {final_path}")