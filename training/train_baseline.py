import os
import csv
from typing import Optional

from stable_baselines3.common.callbacks import BaseCallback

from env.rover_env import RoversEnv
from baseline.baseline_agent import BaselineSACAgent
from utils.config import Config
from evaluation.evaluate import evaluate_agent


class BaselineEvalCallback(BaseCallback):
    """
    Custom callback for evaluating and saving baseline SAC agent during training.
    """

    def __init__(
        self,
        eval_env: RoversEnv,
        agent: BaselineSACAgent,
        eval_freq: int,
        eval_episodes: int,
        log_dir: str,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.agent = agent
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.log_dir = log_dir
        self.models_dir = os.path.join(log_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)

        # CSV logging
        self.csv_path = os.path.join(log_dir, "progress.csv")
        with open(self.csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                ["step", "avg_reward", "success_rate", "avg_length"]
            )

        self.last_step_logged = 0

    def _on_step(self) -> bool:
        """
        Called after each environment step.
        Returns True to continue training.
        """
        # Check if we should evaluate
        if self.num_timesteps - self.last_step_logged >= self.eval_freq:
            self.last_step_logged = self.num_timesteps

            # Evaluate agent
            eval_stats = evaluate_agent(
                env=self.eval_env,
                agent=self.agent,
                num_episodes=self.eval_episodes,
                deterministic=True,
            )

            # Save checkpoint
            checkpoint_path = os.path.join(self.models_dir, f"step_{self.num_timesteps}.zip")
            self.agent.save(checkpoint_path)

            # Log to CSV
            with open(self.csv_path, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(
                    [
                        self.num_timesteps,
                        eval_stats.get("avg_reward", 0.0),
                        eval_stats.get("success_rate", 0.0),
                        eval_stats.get("avg_length", 0.0),
                    ]
                )

            if self.verbose > 0:
                print(f"Step {self.num_timesteps}: avg_reward={eval_stats['avg_reward']:.2f}, "
                      f"success_rate={eval_stats['success_rate']:.2f}")

        return True


def train_baseline(
    env: RoversEnv,
    agent: BaselineSACAgent,
    config: Config,
    log_dir: str,
) -> None:
    """
    Train a baseline SAC agent (stable-baselines3) with logging and evaluation.

    Args:
        env: The environment instance (used for evaluation, training env is inside agent).
        agent: The BaselineSACAgent instance.
        config: Configuration object.
        log_dir: Directory to save logs and checkpoints.
    """
    # Create directories
    os.makedirs(log_dir, exist_ok=True)

    # Create a separate environment for evaluation (to avoid interfering with training)
    eval_env = RoversEnv(config)

    # Create callback
    callback = BaselineEvalCallback(
        eval_env=eval_env,
        agent=agent,
        eval_freq=config.training.eval_freq,
        eval_episodes=config.training.eval_episodes,
        log_dir=log_dir,
        verbose=1,
    )

    # Start training
    agent.learn(total_timesteps=config.training.total_timesteps, callback=callback)

    # Save final model
    final_path = os.path.join(log_dir, "models", "final.zip")
    agent.save(final_path)
    print(f"Training finished. Final model saved to {final_path}")
