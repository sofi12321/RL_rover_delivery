import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from env.rover_env import RoversEnv


def compare_agents(
    env: RoversEnv,
    agents_dict: Dict[str, Any],
    num_episodes: int,
    save_path: Optional[str] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Compare multiple agents on the same set of episodes.

    Args:
        env: The environment.
        agents_dict: Dictionary mapping agent names to agent objects.
        num_episodes: Number of episodes to run.
        save_path: Optional path to save results as CSV.
        seed: Base seed for reproducibility. Each episode uses seed + episode_index.

    Returns:
        Pandas DataFrame with columns: agent, episode, reward, length, success, collision.
    """
    results = []

    for ep in range(num_episodes):
        ep_seed = seed + ep

        for agent_name, agent in agents_dict.items():
            # Reset environment with the same seed for all agents
            obs, info = env.reset(seed=ep_seed)

            done = False
            truncated = False
            ep_reward = 0.0
            ep_length = 0

            while not (done or truncated):
                action = agent.predict(obs, deterministic=True)  # usually deterministic for evaluation
                obs, reward, done, truncated, info = env.step(action)
                ep_reward += reward
                ep_length += 1

            results.append({
                "agent": agent_name,
                "episode": ep,
                "reward": ep_reward,
                "length": ep_length,
                "success": info.get("goal_reached", False),
                "collision": info.get("collision", False),
            })

    df = pd.DataFrame(results)

    if save_path:
        df.to_csv(save_path, index=False)

    return df