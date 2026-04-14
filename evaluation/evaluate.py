import numpy as np
from typing import Dict, List, Optional, Any, Union

from env.rover_env import RoversEnv
from utils.helpers import maybe_render  # предположим, есть такая утилита, но можно и без


def evaluate_agent(
    env: RoversEnv,
    agent: Any,
    num_episodes: int,
    render: bool = False,
    deterministic: bool = True,
    seed: Optional[int] = None,
) -> Dict[str, Union[float, List[float]]]:
    """
    Evaluate an agent over multiple episodes.

    Args:
        env: The environment (will be reset each episode).
        agent: An object with a predict(obs, deterministic) method.
        num_episodes: Number of episodes to run.
        render: Whether to render the environment.
        deterministic: Whether to use deterministic actions.
        seed: Base seed for reproducibility. If provided, each episode uses seed + episode_index.

    Returns:
        Dictionary with statistics:
            - episode_rewards: list of total rewards per episode
            - episode_lengths: list of episode lengths
            - success_rate: fraction of episodes where goal was reached
            - collision_rate: fraction of episodes that ended in collision
            - avg_reward: mean reward
            - avg_length: mean length
    """
    episode_rewards = []
    episode_lengths = []
    successes = []
    collisions = []

    for ep in range(num_episodes):
        # Set seed for reproducibility if provided
        if seed is not None:
            env.reset(seed=seed + ep)
        else:
            env.reset()

        obs = env._get_obs()  # после reset наблюдение уже есть, но можно использовать return
        # На самом деле reset возвращает obs, info. Так что лучше:
        # obs, info = env.reset(seed=seed+ep if seed else None)
        # Но в нашей среде reset возвращает obs, info, поэтому используем так:
        obs, info = env.reset(seed=seed + ep if seed is not None else None)

        done = False
        truncated = False
        ep_reward = 0.0
        ep_length = 0

        while not (done or truncated):
            action = agent.predict(obs, deterministic=deterministic)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            ep_length += 1

            if render:
                # Предположим, у нас есть функция render из env/render.py
                from env.render import render_env
                render_env(env, title=f"Episode {ep+1}, Step {ep_length}")

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)
        successes.append(info.get("goal_reached", False))
        collisions.append(info.get("collision", False))

    success_rate = np.mean(successes)
    collision_rate = np.mean(collisions)
    avg_reward = np.mean(episode_rewards)
    avg_length = np.mean(episode_lengths)

    return {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "success_rate": success_rate,
        "collision_rate": collision_rate,
        "avg_reward": avg_reward,
        "avg_length": avg_length,
    }