import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from typing import Any, Optional

from env.rover_env import RoversEnv
from env.render import render_env


def create_comparison_video(
    env: RoversEnv,
    agent1: Any,
    agent2: Any,
    num_steps: int,
    save_path: str,
    seed: int = 42,
    fps: int = 30,
    figsize: tuple = (16, 8),
    agent_names: list = ["1", "2"]
) -> animation.FuncAnimation:
    """
    Create a side-by-side video comparing two agents on the same initial conditions.

    Args:
        env: The environment template (will be cloned for each agent).
        agent1: First agent.
        agent2: Second agent.
        num_steps: Maximum number of steps.
        save_path: Path to save the video.
        seed: Seed for reproducible initial state.
        fps: Frames per second.
        figsize: Figure size.

    Returns:
        The FuncAnimation object.
    """
    # Create two independent environments
    env1 = RoversEnv(env.config)
    env2 = RoversEnv(env.config)

    # Reset both with same seed
    obs1, _ = env1.reset(seed=seed)
    obs2, _ = env2.reset(seed=seed)  # same seed ensures identical initial state

    done1 = done2 = False
    truncated1 = truncated2 = False
    trail1 = [(env1.robot.x, env1.robot.y)]
    trail2 = [(env2.robot.x, env2.robot.y)]

    # Set up figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    def init():
        ax1.clear()
        ax2.clear()
        return [ax1, ax2]

    def update(frame):
        nonlocal obs1, obs2, env1, env2, done1, done2, truncated1, truncated2, trail1, trail2

        # Render both environments
        render_env(env1, ax=ax1, show_sensors=True, title=f"Agent {agent_names[0]}", trail=trail1)
        render_env(env2, ax=ax2, show_sensors=True, title=f"Agent {agent_names[1]}", trail=trail2)

        # Get actions
        if not (done1 or truncated1):
            action1 = agent1.predict(obs1, deterministic=True)
            # Step both environments
            obs1, reward1, done1, truncated1, info1 = env1.step(action1)
            trail1.append((env1.robot.x, env1.robot.y))

        if not (done2 or truncated2):
            action2 = agent2.predict(obs2, deterministic=True)
            # Step both environments
            obs2, reward2, done2, truncated2, info2 = env2.step(action2)
            trail2.append((env2.robot.x, env2.robot.y))


        return [ax1, ax2]

    anim = animation.FuncAnimation(
        fig, update, frames=num_steps, init_func=init, blit=False, repeat=False
    )

    anim.save(save_path, writer='ffmpeg', fps=fps, dpi=100)
    plt.close(fig)
    return anim