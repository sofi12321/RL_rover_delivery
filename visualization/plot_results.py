import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Optional


def plot_learning_curves(
    log_paths: List[str],
    labels: List[str],
    save_path: Optional[str] = None,
    figsize: tuple = (12, 5),
    sliding_window: int = None
) -> plt.Figure:
    """
    Plot learning curves from multiple training logs.

    Args:
        log_paths: List of paths to CSV files (must contain 'step', 'avg_reward', 'success_rate').
        labels: List of labels for each curve.
        save_path: Optional path to save the figure.
        figsize: Figure size (width, height) in inches.

    Returns:
        Matplotlib Figure object.
    """
    if len(log_paths) != len(labels):
        raise ValueError("Number of log paths must match number of labels")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    for path, label in zip(log_paths, labels):
        df = pd.read_csv(path)
        if sliding_window:
            # Plot results with a sliding window to prevent noise
            sliding_window = int(sliding_window)
            ax1.plot(df['step'].rolling(window=sliding_window).mean(), df['avg_reward'].rolling(window=sliding_window).mean(), label=label)
            ax2.plot(df['step'].rolling(window=sliding_window).mean(), df['success_rate'].rolling(window=sliding_window).mean(), label=label)
        else:
            ax1.plot(df['step'], df['avg_reward'], label=label)
            ax2.plot(df['step'], df['success_rate'], label=label)

    if sliding_window:
        ax1.set_xlabel(f'Timesteps with sliding window {sliding_window}')
        ax1.set_title('Learning Curve (Reward) with sliding window')
    else:
        ax1.set_xlabel('Timesteps')
        ax1.set_title('Learning Curve (Reward)')
    ax1.set_ylabel('Average Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    if sliding_window:
        ax2.set_xlabel(f'Timesteps with sliding window {sliding_window}')
        ax2.set_title('Learning Curve (Success) with sliding window')
    else:
        ax2.set_xlabel('Timesteps')
        ax2.set_title('Learning Curve (Success)')
    ax2.set_ylabel('Success Rate')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig