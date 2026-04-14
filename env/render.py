# env/render.py

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import numpy as np
from typing import Optional, Tuple

from .rover_env import RoversEnv
from .sensors import get_sensor_readings


def render_env(
    env: RoversEnv,
    ax: Optional[plt.Axes] = None,
    show_sensors: bool = True,
    title: Optional[str] = None,
    num_rays: int = 8,
    trail=None,
) -> plt.Figure:
    """
    Render the current state of the rover environment.

    Args:
        env: The RoversEnv instance.
        ax: Matplotlib axes to draw on. If None, a new figure and axes are created.
        show_sensors: If True, draw sensor rays.
        title: Optional title for the plot.
        num_rays: Number of sensor rays to draw (must match env's sensor count).

    Returns:
        The matplotlib Figure object.
    """
    # Create axes if not provided
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig = ax.figure

    # Clear axes for fresh drawing
    ax.clear()

    # Extract environment parameters
    field_width, field_height = env.field_width, env.field_height
    robot = env.robot
    goal = env.goal
    obstacles = env.obstacles
    sensor_range = env.sensor_range

    # Draw field boundary
    rect = patches.Rectangle(
        (0, 0), field_width, field_height,
        linewidth=2, edgecolor='black', facecolor='none'
    )
    ax.add_patch(rect)

    # Draw obstacles (gray circles)
    for obs in obstacles:
        circle = patches.Circle(
            (obs.x, obs.y), obs.radius,
            edgecolor='black', facecolor='gray', alpha=0.7
        )
        ax.add_patch(circle)

    # Draw goal (green circle)
    goal_circle = patches.Circle(
        (goal.x, goal.y), goal.radius,
        edgecolor='darkgreen', facecolor='lightgreen', alpha=0.8
    )
    ax.add_patch(goal_circle)

    if trail is not None and len(trail) > 1:
        trail_x = [p[0] for p in trail]
        trail_y = [p[1] for p in trail]
        ax.plot(trail_x, trail_y, 'b-', linewidth=1, alpha=0.5, label='Path')

    # Draw robot (blue circle) with direction line
    robot_circle = patches.Circle(
        (robot.x, robot.y), robot.radius,
        edgecolor='darkblue', facecolor='skyblue', alpha=0.9
    )
    ax.add_patch(robot_circle)

    # Direction line (length proportional to radius)
    dir_length = robot.radius * 1.5
    dx_dir = dir_length * np.cos(robot.theta)
    dy_dir = dir_length * np.sin(robot.theta)
    direction_line = Line2D(
        [robot.x, robot.x + dx_dir],
        [robot.y, robot.y + dy_dir],
        color='darkblue', linewidth=2
    )
    ax.add_line(direction_line)

    # Draw sensor rays if requested
    if show_sensors:
        # Get normalized sensor readings
        readings = get_sensor_readings(robot, obstacles, sensor_range, (field_width, field_height), num_rays)
        # Compute ray angles
        angles = [robot.theta + (2.0 * np.pi * i / num_rays) for i in range(num_rays)]

        for i, (reading, angle) in enumerate(zip(readings, angles)):
            # Actual distance = reading * sensor_range
            dist = reading * sensor_range
            # If dist is close to sensor_range, ray is unobstructed; we still draw up to max_range.
            # We'll draw a line from robot to the hit point (or to max_range if no hit).
            end_x = robot.x + dist * np.cos(angle)
            end_y = robot.y + dist * np.sin(angle)

            # Use different color for rays that hit something (dist < sensor_range - epsilon)
            color = 'red' if dist < sensor_range - 1e-6 else 'orange'
            alpha = 0.7 if dist < sensor_range - 1e-6 else 0.3

            ray_line = Line2D(
                [robot.x, end_x], [robot.y, end_y],
                color=color, linewidth=1, alpha=alpha
            )
            ax.add_line(ray_line)

            # Optional: mark the hit point with a small dot
            if dist < sensor_range - 1e-6:
                ax.plot(end_x, end_y, 'ro', markersize=2, alpha=0.7)

    # Set axis limits with a small margin
    margin = 1.0
    ax.set_xlim(-margin, field_width + margin)
    ax.set_ylim(-margin, field_height + margin)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.5)

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Step {env.current_step}")

    # Force a draw
    fig.canvas.draw_idle()

    return fig


# Alias for convenience
render_frame = render_env