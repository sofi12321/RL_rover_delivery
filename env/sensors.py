import math
import numpy as np
from typing import List, Tuple

from .entities import Obstacle, Robot


def ray_cast(
    robot_x: float,
    robot_y: float,
    angle: float,
    obstacles: List[Obstacle],
    max_range: float,
    field_size: Tuple,
    eps: float = 1e-6
) -> float:
    """
    Cast a ray from the robot and return the distance to the nearest obstacle.

    Args:
        robot_x, robot_y: ray origin
        angle: ray direction (radians, global frame)
        obstacles: list of Obstacle objects
        max_range: maximum sensing range
        eps: small tolerance for t>0

    Returns:
        Distance to the nearest obstacle along the ray, or max_range if none.
    """
    min_t = float('inf')
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    # Check obstacles
    for obs in obstacles:
        # Vector from robot to obstacle center
        dx = obs.x - robot_x
        dy = obs.y - robot_y

        # Quadratic coefficients: a = 1, b = -2*(dx*cos_a + dy*sin_a), c = dx^2 + dy^2 - r^2
        b = -2.0 * (dx * cos_a + dy * sin_a)
        c = dx * dx + dy * dy - obs.radius * obs.radius
        disc = b * b - 4.0 * c

        if disc >= 0:
            sqrt_disc = math.sqrt(disc)
            t1 = (-b - sqrt_disc) / 2.0
            t2 = (-b + sqrt_disc) / 2.0

            # We need the smallest positive t
            for t in (t1, t2):
                if t > eps and t < min_t:
                    min_t = t

    # Проверка границ поля (прямоугольник [0, width] x [0, height])
    # Луч параметрически: (x0 + t*cos, y0 + t*sin)
    # Пересечения с вертикальными линиями x=0 и x=width
    for x_bound in [0, field_size[0]]:
        if abs(cos_a) > eps:
            t = (x_bound - robot_x) / cos_a
            if t > eps:
                y_hit = robot_y + t * sin_a
                if 0 <= y_hit <= field_size[1]:
                    if t < min_t:
                        min_t = t
    # Пересечения с горизонтальными линиями y=0 и y=height
    for y_bound in [0, field_size[1]]:
        if abs(sin_a) > eps:
            t = (y_bound - robot_y) / sin_a
            if t > eps:
                x_hit = robot_x + t * cos_a
                if 0 <= x_hit <= field_size[0]:
                    if t < min_t:
                        min_t = t

    if math.isfinite(min_t):
        return min_t
    else:
        return max_range


def get_sensor_readings(
    robot: Robot,
    obstacles: List[Obstacle],
    max_range: float,
    field_size: Tuple,
    num_rays: int = 8
) -> np.ndarray:
    """
    Compute sensor readings for a set of rays around the robot.

    Args:
        robot: Robot object
        obstacles: list of Obstacle objects
        max_range: maximum sensing range
        num_rays: number of evenly spaced rays

    Returns:
        numpy array of shape (num_rays,) with values in [0,1].
        reading = distance / max_range, where distance is capped at max_range.
    """
    readings = np.zeros(num_rays, dtype=np.float32)
    for i in range(num_rays):
        # Ray angle = robot orientation + evenly spaced offset
        angle = robot.theta + (2.0 * math.pi * i / num_rays)
        dist = ray_cast(robot.x, robot.y, angle, obstacles, max_range, field_size)
        # Normalize: 1.0 means no obstacle within max_range, 0.0 means obstacle at origin
        readings[i] = min(dist / max_range, 1.0)
    return readings
