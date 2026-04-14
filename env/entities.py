import math
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class Obstacle:
    """Circular obstacle."""
    x: float
    y: float
    radius: float

    def contains(self, px: float, py: float) -> bool:
        """Check if point (px, py) lies inside the obstacle (including boundary)."""
        dx = px - self.x
        dy = py - self.y
        return dx * dx + dy * dy <= self.radius * self.radius


@dataclass
class Goal:
    """Circular goal region."""
    x: float
    y: float
    radius: float

    def reached(self, px: float, py: float) -> bool:
        """Check if point (px, py) is within the goal region (including boundary)."""
        dx = px - self.x
        dy = py - self.y
        return dx * dx + dy * dy <= self.radius * self.radius


@dataclass
class Robot:
    """Simple differential-drive robot model."""
    x: float
    y: float
    theta: float          # orientation in radians
    v: float              # linear velocity (can be negative)
    radius: float

    def update(
        self,
        steer: float,
        acceleration: float,
        dt: float,
        max_speed: float,
        max_steer: float
    ) -> None:
        """
        Update robot state given normalized control inputs.

        Args:
            steer: normalized steering command in [-1, 1]
            acceleration: normalized acceleration command in [-1, 1]
            dt: time step (seconds)
            max_speed: maximum absolute speed (m/s)
            max_steer: maximum steering angle change per second (rad/s)
        """
        # Clip inputs just in case (though they should already be in [-1, 1])
        steer = max(-1.0, min(1.0, steer))
        acceleration = max(-1.0, min(1.0, acceleration))

        # Update velocity
        self.v += acceleration * dt
        self.v = max(-max_speed, min(max_speed, self.v))

        # Update orientation
        self.theta += steer * max_steer * dt

        # Update position
        self.x += self.v * math.cos(self.theta) * dt
        self.y += self.v * math.sin(self.theta) * dt
