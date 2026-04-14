import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any, List

# from tests.test_networks import obs_dim
from utils.config import Config
from .entities import Obstacle, Goal, Robot
from .sensors import get_sensor_readings


class RoversEnv(gym.Env):
    """
    Rover navigation environment with obstacles and a goal.

    Observation space:
        - 8 sensor readings (normalized distances to obstacles) [0, 1]
        - normalized speed [-1, 1]
        - normalized goal offset (dx, dy) in range [-1, 1] (relative to field size)
    Total dim = 13.

    Action space:
        - steer: continuous in [-1, 1]
        - acceleration: continuous in [-1, 1]
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        env_cfg = config.environment

        # Spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # Observation bounds: sensors [0,1], speed [-1,1], dx,dy [-1,1]
        obs_dim = 8 + 1 + 2 + 2
        obs_low = np.array([0.0] * 8 + [-1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32)
        obs_high = np.array([1.0] * 8 + [1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

        # Environment parameters
        self.field_width, self.field_height = env_cfg.field_size
        self.max_steps = env_cfg.max_steps
        self.dt = env_cfg.dt
        self.robot_radius = env_cfg.robot_radius
        self.goal_radius = env_cfg.goal_radius
        self.min_obstacles = env_cfg.min_obstacles
        self.max_obstacles = env_cfg.max_obstacles
        self.obstacle_radius_range = env_cfg.obstacle_radius_range
        self.init_position = env_cfg.init_position
        self.goal_position = env_cfg.goal_position
        self.sensor_range = env_cfg.sensor_range
        self.max_speed = env_cfg.max_speed
        self.max_steer = env_cfg.max_steer
        self.reward_weights = env_cfg.reward_weights

        # State
        self.obstacles: List[Obstacle] = []
        self.goal: Optional[Goal] = None
        self.robot: Optional[Robot] = None
        self.current_step = 0
        self.prev_distance_to_goal = 0.0

        # Seeding
        self.np_random = None
        self.seed()

    def seed(self, seed: Optional[int] = None):
        self.np_random = np.random.RandomState(seed)
        return [seed]

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)

        # Generate obstacles
        self._generate_obstacles()

        # Generate goal
        self._generate_goal()

        # Generate robot
        self._generate_robot()

        self.current_step = 0
        self.prev_distance_to_goal = self._distance_to_goal()

        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # Ensure action is within bounds (clip just in case)
        action = np.clip(action, -1.0, 1.0)

        steer, acceleration = action

        # Update robot
        self.robot.update(
            steer, acceleration, self.dt, self.max_speed, self.max_steer
        )

        # Check collision
        collision = self._check_collision()

        # Check goal reached
        goal_reached = bool(self.goal.reached(self.robot.x, self.robot.y))

        # Compute new distance to goal
        new_dist = self._distance_to_goal()

        # Compute reward
        reward = self._compute_reward(
            action, collision, goal_reached, self.prev_distance_to_goal, new_dist
        )

        # Update previous distance for next step
        self.prev_distance_to_goal = new_dist

        # Get observation
        obs = self._get_obs()

        # Determine termination and truncation
        done = bool(collision or goal_reached)
        truncated = self.current_step >= self.max_steps - 1  # because step increments after
        self.current_step += 1

        info = {
            "collision": collision,
            "goal_reached": goal_reached,
            "distance": new_dist,
        }

        return obs, reward, done, truncated, info

    def _get_obs(self) -> np.ndarray:
        """Construct observation vector."""
        # Sensor readings (8 rays)
        sensors = get_sensor_readings(
            self.robot, self.obstacles, self.sensor_range, (self.field_width, self.field_height), num_rays=8
        )

        # Normalized speed
        speed_norm = self.robot.v / self.max_speed  # range [-1, 1]

        # Normalized goal offset
        dx = (self.goal.x - self.robot.x) / self.field_width
        dy = (self.goal.y - self.robot.y) / self.field_height

        # Current steering wheel turn
        sin_theta = math.sin(self.robot.theta)
        cos_theta = math.cos(self.robot.theta)

        obs = np.concatenate([sensors, [speed_norm, dx, dy, sin_theta, cos_theta]]).astype(np.float32)
        return obs

    def _distance_to_goal(self) -> float:
        """Euclidean distance from robot to goal center."""
        dx = self.goal.x - self.robot.x
        dy = self.goal.y - self.robot.y
        return math.hypot(dx, dy)

    def _check_collision(self) -> bool:
        """Check if robot collides with any obstacle."""

        # Check if robot collides with circle obstacles
        for obs in self.obstacles:
            dx = obs.x - self.robot.x
            dy = obs.y - self.robot.y
            dist = math.hypot(dx, dy)
            if dist <= self.robot_radius + obs.radius:
                return True

        # Check if robot collides with field borders
        if (self.robot.x - self.robot_radius < 0 or
                self.robot.x + self.robot_radius > self.field_width or
                self.robot.y - self.robot_radius < 0 or
                self.robot.y + self.robot_radius > self.field_height):
            return True
        return False

    def _generate_obstacles(self, max_attempts: int = 1000):
        """Generate non-overlapping obstacles."""
        num_obstacles = self.np_random.randint(
            self.min_obstacles, self.max_obstacles + 1
        )
        self.obstacles = []
        attempts = 0
        while len(self.obstacles) < num_obstacles and attempts < max_attempts:
            radius = self.np_random.uniform(*self.obstacle_radius_range)
            x = self.np_random.uniform(radius, self.field_width - radius)
            y = self.np_random.uniform(radius, self.field_height - radius)
            new_obs = Obstacle(x, y, radius)

            # Check overlap with existing obstacles
            overlap = False
            for obs in self.obstacles:
                dx = obs.x - new_obs.x
                dy = obs.y - new_obs.y
                min_dist = obs.radius + new_obs.radius + 0.1  # small buffer
                if dx * dx + dy * dy < min_dist * min_dist:
                    overlap = True
                    break
            if not overlap:
                self.obstacles.append(new_obs)
            attempts += 1

        if len(self.obstacles) < num_obstacles:
            # If failed to place all obstacles, proceed with what we have (but warn)
            print(f"Warning: only placed {len(self.obstacles)}/{num_obstacles} obstacles")

    def _generate_goal(self, max_attempts: int = 1000):
        """Place the goal, ensuring it doesn't overlap with obstacles."""
        if isinstance(self.goal_position, dict):
            x = self.goal_position.get('x', self.field_width / 2)
            y = self.goal_position.get('y', self.field_height / 2)
            self.goal = Goal(x, y, self.goal_radius)
            # Even if specified, ensure it's not inside obstacles (if it is, we'll still place it? better to allow)
            # We'll allow, but could add a check.
        else:  # 'random'
            attempts = 0
            while attempts < max_attempts:
                x = self.np_random.uniform(self.goal_radius, self.field_width - self.goal_radius)
                y = self.np_random.uniform(self.goal_radius, self.field_height - self.goal_radius)
                goal = Goal(x, y, self.goal_radius)

                # Check overlap with obstacles
                overlap = False
                for obs in self.obstacles:
                    dx = obs.x - goal.x
                    dy = obs.y - goal.y
                    min_dist = obs.radius + goal.radius + 0.1
                    if dx * dx + dy * dy < min_dist * min_dist:
                        overlap = True
                        break
                if not overlap:
                    self.goal = goal
                    return
                attempts += 1
            # Fallback: place at center (should be rare)
            self.goal = Goal(self.field_width / 2, self.field_height / 2, self.goal_radius)

    def _generate_robot(self, max_attempts: int = 1000):
        """Place the robot, ensuring it doesn't overlap with obstacles or goal."""
        if isinstance(self.init_position, dict):
            x = self.init_position.get('x', self.field_width / 2)
            y = self.init_position.get('y', self.field_height / 2)
            theta = self.init_position.get('theta', self.np_random.uniform(0, 2 * math.pi))
            self.robot = Robot(x, y, theta, 0.0, self.robot_radius)
        else:  # 'random'
            attempts = 0
            while attempts < max_attempts:
                x = self.np_random.uniform(self.robot_radius, self.field_width - self.robot_radius)
                y = self.np_random.uniform(self.robot_radius, self.field_height - self.robot_radius)
                theta = self.np_random.uniform(0, 2 * math.pi)
                robot = Robot(x, y, theta, 0.0, self.robot_radius)

                # Check overlap with obstacles
                overlap = False
                for obs in self.obstacles:
                    dx = obs.x - robot.x
                    dy = obs.y - robot.y
                    min_dist = obs.radius + robot.radius + 0.1
                    if dx * dx + dy * dy < min_dist * min_dist:
                        overlap = True
                        break
                if overlap:
                    attempts += 1
                    continue

                # Check distance to goal
                if self.goal is not None:
                    dx = self.goal.x - robot.x
                    dy = self.goal.y - robot.y
                    min_dist_to_goal = self.goal_radius + robot.radius + 0.1
                    if dx * dx + dy * dy < min_dist_to_goal * min_dist_to_goal:
                        attempts += 1
                        continue

                self.robot = robot
                return
            # Fallback: place at (1,1)
            self.robot = Robot(1.0, 1.0, 0.0, 0.0, self.robot_radius)

    def _compute_reward(
        self,
        action: np.ndarray,
        collision: bool,
        goal_reached: bool,
        prev_dist: float,
        new_dist: float,
    ) -> float:
        """Compute reward based on weights."""
        w = self.reward_weights

        reward = 0.0

        # Progress towards goal
        progress = prev_dist - new_dist
        reward += w.progress * progress

        # Collision penalty
        if collision:
            reward -= w.collision

        # Goal bonus
        if goal_reached:
            reward += w.goal

        # Steering penalty (encourage smooth steering)
        reward -= w.steer * abs(action[0])

        # Speed penalty: encourage moving fast (positive speed)
        # We penalize if speed is less than max_speed
        speed_penalty = w.speed * (1.0 - abs(self.robot.v) / self.max_speed)
        reward -= speed_penalty

        # Negative speed penalty
        if self.robot.v < 0:
            reward -= w.speed * abs(self.robot.v) / self.max_speed * 0.25

        # Time penalty (constant per step)
        reward -= w.time

        return float(reward)
