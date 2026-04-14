import os
import yaml
import torch
from typing import Any, Dict, Optional, Union

def deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merges two dictionaries. Values from `update` overwrite those in `base`.
    If both values are dictionaries, they are merged recursively.
    """
    for key, value in update.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


class Config:
    """
    Configuration class that provides attribute-style access to nested dictionaries.

    Example:
        >>> config = Config({'training': {'batch_size': 256}})
        >>> config.training.batch_size
        256
        >>> config.training.batch_size = 512
        >>> config.training.batch_size
        512
        >>> config.save('config.yaml')
        >>> config.load('config.yaml')
    """

    def __init__(self, data: Optional[Union[str, Dict[str, Any]]] = None):
        """
        Initialize the configuration.

        Args:
            data: Either a path to a YAML file, a dictionary with configuration data,
                  or None (default values are used).
        """
        # Default configuration
        self._data: Dict[str, Any] = {
            'environment': {
                'field_size': [20, 20],
                'max_steps': 1000,
                'dt': 0.1,
                'robot_radius': 0.5,
                'goal_radius': 1.5,
                'min_obstacles': 2,
                'max_obstacles': 4,
                'obstacle_radius_range': [1, 2],
                'init_position': 'random',
                'goal_position': 'random',
                'sensor_range': 5.0,
                'reward_weights': {
                    'progress': 1.0,
                    'collision': 100.0,
                    'steer': 0.1,
                    'speed': 0.01,
                    'goal': 500.0,
                    'time': 0.1
                },
                'max_speed': 1.0,
                'max_steer': 0.5
            },
            'training': {
                'buffer_size': 1000000,
                'batch_size': 256,
                'gamma': 0.99,
                'tau': 0.005,
                'lr': 3e-4,
                'alpha': 'auto',
                'target_entropy': 'auto',
                'total_timesteps': 1000000,
                'eval_freq': 10000,
                'eval_episodes': 20
            },
            'device': 'auto'
        }

        if isinstance(data, str):
            self.load(data)
        elif isinstance(data, dict):
            deep_merge(self._data, data)
        # Resolve 'auto' device
        self._resolve_device()

    def _resolve_device(self) -> None:
        """Replace 'auto' in device field with actual torch device string."""
        if self._data.get('device') == 'auto':
            self._data['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    def load(self, path: str) -> None:
        """
        Load configuration from a YAML file and merge with current settings.

        Args:
            path: Path to the YAML file.
        """
        with open(path, 'r') as f:
            loaded = yaml.safe_load(f)
        deep_merge(self._data, loaded)
        self._resolve_device()

    def save(self, path: str) -> None:
        """
        Save current configuration to a YAML file.

        Args:
            path: Path where the YAML file will be written.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self._data, f, default_flow_style=False)

    def __getattr__(self, name: str) -> Any:
        """Access nested configuration items as attributes."""
        if name == '_data':
            raise AttributeError
        if '_data' not in self.__dict__:
            raise AttributeError
        if name in self._data:
            value = self._data[name]
            if isinstance(value, dict):
                return Config(value)
            return value
        raise AttributeError

    def __setattr__(self, name: str, value: Any) -> None:
        """Set configuration items as attributes."""
        if name.startswith('_'):  # internal attribute
            super().__setattr__(name, value)
        else:
            self._data[name] = value

    def __repr__(self) -> str:
        return f"Config({self._data})"

class _ConfigProxy:
    """Proxy for nested dictionary access that allows chained assignment."""
    def __init__(self, root: Config, path: list):
        self._root = root
        self._path = path

    def __getattr__(self, name: str) -> Any:
        # Navigate to the current dictionary
        current = self._root._data
        for key in self._path:
            current = current[key]
        if name in current:
            value = current[name]
            if isinstance(value, dict):
                # Return a proxy object that knows its parent and key
                return _ConfigProxy(self._root, self._path + [name])
            return value
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any):
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            # Navigate to the parent dictionary
            current = self._root._data
            for key in self._path:
                current = current[key]
            current[name] = value

    def __repr__(self):
        return f"_ConfigProxy(path={self._path})"


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from a YAML file or return default configuration.

    Args:
        config_path: Optional path to a YAML configuration file.

    Returns:
        Config object.
    """
    return Config(config_path)