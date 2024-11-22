"""Configuration loader module."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class Config:
    """Configuration class."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize config from dictionary."""
        self._config = config_dict
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated key.
        
        Args:
            key: Dot-separated key (e.g., 'database.path')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def __getitem__(self, key: str) -> Any:
        """Get configuration value."""
        return self._config[key]
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return key in self._config


def load_config(config_path: Optional[Path] = None) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file (default: config/config.yaml)
        
    Returns:
        Config object
    """
    if config_path is None:
        # Default to config/config.yaml in project root
        project_root = Path(__file__).parent.parent
        config_path = project_root / "config" / "config.yaml"
    
    # Check environment variable for config path
    env_config_path = os.getenv("MODEL_MONITOR_CONFIG")
    if env_config_path:
        config_path = Path(env_config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Override with environment variables
    config_dict = _override_with_env(config_dict)
    
    return Config(config_dict)


def _override_with_env(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Override config values with environment variables.
    
    Environment variables should be prefixed with MODEL_MONITOR_ and use underscores.
    Example: MODEL_MONITOR_DATABASE_PATH
    """
    env_prefix = "MODEL_MONITOR_"
    
    # Common overrides
    if os.getenv(f"{env_prefix}DATABASE_PATH"):
        config_dict.setdefault("database", {})["path"] = os.getenv(f"{env_prefix}DATABASE_PATH")
    
    if os.getenv(f"{env_prefix}MODEL_PATH"):
        config_dict.setdefault("model", {})["path"] = os.getenv(f"{env_prefix}MODEL_PATH")
    
    if os.getenv(f"{env_prefix}API_HOST"):
        config_dict.setdefault("api", {})["host"] = os.getenv(f"{env_prefix}API_HOST")
    
    if os.getenv(f"{env_prefix}API_PORT"):
        config_dict.setdefault("api", {})["port"] = int(os.getenv(f"{env_prefix}API_PORT"))
    
    if os.getenv(f"{env_prefix}LOG_LEVEL"):
        config_dict.setdefault("logging", {})["level"] = os.getenv(f"{env_prefix}LOG_LEVEL")
    
    return config_dict


# Global config instance (lazy loaded)
_config: Optional[Config] = None


def get_config() -> Config:
    """
    Get global config instance (lazy loaded).
    
    Returns:
        Config object
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reload_config() -> Config:
    """
    Reload configuration.
    
    Returns:
        Config object
    """
    global _config
    _config = load_config()
    return _config
