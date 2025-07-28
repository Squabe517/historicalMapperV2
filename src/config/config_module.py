"""
Configuration management module for Historical ePub Map Enhancer.

Handles loading environment variables, accessing configuration values,
and validating required configuration keys.
"""

import os
import logging
from typing import Any, List
from dotenv import load_dotenv


class ConfigError(Exception):
    """Custom exception for configuration-related errors."""
    pass


def load_config(env_path: str = ".env") -> None:
    """
    Load environment variables from a .env file.
    
    Args:
        env_path: Path to the .env file (default: ".env")
    """
    logger = logging.getLogger(__name__)
    
    if os.path.exists(env_path):
        load_dotenv(env_path, override=True)
        logger.info(f"Loaded configuration from {env_path}")
    else:
        logger.warning(f"Configuration file {env_path} not found, using system environment variables only")


def get_config(key: str, default: Any = None) -> Any:
    """
    Get a configuration value from environment variables.
    
    Args:
        key: Environment variable key
        default: Default value if key not found
        
    Returns:
        Configuration value or default
    """
    logger = logging.getLogger(__name__)
    
    value = os.getenv(key, default)
    
    if value == default and default is not None:
        logger.warning(f"Configuration key '{key}' not found, using default value: {default}")
    elif value is None:
        logger.warning(f"Configuration key '{key}' not found and no default provided")
    
    return value


def validate_config(required_keys: List[str]) -> None:
    """
    Validate that all required configuration keys are present and non-empty.
    
    Args:
        required_keys: List of required environment variable keys
        
    Raises:
        ConfigError: If any required key is missing or empty
    """
    logger = logging.getLogger(__name__)
    missing_keys = []
    empty_keys = []
    
    for key in required_keys:
        value = os.getenv(key)
        if value is None:
            missing_keys.append(key)
        elif value.strip() == "":
            empty_keys.append(key)
    
    if missing_keys or empty_keys:
        error_msg = "Configuration validation failed:"
        if missing_keys:
            error_msg += f" Missing keys: {', '.join(missing_keys)}."
        if empty_keys:
            error_msg += f" Empty keys: {', '.join(empty_keys)}."
        
        logger.error(error_msg)
        raise ConfigError(error_msg)
    
    logger.info(f"Configuration validation passed for keys: {', '.join(required_keys)}")
