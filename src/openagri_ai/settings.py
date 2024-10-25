"""
Module to dynamically load the project settings
"""

from .config import OpenagriConfig
import importlib
import os

_config_name = ""
config = None


def _load_config(config_name: str) -> dict:
    """Loads a configuration dynamically

    Args:
        config_name (str): Name of the configuration file

    Returns:
        dict: Module description
    """
    config_module = importlib.import_module(f".config.{config_name}", "openagri_ai")
    return {k: v for k, v in vars(config_module).items() if not k.startswith("__")}


def get_config() -> OpenagriConfig:
    """Get the configuration based on the latest CONFIG_NAME environmental variable.

    Returns:
        OpenagriConfig: The latest openagri config
    """
    global config, _config_name
    config_name = os.getenv("CONFIG_NAME", "default")
    if config_name == _config_name:
        return config
    _config_name = config_name
    config = _load_config(_config_name)["config"]
    return config
