"""
Configuration Management

This module provides configuration loading and management for
TinyRecursiveControl. It supports YAML-based configurations for
both control problems and training settings.

Usage:
    >>> from src.config import get_config
    >>> config = get_config("double_integrator")
    >>> print(config["problem"]["dynamics"]["dt"])
"""

from .loader import ConfigLoader, get_config, deep_merge

__all__ = [
    "ConfigLoader",
    "get_config",
    "deep_merge",
]
