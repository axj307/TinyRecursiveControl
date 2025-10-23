"""
Configuration Loader

Load and merge YAML configuration files for control problems and training.
Provides a unified interface for accessing all configuration parameters.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import copy
import os


class ConfigLoader:
    """
    Load and merge YAML configuration files.

    This class provides methods to load problem-specific configs,
    training configs, and merge them together with optional overrides.
    """

    def __init__(self, config_dir: str = "configs"):
        """
        Initialize ConfigLoader.

        Args:
            config_dir: Root directory containing config files
                       (default: "configs" relative to project root)
        """
        # Handle both absolute and relative paths
        if Path(config_dir).is_absolute():
            self.config_dir = Path(config_dir)
        else:
            # Try relative to current working directory
            self.config_dir = Path(config_dir)

            # If doesn't exist, try relative to this file's directory
            if not self.config_dir.exists():
                # Go up from src/config/ to project root
                project_root = Path(__file__).parent.parent.parent
                self.config_dir = project_root / config_dir

        if not self.config_dir.exists():
            raise FileNotFoundError(
                f"Config directory not found: {config_dir}\n"
                f"Tried: {self.config_dir.absolute()}"
            )

        self.problems_dir = self.config_dir / "problems"
        self.training_dir = self.config_dir / "training"

        # Validate structure
        if not self.problems_dir.exists():
            raise FileNotFoundError(
                f"Problems config directory not found: {self.problems_dir}"
            )
        if not self.training_dir.exists():
            raise FileNotFoundError(
                f"Training config directory not found: {self.training_dir}"
            )

    def load_problem_config(self, problem_name: str) -> Dict[str, Any]:
        """
        Load problem-specific configuration.

        Args:
            problem_name: Problem name (e.g., "double_integrator", "pendulum")

        Returns:
            Dictionary with problem configuration

        Raises:
            FileNotFoundError: If config file doesn't exist
        """
        config_path = self.problems_dir / f"{problem_name}.yaml"

        if not config_path.exists():
            available = [
                f.stem for f in self.problems_dir.glob("*.yaml")
            ]
            raise FileNotFoundError(
                f"Problem config not found: {config_path}\n"
                f"Available problems: {', '.join(available)}"
            )

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config

    def load_training_config(self, config_name: str = "default") -> Dict[str, Any]:
        """
        Load training configuration.

        Args:
            config_name: Config name (e.g., "default", "ablation")

        Returns:
            Dictionary with training configuration

        Raises:
            FileNotFoundError: If config file doesn't exist
        """
        config_path = self.training_dir / f"{config_name}.yaml"

        if not config_path.exists():
            available = [
                f.stem for f in self.training_dir.glob("*.yaml")
            ]
            raise FileNotFoundError(
                f"Training config not found: {config_path}\n"
                f"Available configs: {', '.join(available)}"
            )

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config

    def load_full_config(
        self,
        problem_name: str,
        training_config: str = "default",
        overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Load and merge full configuration.

        This loads both problem and training configs and merges them into
        a single configuration dictionary.

        Args:
            problem_name: Name of control problem
            training_config: Name of training config (default: "default")
            overrides: Dictionary of overrides to apply on top
                      (useful for command-line arguments)

        Returns:
            Merged configuration dictionary with structure:
            {
                "problem": {...},      # Problem-specific config
                "training": {...},     # Training config
                # Any overrides applied
            }

        Example:
            >>> loader = ConfigLoader()
            >>> config = loader.load_full_config(
            ...     "double_integrator",
            ...     overrides={"training": {"epochs": 200}}
            ... )
        """
        # Load base configs
        problem_cfg = self.load_problem_config(problem_name)
        training_cfg = self.load_training_config(training_config)

        # Merge into full config
        full_config = {
            "problem": problem_cfg,
            "training": training_cfg,
        }

        # Apply overrides
        if overrides:
            full_config = deep_merge(full_config, overrides)

        return full_config

    def list_problems(self) -> list:
        """
        List available problem configurations.

        Returns:
            List of problem names
        """
        return sorted([f.stem for f in self.problems_dir.glob("*.yaml")])

    def list_training_configs(self) -> list:
        """
        List available training configurations.

        Returns:
            List of training config names
        """
        return sorted([f.stem for f in self.training_dir.glob("*.yaml")])


# =============================================================================
# Utility Functions
# =============================================================================

def deep_merge(base: Dict, update: Dict) -> Dict:
    """
    Deep merge two dictionaries.

    The `update` dict is merged into `base`, with nested dicts being
    merged recursively. Non-dict values in `update` override those in `base`.

    Args:
        base: Base dictionary
        update: Update dictionary (takes precedence)

    Returns:
        Merged dictionary (new copy, doesn't modify inputs)

    Example:
        >>> base = {"a": 1, "b": {"c": 2, "d": 3}}
        >>> update = {"b": {"c": 10}, "e": 4}
        >>> result = deep_merge(base, update)
        >>> result
        {'a': 1, 'b': {'c': 10, 'd': 3}, 'e': 4}
    """
    result = copy.deepcopy(base)

    for key, value in update.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            # Recursively merge nested dicts
            result[key] = deep_merge(result[key], value)
        else:
            # Override with update value
            result[key] = copy.deepcopy(value)

    return result


def get_config(
    problem_name: str,
    training_config: str = "default",
    config_dir: str = "configs",
    **overrides
) -> Dict[str, Any]:
    """
    Convenience function to load configuration.

    This is a shorthand for creating a ConfigLoader and calling
    load_full_config().

    Args:
        problem_name: Name of control problem
        training_config: Name of training config (default: "default")
        config_dir: Config directory (default: "configs")
        **overrides: Additional overrides as keyword arguments

    Returns:
        Merged configuration dictionary

    Example:
        >>> # Simple usage
        >>> config = get_config("double_integrator")

        >>> # With training config
        >>> config = get_config("pendulum", training_config="custom")

        >>> # With overrides
        >>> config = get_config(
        ...     "double_integrator",
        ...     epochs=200,
        ...     batch_size=128
        ... )
    """
    loader = ConfigLoader(config_dir=config_dir)

    # Convert flat overrides to nested structure
    # e.g., epochs=200 -> {"training": {"training": {"epochs": 200}}}
    nested_overrides = {}
    if overrides:
        # Simple heuristic: check if keys match known training params
        training_keys = {
            "epochs", "batch_size", "learning_rate", "weight_decay",
            "optimizer", "scheduler", "early_stopping", "gradient_clip"
        }

        for key, value in overrides.items():
            if key in training_keys:
                if "training" not in nested_overrides:
                    nested_overrides["training"] = {"training": {}}
                nested_overrides["training"]["training"][key] = value
            else:
                # Unknown key, put in root
                nested_overrides[key] = value

    return loader.load_full_config(
        problem_name,
        training_config,
        overrides=nested_overrides if nested_overrides else None
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "ConfigLoader",
    "deep_merge",
    "get_config",
]
