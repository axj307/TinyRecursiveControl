"""
Control Problem Metadata

Unified metadata schema for control problem datasets.
Inspired by TinyRecursiveModels' PuzzleDatasetMetadata.

This provides a standardized way to store and load dataset metadata
across all control problems.
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional, List
import json
import numpy as np
from pathlib import Path


@dataclass
class ControlProblemMetadata:
    """
    Metadata for control problem datasets.

    This stores all information needed to understand and reproduce
    a dataset of optimal control trajectories.
    """

    # =========================================================================
    # Problem Identification
    # =========================================================================
    problem_name: str
    """Name of the control problem (e.g., 'double_integrator', 'pendulum')"""

    problem_type: str
    """Type of system: 'linear' or 'nonlinear'"""

    # =========================================================================
    # Dimensions
    # =========================================================================
    state_dim: int
    """Dimension of state space"""

    control_dim: int
    """Dimension of control input space"""

    horizon: int
    """Control horizon (number of timesteps)"""

    # =========================================================================
    # Discretization
    # =========================================================================
    dt: float
    """Time step (discretization)"""

    total_time: float
    """Total time horizon (dt * horizon)"""

    # =========================================================================
    # Bounds
    # =========================================================================
    state_bounds: Dict[str, List[float]]
    """State space bounds: {'lower': [...], 'upper': [...]}"""

    control_bounds: Dict[str, List[float]]
    """Control input bounds: {'lower': [...], 'upper': [...]}"""

    # =========================================================================
    # Dataset Statistics
    # =========================================================================
    num_samples: int
    """Total number of trajectories in dataset"""

    num_train: Optional[int] = None
    """Number of training samples (if split)"""

    num_test: Optional[int] = None
    """Number of test samples (if split)"""

    # =========================================================================
    # Cost/Optimization Parameters
    # =========================================================================
    lqr_params: Optional[Dict[str, Any]] = None
    """LQR cost parameters (Q, R, terminal weight, etc.)"""

    controller_type: str = "lqr"
    """Controller used for generation ('lqr', 'minimum_energy', 'mpc', etc.)"""

    # =========================================================================
    # Generation Info
    # =========================================================================
    seed: Optional[int] = None
    """Random seed used for generation"""

    generation_timestamp: Optional[str] = None
    """ISO format timestamp of generation"""

    generation_time_seconds: Optional[float] = None
    """Wall-clock time for generation"""

    # =========================================================================
    # Additional Info
    # =========================================================================
    notes: Optional[str] = None
    """Free-form notes about the dataset"""

    problem_specific_params: Optional[Dict[str, Any]] = None
    """Problem-specific parameters (e.g., mass, length for pendulum)"""

    dataset_version: str = "1.0"
    """Version of dataset format"""

    # =========================================================================
    # Methods
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metadata to dictionary.

        Returns:
            Dictionary representation
        """
        return asdict(self)

    def to_json(self, filepath: str):
        """
        Save metadata to JSON file.

        Args:
            filepath: Path to output JSON file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=_json_serializer)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ControlProblemMetadata":
        """
        Load metadata from dictionary.

        Args:
            data: Dictionary with metadata fields

        Returns:
            ControlProblemMetadata instance
        """
        return cls(**data)

    @classmethod
    def from_json(cls, filepath: str) -> "ControlProblemMetadata":
        """
        Load metadata from JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            ControlProblemMetadata instance
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def summary(self) -> str:
        """
        Get human-readable summary of metadata.

        Returns:
            Multi-line string with key information
        """
        lines = [
            "=" * 70,
            "Control Problem Dataset Metadata",
            "=" * 70,
            f"Problem: {self.problem_name} ({self.problem_type})",
            f"State dim: {self.state_dim}, Control dim: {self.control_dim}",
            f"Horizon: {self.horizon} steps × {self.dt}s = {self.total_time}s",
            "",
            f"Samples: {self.num_samples}",
        ]

        if self.num_train or self.num_test:
            lines.append(f"  Train: {self.num_train}, Test: {self.num_test}")

        lines.extend([
            "",
            f"Controller: {self.controller_type}",
        ])

        if self.seed is not None:
            lines.append(f"Seed: {self.seed}")

        if self.generation_timestamp:
            lines.append(f"Generated: {self.generation_timestamp}")

        if self.notes:
            lines.extend([
                "",
                "Notes:",
                f"  {self.notes}",
            ])

        lines.append("=" * 70)

        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


# =============================================================================
# Utility Functions
# =============================================================================

def _json_serializer(obj):
    """
    Handle numpy types in JSON serialization.

    Args:
        obj: Object to serialize

    Returns:
        JSON-serializable object

    Raises:
        TypeError: If object cannot be serialized
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)

    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def create_metadata_from_problem(
    problem,
    num_samples: int,
    controller_type: str = "lqr",
    seed: Optional[int] = None,
    **kwargs
) -> ControlProblemMetadata:
    """
    Create metadata from a BaseControlProblem instance.

    This is a convenience function to automatically extract
    metadata from a problem instance.

    Args:
        problem: BaseControlProblem instance
        num_samples: Number of samples in dataset
        controller_type: Controller type used
        seed: Random seed
        **kwargs: Additional metadata fields

    Returns:
        ControlProblemMetadata instance

    Example:
        >>> from src.environments import get_problem
        >>> problem = get_problem("double_integrator")
        >>> metadata = create_metadata_from_problem(
        ...     problem,
        ...     num_samples=10000,
        ...     controller_type="minimum_energy",
        ...     seed=42
        ... )
    """
    from .base import BaseControlProblem

    if not isinstance(problem, BaseControlProblem):
        raise TypeError("problem must be a BaseControlProblem instance")

    # Get bounds
    state_lower, state_upper = problem.get_state_bounds()
    control_lower, control_upper = problem.get_control_bounds()

    # Determine problem type
    try:
        problem.get_system_matrices()
        problem_type = "linear"
    except NotImplementedError:
        problem_type = "nonlinear"

    # Get LQR params
    lqr_params = problem.get_lqr_params()

    # Get problem info
    info = problem.get_info()
    problem_specific_params = info.get('physical_params', None)

    metadata = ControlProblemMetadata(
        problem_name=problem.name,
        problem_type=problem_type,
        state_dim=problem.state_dim,
        control_dim=problem.control_dim,
        horizon=problem.horizon,
        dt=problem.dt,
        total_time=problem.dt * problem.horizon,
        state_bounds={
            'lower': state_lower.tolist(),
            'upper': state_upper.tolist()
        },
        control_bounds={
            'lower': control_lower.tolist(),
            'upper': control_upper.tolist()
        },
        num_samples=num_samples,
        lqr_params=_make_json_serializable(lqr_params),
        controller_type=controller_type,
        seed=seed,
        problem_specific_params=problem_specific_params,
        **kwargs
    )

    return metadata


def _make_json_serializable(obj):
    """Recursively convert numpy types to JSON-serializable types."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "ControlProblemMetadata",
    "create_metadata_from_problem",
]
