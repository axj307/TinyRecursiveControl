"""
Environment Registry

Central registry for all control problems. This allows easy discovery and
instantiation of control problems by name.

Usage:
    >>> from src.environments import get_problem, list_problems
    >>> problem = get_problem("double_integrator", dt=0.1, horizon=20)
    >>> print(list_problems())
"""

from .base import BaseControlProblem
from .double_integrator import DoubleIntegrator
from .pendulum import Pendulum
from .vanderpol import VanderpolOscillator


# =============================================================================
# Problem Registry
# =============================================================================

PROBLEM_REGISTRY = {
    "double_integrator": DoubleIntegrator,
    "pendulum": Pendulum,
    "vanderpol": VanderpolOscillator,
}


# =============================================================================
# Factory Functions
# =============================================================================

def get_problem(name: str, **kwargs) -> BaseControlProblem:
    """
    Get a control problem instance by name.

    This is the main factory function for creating control problems.
    All problem-specific parameters are passed as kwargs.

    Args:
        name: Problem name (must be in PROBLEM_REGISTRY)
        **kwargs: Problem-specific parameters (dt, horizon, etc.)

    Returns:
        Instantiated BaseControlProblem subclass

    Raises:
        ValueError: If problem name is not registered

    Examples:
        >>> # Create double integrator with default parameters
        >>> problem = get_problem("double_integrator")

        >>> # Create with custom parameters
        >>> problem = get_problem("double_integrator",
        ...                       dt=0.1,
        ...                       horizon=20,
        ...                       control_bounds=5.0)

        >>> # Create pendulum
        >>> problem = get_problem("pendulum",
        ...                       dt=0.05,
        ...                       mass=2.0,
        ...                       length=0.5)
    """
    if name not in PROBLEM_REGISTRY:
        available = ", ".join(sorted(PROBLEM_REGISTRY.keys()))
        raise ValueError(
            f"Unknown problem '{name}'. "
            f"Available problems: {available}"
        )

    problem_class = PROBLEM_REGISTRY[name]
    return problem_class(**kwargs)


def list_problems() -> list:
    """
    Return list of available problem names.

    Returns:
        List of registered problem names

    Example:
        >>> problems = list_problems()
        >>> print(problems)
        ['double_integrator', 'pendulum']
    """
    return sorted(PROBLEM_REGISTRY.keys())


def get_problem_class(name: str):
    """
    Get the problem class (not instance) by name.

    This is useful for introspection or when you want to create
    instances manually.

    Args:
        name: Problem name

    Returns:
        Problem class (not instantiated)

    Raises:
        ValueError: If problem name is not registered

    Example:
        >>> cls = get_problem_class("double_integrator")
        >>> print(cls.__name__)
        'DoubleIntegrator'
    """
    if name not in PROBLEM_REGISTRY:
        available = ", ".join(sorted(PROBLEM_REGISTRY.keys()))
        raise ValueError(
            f"Unknown problem '{name}'. "
            f"Available problems: {available}"
        )

    return PROBLEM_REGISTRY[name]


def register_problem(name: str, problem_class: type):
    """
    Register a new control problem.

    This allows external code to add new problems to the registry
    without modifying this file.

    Args:
        name: Problem name (must be unique)
        problem_class: Class inheriting from BaseControlProblem

    Raises:
        ValueError: If name is already registered
        TypeError: If class doesn't inherit from BaseControlProblem

    Example:
        >>> from src.environments import register_problem, BaseControlProblem
        >>>
        >>> class MyProblem(BaseControlProblem):
        ...     # Implementation...
        ...     pass
        >>>
        >>> register_problem("my_problem", MyProblem)
    """
    if name in PROBLEM_REGISTRY:
        raise ValueError(f"Problem '{name}' is already registered")

    if not issubclass(problem_class, BaseControlProblem):
        raise TypeError(
            f"Problem class must inherit from BaseControlProblem, "
            f"got {problem_class}"
        )

    PROBLEM_REGISTRY[name] = problem_class


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Base class
    "BaseControlProblem",

    # Problem classes
    "DoubleIntegrator",
    "Pendulum",
    "VanderpolOscillator",

    # Registry
    "PROBLEM_REGISTRY",

    # Factory functions
    "get_problem",
    "list_problems",
    "get_problem_class",
    "register_problem",
]
