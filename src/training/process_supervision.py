"""
Process Supervision for TRM-Style Training

Implements process-level supervision that trains the model on intermediate
refinement steps, not just the final output. This is inspired by TRM's approach
of supervising the reasoning process, adapted for continuous control.

Key Idea:
- Instead of: loss = MSE(final_controls, optimal_controls)
- We use: loss = final_accuracy + Σ(improvement_rewards)

Where improvement_rewards encourage the model to progressively refine
its solution across iterations.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Callable
import logging

logger = logging.getLogger(__name__)


def compute_trajectory_cost(
    states: torch.Tensor,
    controls: torch.Tensor,
    target_state: torch.Tensor,
    Q: Optional[torch.Tensor] = None,
    R: Optional[torch.Tensor] = None,
    Q_final: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute LQR-style trajectory cost.

    Cost = Σ_t (state_error' Q state_error + control' R control) + final_error' Q_final final_error

    Args:
        states: State trajectory [batch, horizon+1, state_dim]
        controls: Control sequence [batch, horizon, control_dim]
        target_state: Target state [batch, state_dim]
        Q: State cost matrix [state_dim, state_dim] or scalar
        R: Control cost matrix [control_dim, control_dim] or scalar
        Q_final: Final state cost matrix [state_dim, state_dim] or scalar

    Returns:
        cost: Trajectory cost for each sample [batch]
    """
    batch_size = states.shape[0]
    horizon = controls.shape[1]
    state_dim = states.shape[2]
    control_dim = controls.shape[2]
    device = states.device

    # Default cost matrices (identity)
    if Q is None:
        Q = torch.eye(state_dim, device=device)
    elif isinstance(Q, (int, float)):
        Q = Q * torch.eye(state_dim, device=device)
    else:
        Q = Q.to(device)  # Move existing tensor to device

    if R is None:
        R = 0.01 * torch.eye(control_dim, device=device)
    elif isinstance(R, (int, float)):
        R = R * torch.eye(control_dim, device=device)
    else:
        R = R.to(device)  # Move existing tensor to device

    if Q_final is None:
        Q_final = 10.0 * Q  # Final cost is higher
    elif isinstance(Q_final, (int, float)):
        Q_final = Q_final * torch.eye(state_dim, device=device)
    else:
        Q_final = Q_final.to(device)  # Move existing tensor to device

    # Compute running cost
    total_cost = torch.zeros(batch_size, device=device)

    for t in range(horizon):
        # State error at time t
        state_error = states[:, t, :] - target_state  # [batch, state_dim]

        # Quadratic state cost: e' Q e
        state_cost = torch.einsum('bi,ij,bj->b', state_error, Q, state_error)

        # Control cost: u' R u
        control = controls[:, t, :]  # [batch, control_dim]
        control_cost = torch.einsum('bi,ij,bj->b', control, R, control)

        total_cost += state_cost + control_cost

    # Final state cost
    final_error = states[:, -1, :] - target_state
    final_cost = torch.einsum('bi,ij,bj->b', final_error, Q_final, final_error)
    total_cost += final_cost

    return total_cost


def compute_process_supervision_loss(
    model_output: Dict[str, torch.Tensor],
    target_controls: torch.Tensor,
    initial_state: torch.Tensor,
    target_state: torch.Tensor,
    dynamics_fn: Callable,
    process_weight: float = 0.1,
    cost_params: Optional[Dict] = None,
    normalize_costs: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Compute process supervision loss.

    Supervises ALL refinement iterations, not just the final output.
    Rewards the model for progressively improving its solution.

    Args:
        model_output: Dictionary from model forward pass, must contain:
            - 'controls': Final control sequence [batch, horizon, control_dim]
            - 'all_controls': All iterations [batch, num_iters, horizon, control_dim]
        target_controls: Ground truth optimal controls [batch, horizon, control_dim]
        initial_state: Initial state [batch, state_dim]
        target_state: Target state [batch, state_dim]
        dynamics_fn: Function to simulate trajectory: (state, controls) -> states
        process_weight: Weight for process rewards (λ in the loss equation)
        cost_params: Dictionary with Q, R, Q_final matrices
        normalize_costs: Whether to normalize costs by first iteration cost

    Returns:
        Dictionary containing:
            - 'total_loss': Combined loss
            - 'final_control_loss': MSE on final controls
            - 'process_reward': Average improvement reward
            - 'iteration_costs': Costs at each iteration [batch, num_iters]
            - 'improvements': Cost improvements [batch, num_iters-1]
    """
    # Extract all control iterations
    all_controls = model_output['all_controls']  # [batch, num_iters, horizon, control_dim]
    final_controls = model_output['controls']     # [batch, horizon, control_dim]

    batch_size, num_iters, horizon, control_dim = all_controls.shape

    # Cost parameters
    if cost_params is None:
        cost_params = {}

    # 1. Final control accuracy loss (behavior cloning component)
    final_control_loss = F.mse_loss(final_controls, target_controls)

    # 2. Compute trajectory costs for each iteration
    iteration_costs = []

    for k in range(num_iters):
        # Get controls at iteration k
        controls_k = all_controls[:, k, :, :]  # [batch, horizon, control_dim]

        # Simulate trajectory with these controls
        states_k = dynamics_fn(initial_state, controls_k)  # [batch, horizon+1, state_dim]

        # Compute LQR cost
        cost_k = compute_trajectory_cost(
            states=states_k,
            controls=controls_k,
            target_state=target_state,
            **cost_params
        )  # [batch]

        iteration_costs.append(cost_k)

    # Stack costs: [batch, num_iters]
    iteration_costs = torch.stack(iteration_costs, dim=1)

    # Normalize costs by first iteration (optional, for stability)
    if normalize_costs:
        # Avoid division by zero
        first_iter_cost = iteration_costs[:, 0:1] + 1e-6
        normalized_costs = iteration_costs / first_iter_cost
    else:
        normalized_costs = iteration_costs

    # 3. Compute improvement rewards
    # Improvement from iteration k-1 to k: cost[k-1] - cost[k]
    # Positive value means improvement (cost decreased)
    improvements = normalized_costs[:, :-1] - normalized_costs[:, 1:]  # [batch, num_iters-1]

    # Average improvement per iteration
    avg_improvement = improvements.mean()

    # 4. Process supervision reward
    # We want to maximize improvements, so we negate (since we minimize loss)
    # Loss decreases when improvements are positive (cost reduces)
    process_reward = -avg_improvement

    # 5. Combined loss
    total_loss = final_control_loss + process_weight * process_reward

    # Return detailed metrics
    return {
        'total_loss': total_loss,
        'final_control_loss': final_control_loss,
        'process_reward': process_reward,
        'avg_improvement': avg_improvement,
        'iteration_costs': iteration_costs,  # [batch, num_iters]
        'improvements': improvements,         # [batch, num_iters-1]
    }


def compute_value_prediction_loss(
    model_output: Dict[str, torch.Tensor],
    initial_state: torch.Tensor,
    target_state: torch.Tensor,
    dynamics_fn: Callable,
    value_predictor: Callable,
    cost_params: Optional[Dict] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute value prediction loss for cost predictor head.

    Trains a value function to predict trajectory cost from latent state.
    This can be used for reward shaping and adaptive halting.

    Args:
        model_output: Dictionary from model forward pass, must contain:
            - 'controls': Final control sequence
            - 'final_latent': Final latent state [batch, latent_dim]
            - 'all_latents': All latent states [batch, num_iters, latent_dim] (optional)
        initial_state: Initial state [batch, state_dim]
        target_state: Target state [batch, state_dim]
        dynamics_fn: Function to simulate trajectory
        value_predictor: Neural network that predicts cost from latent state
        cost_params: Dictionary with Q, R, Q_final matrices

    Returns:
        Dictionary containing:
            - 'value_loss': MSE between predicted and actual costs
            - 'predicted_cost': Predicted cost [batch]
            - 'actual_cost': Actual cost [batch]
    """
    if cost_params is None:
        cost_params = {}

    # Get final latent state
    final_latent = model_output['final_latent']  # [batch, latent_dim]
    final_controls = model_output['controls']     # [batch, horizon, control_dim]

    # Predict cost from latent state
    predicted_cost = value_predictor(final_latent).squeeze(-1)  # [batch]

    # Compute actual cost by simulating trajectory
    states = dynamics_fn(initial_state, final_controls)
    actual_cost = compute_trajectory_cost(
        states=states,
        controls=final_controls,
        target_state=target_state,
        **cost_params
    )  # [batch]

    # MSE loss between predicted and actual
    value_loss = F.mse_loss(predicted_cost, actual_cost.detach())

    return {
        'value_loss': value_loss,
        'predicted_cost': predicted_cost,
        'actual_cost': actual_cost,
    }


def compute_combined_supervision_loss(
    model_output: Dict[str, torch.Tensor],
    target_controls: torch.Tensor,
    initial_state: torch.Tensor,
    target_state: torch.Tensor,
    dynamics_fn: Callable,
    process_weight: float = 0.1,
    value_predictor: Optional[Callable] = None,
    value_weight: float = 0.01,
    cost_params: Optional[Dict] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute combined loss with process supervision and optional value prediction.

    Args:
        model_output: Dictionary from model forward pass
        target_controls: Ground truth optimal controls
        initial_state: Initial state
        target_state: Target state
        dynamics_fn: Function to simulate trajectory
        process_weight: Weight for process supervision
        value_predictor: Optional value function network
        value_weight: Weight for value prediction loss
        cost_params: Dictionary with Q, R, Q_final matrices

    Returns:
        Dictionary with all loss components and metrics
    """
    # Compute process supervision loss
    ps_metrics = compute_process_supervision_loss(
        model_output=model_output,
        target_controls=target_controls,
        initial_state=initial_state,
        target_state=target_state,
        dynamics_fn=dynamics_fn,
        process_weight=process_weight,
        cost_params=cost_params,
    )

    total_loss = ps_metrics['total_loss']
    metrics = ps_metrics

    # Add value prediction loss if value predictor provided
    if value_predictor is not None:
        value_metrics = compute_value_prediction_loss(
            model_output=model_output,
            initial_state=initial_state,
            target_state=target_state,
            dynamics_fn=dynamics_fn,
            value_predictor=value_predictor,
            cost_params=cost_params,
        )

        total_loss = total_loss + value_weight * value_metrics['value_loss']
        metrics['value_loss'] = value_metrics['value_loss']
        metrics['predicted_cost'] = value_metrics['predicted_cost']
        metrics['actual_cost'] = value_metrics['actual_cost']
        metrics['total_loss'] = total_loss

    return metrics
