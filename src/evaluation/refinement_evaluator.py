"""
Refinement Evaluator for Process Supervision

Evaluates how well the model learns to progressively refine its solutions
across iterations. This is crucial for assessing process supervision effectiveness.

Key Metrics:
1. Cost reduction per iteration (do solutions improve?)
2. Convergence rate (how fast do they improve?)
3. Refinement quality (smooth or erratic improvement?)
4. Final vs intermediate performance

Usage:
    from src.evaluation.refinement_evaluator import RefinementEvaluator

    evaluator = RefinementEvaluator(model, problem, device='cuda')
    metrics = evaluator.evaluate(test_loader)
    evaluator.plot_refinement_curves(metrics, output_path='refinement.png')
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RefinementMetrics:
    """Container for refinement evaluation metrics."""
    # Per-iteration costs
    iteration_costs: np.ndarray  # [num_samples, num_iterations]

    # Improvements
    iteration_improvements: np.ndarray  # [num_samples, num_iterations-1]

    # Control MSE per iteration
    control_mse_per_iteration: np.ndarray  # [num_iterations]

    # Summary statistics
    avg_cost_reduction: float  # Average total cost reduction
    avg_convergence_rate: float  # Average rate of improvement
    final_cost: float  # Average final cost
    initial_cost: float  # Average initial cost

    # Iteration-wise stats
    avg_costs_per_iteration: np.ndarray  # [num_iterations]
    std_costs_per_iteration: np.ndarray  # [num_iterations]


class RefinementEvaluator:
    """
    Evaluates model's refinement process across iterations.

    Analyzes how solutions improve from iteration 0 to final iteration.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        problem,
        dynamics_fn: Callable,
        cost_params: Optional[Dict] = None,
        device: str = 'cuda',
    ):
        """
        Args:
            model: Trained TRC model
            problem: Control problem instance
            dynamics_fn: Differentiable dynamics function
            cost_params: Cost function parameters (Q, R, Q_final)
            device: Device for computation
        """
        self.model = model
        self.problem = problem
        self.dynamics_fn = dynamics_fn
        self.device = device

        # Default cost params
        if cost_params is None:
            state_dim = problem.state_dim
            control_dim = problem.control_dim
            self.cost_params = {
                'Q': torch.eye(state_dim, device=device),
                'R': 0.01 * torch.eye(control_dim, device=device),
                'Q_final': 10.0 * torch.eye(state_dim, device=device),
            }
        else:
            self.cost_params = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in cost_params.items()
            }

        self.model.to(device)
        self.model.eval()

    def compute_trajectory_cost(
        self,
        states: torch.Tensor,
        controls: torch.Tensor,
        target_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute LQR-style trajectory cost.

        Args:
            states: State trajectory [batch, horizon+1, state_dim]
            controls: Control sequence [batch, horizon, control_dim]
            target_state: Target state [batch, state_dim]

        Returns:
            cost: Trajectory cost for each sample [batch]
        """
        from ..training.process_supervision import compute_trajectory_cost

        return compute_trajectory_cost(
            states=states,
            controls=controls,
            target_state=target_state,
            **self.cost_params,
        )

    def evaluate(
        self,
        data_loader,
        num_samples: Optional[int] = None,
    ) -> RefinementMetrics:
        """
        Evaluate refinement quality on a dataset.

        Args:
            data_loader: DataLoader with (initial, target, controls_gt, states_gt)
            num_samples: Maximum number of samples to evaluate (None = all)

        Returns:
            RefinementMetrics with detailed analysis
        """
        logger.info("Evaluating refinement quality...")

        all_iteration_costs = []
        all_control_mses = []
        num_evaluated = 0

        with torch.no_grad():
            for batch_data in data_loader:
                # Unpack batch
                if len(batch_data) == 4:
                    initial, target, controls_gt, states_gt = batch_data
                else:
                    initial, target, controls_gt = batch_data
                    states_gt = None

                initial = initial.to(self.device)
                target = target.to(self.device)
                controls_gt = controls_gt.to(self.device)

                # Forward pass with all iterations
                output = self.model(
                    current_state=initial,
                    target_state=target,
                    return_all_iterations=True,
                )

                all_controls = output['all_controls']  # [batch, num_iters, horizon, control_dim]
                batch_size, num_iters, horizon, control_dim = all_controls.shape

                # Compute costs for each iteration
                iteration_costs_batch = []
                control_mse_batch = []

                for k in range(num_iters):
                    controls_k = all_controls[:, k, :, :]  # [batch, horizon, control_dim]

                    # Simulate trajectory
                    states_k = self.dynamics_fn(initial, controls_k)

                    # Compute cost
                    cost_k = self.compute_trajectory_cost(
                        states=states_k,
                        controls=controls_k,
                        target_state=target,
                    )  # [batch]

                    iteration_costs_batch.append(cost_k.cpu().numpy())

                    # Compute control MSE vs ground truth
                    control_mse_k = F.mse_loss(controls_k, controls_gt).item()
                    control_mse_batch.append(control_mse_k)

                # Stack: [batch, num_iters]
                iteration_costs_batch = np.stack(iteration_costs_batch, axis=1)
                all_iteration_costs.append(iteration_costs_batch)
                all_control_mses.append(control_mse_batch)

                num_evaluated += batch_size

                if num_samples is not None and num_evaluated >= num_samples:
                    break

        # Concatenate all batches
        iteration_costs = np.concatenate(all_iteration_costs, axis=0)  # [num_samples, num_iters]
        control_mse_per_iteration = np.mean(all_control_mses, axis=0)  # [num_iters]

        # Compute improvements (cost reduction from k to k+1)
        iteration_improvements = iteration_costs[:, :-1] - iteration_costs[:, 1:]  # [num_samples, num_iters-1]

        # Summary statistics
        avg_cost_reduction = (iteration_costs[:, 0] - iteration_costs[:, -1]).mean()
        avg_convergence_rate = iteration_improvements.mean()
        final_cost = iteration_costs[:, -1].mean()
        initial_cost = iteration_costs[:, 0].mean()

        # Per-iteration statistics
        avg_costs_per_iteration = iteration_costs.mean(axis=0)
        std_costs_per_iteration = iteration_costs.std(axis=0)

        metrics = RefinementMetrics(
            iteration_costs=iteration_costs,
            iteration_improvements=iteration_improvements,
            control_mse_per_iteration=control_mse_per_iteration,
            avg_cost_reduction=avg_cost_reduction,
            avg_convergence_rate=avg_convergence_rate,
            final_cost=final_cost,
            initial_cost=initial_cost,
            avg_costs_per_iteration=avg_costs_per_iteration,
            std_costs_per_iteration=std_costs_per_iteration,
        )

        logger.info("Refinement evaluation complete!")
        logger.info(f"  Samples evaluated: {len(iteration_costs)}")
        logger.info(f"  Iterations: {iteration_costs.shape[1]}")
        logger.info(f"  Initial cost: {initial_cost:.4f}")
        logger.info(f"  Final cost: {final_cost:.4f}")
        logger.info(f"  Cost reduction: {avg_cost_reduction:.4f} ({100 * avg_cost_reduction / initial_cost:.1f}%)")
        logger.info(f"  Avg improvement/iter: {avg_convergence_rate:.4f}")

        return metrics

    def plot_refinement_curves(
        self,
        metrics: RefinementMetrics,
        output_path: str = 'refinement_analysis.png',
        num_examples: int = 10,
    ):
        """
        Plot refinement quality analysis.

        Creates a multi-panel figure showing:
        1. Cost vs iteration (average + examples)
        2. Cost reduction per iteration
        3. Control MSE vs iteration
        4. Improvement distribution

        Args:
            metrics: RefinementMetrics from evaluate()
            output_path: Path to save figure
            num_examples: Number of example trajectories to plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        num_iters = len(metrics.avg_costs_per_iteration)
        iterations = np.arange(num_iters)

        # 1. Cost vs Iteration (top-left)
        ax = axes[0, 0]

        # Plot average cost with std
        ax.plot(iterations, metrics.avg_costs_per_iteration,
                'o-', linewidth=2, markersize=8, label='Average', color='blue')
        ax.fill_between(
            iterations,
            metrics.avg_costs_per_iteration - metrics.std_costs_per_iteration,
            metrics.avg_costs_per_iteration + metrics.std_costs_per_iteration,
            alpha=0.2, color='blue'
        )

        # Plot a few example trajectories
        for i in range(min(num_examples, len(metrics.iteration_costs))):
            ax.plot(iterations, metrics.iteration_costs[i],
                   '-', alpha=0.3, color='gray', linewidth=1)

        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Trajectory Cost', fontsize=12)
        ax.set_title('Cost Reduction Across Iterations', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Cost Improvement per Iteration (top-right)
        ax = axes[0, 1]

        improvements_mean = metrics.iteration_improvements.mean(axis=0)
        improvements_std = metrics.iteration_improvements.std(axis=0)

        ax.bar(np.arange(num_iters - 1), improvements_mean,
               yerr=improvements_std, capsize=5, alpha=0.7, color='green')
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Cost Improvement', fontsize=12)
        ax.set_title('Cost Reduction Per Iteration', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # 3. Control MSE vs Iteration (bottom-left)
        ax = axes[1, 0]

        ax.plot(iterations, metrics.control_mse_per_iteration,
                'o-', linewidth=2, markersize=8, color='orange')
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Control MSE', fontsize=12)
        ax.set_title('Control Accuracy vs Iteration', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        # 4. Improvement Distribution (bottom-right)
        ax = axes[1, 1]

        # Histogram of total cost reductions
        total_reductions = metrics.iteration_costs[:, 0] - metrics.iteration_costs[:, -1]
        ax.hist(total_reductions, bins=30, alpha=0.7, color='purple', edgecolor='black')
        ax.axvline(x=total_reductions.mean(), color='red',
                  linestyle='--', linewidth=2, label=f'Mean: {total_reductions.mean():.3f}')
        ax.set_xlabel('Total Cost Reduction', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Distribution of Cost Reductions', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Refinement analysis plot saved to: {output_path}")

        plt.close()

    def analyze_convergence(
        self,
        metrics: RefinementMetrics,
    ) -> Dict[str, float]:
        """
        Analyze convergence properties of refinement.

        Args:
            metrics: RefinementMetrics from evaluate()

        Returns:
            Dictionary with convergence statistics
        """
        # Compute what percentage of samples improve at each iteration
        improvements = metrics.iteration_improvements  # [num_samples, num_iters-1]
        improvement_rate_per_iter = (improvements > 0).mean(axis=0)  # [num_iters-1]

        # Compute convergence rate (exponential fit)
        # Assumes cost_k ≈ final_cost + (initial_cost - final_cost) * exp(-k/tau)
        costs = metrics.avg_costs_per_iteration
        try:
            from scipy.optimize import curve_fit

            def exponential_decay(k, final_cost, amplitude, tau):
                return final_cost + amplitude * np.exp(-k / tau)

            iterations = np.arange(len(costs))
            popt, _ = curve_fit(
                exponential_decay,
                iterations,
                costs,
                p0=[costs[-1], costs[0] - costs[-1], 1.0],
                maxfev=10000,
            )
            convergence_tau = popt[2]
        except:
            convergence_tau = np.nan

        return {
            'improvement_rate_per_iter': improvement_rate_per_iter,
            'convergence_tau': convergence_tau,
            'pct_samples_improving': (improvements.sum(axis=1) > 0).mean() * 100,
            'pct_monotonic': ((improvements > 0).all(axis=1)).mean() * 100,
        }

    def compare_to_baseline(
        self,
        baseline_cost: float,
        metrics: RefinementMetrics,
    ) -> Dict[str, float]:
        """
        Compare refinement to a baseline (e.g., behavior cloning).

        Args:
            baseline_cost: Average cost of baseline model
            metrics: RefinementMetrics from this model

        Returns:
            Dictionary with comparison statistics
        """
        final_cost = metrics.final_cost
        improvement_vs_baseline = (baseline_cost - final_cost) / baseline_cost * 100

        return {
            'baseline_cost': baseline_cost,
            'process_supervision_cost': final_cost,
            'improvement_pct': improvement_vs_baseline,
            'cost_reduction': baseline_cost - final_cost,
        }
