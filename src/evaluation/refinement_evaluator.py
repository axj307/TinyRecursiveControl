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
        baseline_metrics: 'RefinementMetrics' = None,
    ):
        """
        Plot refinement quality analysis with BC vs PS comparison.

        Creates a redesigned 4-panel figure showing:
        1. Refinement curves (PS vs BC) with explanatory annotations
        2. Final performance comparison (iteration 3 costs)
        3. Total refinement capability (cost reduction comparison)
        4. Distribution analysis with clear annotations

        Args:
            metrics: RefinementMetrics from evaluate() (PS model)
            output_path: Path to save figure
            num_examples: Number of example trajectories to plot
            baseline_metrics: Optional baseline (BC) RefinementMetrics for comparison
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 11))

        num_iters = len(metrics.avg_costs_per_iteration)
        iterations = np.arange(num_iters)

        # 1. Refinement Curves with Explanatory Annotations (top-left)
        ax = axes[0, 0]

        # Plot PS average cost with std
        ax.plot(iterations, metrics.avg_costs_per_iteration,
                'o-', linewidth=2.5, markersize=9, label='PS (Process Supervision)',
                color='#1f77b4', zorder=3)
        ax.fill_between(
            iterations,
            metrics.avg_costs_per_iteration - metrics.std_costs_per_iteration,
            metrics.avg_costs_per_iteration + metrics.std_costs_per_iteration,
            alpha=0.2, color='#1f77b4', zorder=2
        )

        # Plot a few PS example trajectories
        for i in range(min(num_examples, len(metrics.iteration_costs))):
            ax.plot(iterations, metrics.iteration_costs[i],
                   '-', alpha=0.2, color='gray', linewidth=1, zorder=1)

        # Add BC refinement curve if provided
        if baseline_metrics is not None:
            ax.plot(iterations, baseline_metrics.avg_costs_per_iteration,
                   'o-', linewidth=2.5, markersize=9, label='BC (Behavior Cloning)',
                   color='#d62728', alpha=0.9, zorder=3)
            ax.fill_between(
                iterations,
                baseline_metrics.avg_costs_per_iteration - baseline_metrics.std_costs_per_iteration,
                baseline_metrics.avg_costs_per_iteration + baseline_metrics.std_costs_per_iteration,
                alpha=0.15, color='#d62728', zorder=2
            )

            # Add explanatory text box about BC behavior
            textstr = 'BC trains only final iteration (iter 3)\nIntermediate iterations unsupervised\n→ Flat/non-monotonic curve expected'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='#d62728', linewidth=2)
            ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', horizontalalignment='right', bbox=props)

        ax.set_xlabel('Iteration', fontsize=13, fontweight='bold')
        ax.set_ylabel('Trajectory Cost (LQR)', fontsize=13, fontweight='bold')
        ax.set_title('Refinement Behavior: PS vs BC', fontsize=15, fontweight='bold', pad=15)
        ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xticks(iterations)

        # 2. Final Performance Comparison (top-right) - NEW
        ax = axes[0, 1]

        if baseline_metrics is not None:
            ps_final = metrics.final_cost
            bc_final = baseline_metrics.final_cost

            methods = ['PS\n(Process\nSupervision)', 'BC\n(Behavior\nCloning)']
            costs = [ps_final, bc_final]
            colors = ['#1f77b4', '#d62728']

            bars = ax.barh(methods, costs, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

            # Add value labels on bars
            for i, (bar, cost) in enumerate(zip(bars, costs)):
                width = bar.get_width()
                ax.text(width * 1.02, bar.get_y() + bar.get_height()/2,
                       f'{cost:.1f}',
                       ha='left', va='center', fontsize=12, fontweight='bold')

            # Calculate improvement percentage
            if bc_final > 0:
                improvement_pct = ((bc_final - ps_final) / bc_final) * 100
                if improvement_pct > 0:
                    result_text = f'PS achieves {improvement_pct:.1f}% lower cost'
                    color = 'green'
                else:
                    result_text = f'BC achieves {-improvement_pct:.1f}% lower cost'
                    color = 'red'

                ax.text(0.5, -0.15, result_text, transform=ax.transAxes,
                       ha='center', va='top', fontsize=11, fontweight='bold',
                       color=color, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

            ax.set_xlabel('Final Cost (Iteration 3)', fontsize=13, fontweight='bold')
            ax.set_title('Final Performance Comparison', fontsize=15, fontweight='bold', pad=15)
            ax.grid(True, alpha=0.3, axis='x', linestyle='--')
        else:
            # If no baseline, show PS performance only
            ax.text(0.5, 0.5, 'No BC baseline\nprovided',
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=14, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
            ax.set_title('Final Performance', fontsize=15, fontweight='bold', pad=15)

        # 3. Total Refinement Capability (bottom-left) - NEW
        ax = axes[1, 0]

        if baseline_metrics is not None:
            ps_reduction = metrics.avg_cost_reduction
            bc_reduction = baseline_metrics.avg_cost_reduction

            methods = ['PS', 'BC']
            reductions = [ps_reduction, bc_reduction]
            colors = ['#1f77b4', '#d62728']

            bars = ax.bar(methods, reductions, color=colors, alpha=0.7,
                         edgecolor='black', linewidth=1.5, width=0.6)

            # Add value labels on bars
            for bar, reduction in zip(bars, reductions):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height * 1.02,
                       f'{reduction:.1f}',
                       ha='center', va='bottom', fontsize=12, fontweight='bold')

            # Add ratio annotation
            if bc_reduction > 0:
                ratio = ps_reduction / bc_reduction
                ratio_text = f'PS refines {ratio:.1f}× better than BC'
                ax.text(0.5, 0.95, ratio_text, transform=ax.transAxes,
                       ha='center', va='top', fontsize=11, fontweight='bold',
                       color='darkgreen', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6))

            ax.set_ylabel('Total Cost Reduction', fontsize=13, fontweight='bold')
            ax.set_title('Refinement Capability', fontsize=15, fontweight='bold', pad=15)
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
            ax.set_ylim(0, max(reductions) * 1.15)
        else:
            # If no baseline, show PS reduction only
            ps_reduction = metrics.avg_cost_reduction
            ax.bar(['PS'], [ps_reduction], color='#1f77b4', alpha=0.7,
                  edgecolor='black', linewidth=1.5, width=0.4)
            ax.text(0, ps_reduction * 1.02, f'{ps_reduction:.1f}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
            ax.set_ylabel('Total Cost Reduction', fontsize=13, fontweight='bold')
            ax.set_title('Refinement Capability', fontsize=15, fontweight='bold', pad=15)
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')

        # 4. Distribution Analysis (bottom-right) - ENHANCED
        ax = axes[1, 1]

        # Histogram of PS total cost reductions
        ps_total_reductions = metrics.iteration_costs[:, 0] - metrics.iteration_costs[:, -1]
        ax.hist(ps_total_reductions, bins=30, alpha=0.6, color='#1f77b4', edgecolor='black',
               label='PS Distribution', linewidth=0.5)

        ps_mean = ps_total_reductions.mean()
        ax.axvline(x=ps_mean, color='#1f77b4',
                  linestyle='--', linewidth=2.5,
                  label=f'PS Mean = {ps_mean:.1f}', zorder=3)

        # Add BC comparison if provided
        if baseline_metrics is not None:
            bc_total_reductions = baseline_metrics.iteration_costs[:, 0] - baseline_metrics.iteration_costs[:, -1]
            bc_mean = bc_total_reductions.mean()
            ax.axvline(x=bc_mean, color='#d62728',
                      linestyle='--', linewidth=2.5,
                      label=f'BC Mean = {bc_mean:.1f}', zorder=3)

            # Add shaded region between means
            ax.axvspan(min(ps_mean, bc_mean), max(ps_mean, bc_mean),
                      alpha=0.15, color='yellow', zorder=1,
                      label=f'Difference = {abs(ps_mean - bc_mean):.1f}')

        ax.set_xlabel('Total Cost Reduction (Initial → Final)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=13, fontweight='bold')
        title = 'Distribution of Cost Reductions'
        if baseline_metrics is not None:
            title += ' (PS vs BC)'
        ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Refinement comparison plot saved to: {output_path}")

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
