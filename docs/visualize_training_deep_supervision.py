#!/usr/bin/env python3
"""
Visualization tools for monitoring deep supervision training.

Features:
- Real-time training dashboard
- Per-step loss visualization
- Gradient flow analysis
- Refinement quality metrics
- Curriculum learning progress
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation
from pathlib import Path
from typing import Dict, List, Optional
import json
from tensorboard.backend.event_processing import event_accumulator


class DeepSupervisionMonitor:
    """Real-time monitoring for deep supervision training."""

    def __init__(self, log_dir: str, num_steps: int = 3):
        """
        Args:
            log_dir: Path to tensorboard logs
            num_steps: Number of H_cycle steps to monitor
        """
        self.log_dir = Path(log_dir)
        self.num_steps = num_steps
        self.ea = None

    def load_tensorboard_logs(self):
        """Load tensorboard event files."""
        try:
            self.ea = event_accumulator.EventAccumulator(str(self.log_dir))
            self.ea.Reload()
            print(f"Loaded logs from {self.log_dir}")
            print(f"Available tags: {len(self.ea.Tags()['scalars'])} scalar tags")
        except Exception as e:
            print(f"Error loading logs: {e}")
            return False
        return True

    def get_scalar_data(self, tag: str) -> tuple:
        """Get time series data for a scalar tag."""
        if self.ea is None:
            return np.array([]), np.array([])

        try:
            events = self.ea.Scalars(tag)
            steps = np.array([e.step for e in events])
            values = np.array([e.value for e in events])
            return steps, values
        except:
            return np.array([]), np.array([])

    def visualize_comprehensive(self, save_path: Optional[str] = None):
        """
        Create comprehensive visualization dashboard.
        """
        if not self.load_tensorboard_logs():
            print("Failed to load logs. Make sure training has started.")
            return

        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)

        # ====================================================================
        # Row 1: Loss Curves
        # ====================================================================

        # 1. Total losses
        ax1 = fig.add_subplot(gs[0, 0])
        steps, total_loss = self.get_scalar_data("train/total_loss")
        _, act_loss = self.get_scalar_data("train/act_loss")
        _, sup_loss = self.get_scalar_data("train/supervised_loss")

        if len(steps) > 0:
            ax1.plot(steps, total_loss, label="Total Loss", linewidth=2)
            ax1.plot(steps, act_loss, label="ACT Loss", linewidth=1.5, alpha=0.7)
            ax1.plot(steps, sup_loss, label="Supervised Loss", linewidth=1.5, alpha=0.7)
            ax1.set_xlabel("Training Step")
            ax1.set_ylabel("Loss")
            ax1.set_title("Total Loss Components")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, "No data yet", ha='center', va='center')
            ax1.set_title("Total Loss Components")

        # 2. Per-step losses
        ax2 = fig.add_subplot(gs[0, 1])
        colors = plt.cm.viridis(np.linspace(0, 1, self.num_steps))

        for i in range(self.num_steps):
            steps_i, loss_i = self.get_scalar_data(f"train/deep_supervision/step_{i}_loss")
            if len(steps_i) > 0:
                ax2.plot(steps_i, loss_i, label=f"H_cycle {i}",
                        color=colors[i], linewidth=2)

        ax2.set_xlabel("Training Step")
        ax2.set_ylabel("Loss")
        ax2.set_title("Per-Step Losses (Refinement)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Loss delta (refinement quality)
        ax3 = fig.add_subplot(gs[0, 2])
        # Compute improvement from step 0 to final step
        steps_0, loss_0 = self.get_scalar_data("train/deep_supervision/step_0_loss")
        steps_final, loss_final = self.get_scalar_data(f"train/deep_supervision/step_{self.num_steps-1}_loss")

        if len(steps_0) > 0 and len(steps_final) > 0:
            # Align arrays
            min_len = min(len(loss_0), len(loss_final))
            improvement = loss_0[:min_len] - loss_final[:min_len]
            ax3.plot(steps_0[:min_len], improvement, linewidth=2, color='green')
            ax3.axhline(0, color='red', linestyle='--', alpha=0.5)
            ax3.fill_between(steps_0[:min_len], 0, improvement, alpha=0.3, color='green')
            ax3.set_xlabel("Training Step")
            ax3.set_ylabel("Loss Improvement")
            ax3.set_title("Refinement Quality\n(Step 0 - Final)")
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, "No data yet", ha='center', va='center')
            ax3.set_title("Refinement Quality")

        # 4. Curriculum weight
        ax4 = fig.add_subplot(gs[0, 3])
        steps, curr_weight = self.get_scalar_data("train/curriculum_weight")
        if len(steps) > 0:
            ax4.plot(steps, curr_weight, linewidth=2, color='purple')
            ax4.set_xlabel("Training Step")
            ax4.set_ylabel("Supervision Weight")
            ax4.set_title("Curriculum Learning Progress")
            ax4.set_ylim([0, 1.0])
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, "No curriculum\nor no data yet", ha='center', va='center')
            ax4.set_title("Curriculum Learning")

        # ====================================================================
        # Row 2: Gradient Analysis
        # ====================================================================

        # 5. Gradient norms over time
        ax5 = fig.add_subplot(gs[1, 0])
        steps, grad_norm = self.get_scalar_data("train/grad_norm")
        if len(steps) > 0:
            ax5.plot(steps, grad_norm, linewidth=1, alpha=0.7)
            ax5.set_xlabel("Training Step")
            ax5.set_ylabel("Gradient Norm")
            ax5.set_title("Total Gradient Norm\n(clipped at 1.0)")
            ax5.axhline(1.0, color='red', linestyle='--', label='Clip threshold')
            ax5.set_yscale('log')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, "No data yet", ha='center', va='center')
            ax5.set_title("Gradient Norm")

        # 6. Per-step gradient norms (if available)
        ax6 = fig.add_subplot(gs[1, 1])
        grad_data_available = False
        for i in range(self.num_steps):
            steps_i, grad_i = self.get_scalar_data(f"train/gradients/step_{i}_grad_norm_mean")
            if len(steps_i) > 0:
                ax6.plot(steps_i, grad_i, label=f"Step {i}", color=colors[i], linewidth=2)
                grad_data_available = True

        if grad_data_available:
            ax6.set_xlabel("Training Step")
            ax6.set_ylabel("Mean Gradient Norm")
            ax6.set_title("Per-Step Gradient Norms")
            ax6.legend()
            ax6.set_yscale('log')
            ax6.grid(True, alpha=0.3)
        else:
            ax6.text(0.5, 0.5, "Gradient stats\nnot enabled", ha='center', va='center')
            ax6.set_title("Per-Step Gradients")

        # 7. Learning rate schedule
        ax7 = fig.add_subplot(gs[1, 2])
        steps, lr = self.get_scalar_data("train/learning_rate")
        if len(steps) > 0:
            ax7.plot(steps, lr, linewidth=2, color='orange')
            ax7.set_xlabel("Training Step")
            ax7.set_ylabel("Learning Rate")
            ax7.set_title("Learning Rate Schedule")
            ax7.set_yscale('log')
            ax7.grid(True, alpha=0.3)
        else:
            ax7.text(0.5, 0.5, "No data yet", ha='center', va='center')
            ax7.set_title("Learning Rate")

        # 8. ACT metrics
        ax8 = fig.add_subplot(gs[1, 3])
        steps, mean_steps = self.get_scalar_data("train/act/mean_steps")
        if len(steps) > 0:
            ax8.plot(steps, mean_steps, linewidth=2, color='brown')
            ax8.set_xlabel("Training Step")
            ax8.set_ylabel("Mean ACT Steps")
            ax8.set_title("Adaptive Computation Time\nMean Steps")
            ax8.grid(True, alpha=0.3)
        else:
            ax8.text(0.5, 0.5, "No ACT data", ha='center', va='center')
            ax8.set_title("ACT Mean Steps")

        # ====================================================================
        # Row 3: Analysis & Summary
        # ====================================================================

        # 9. Loss landscape (recent window)
        ax9 = fig.add_subplot(gs[2, 0])
        window = 1000  # Last 1000 steps

        loss_matrix = []
        step_labels = []
        for i in range(self.num_steps):
            _, loss_i = self.get_scalar_data(f"train/deep_supervision/step_{i}_loss")
            if len(loss_i) > 0:
                loss_matrix.append(loss_i[-window:])
                step_labels.append(f"Step {i}")

        if loss_matrix:
            loss_matrix = np.array(loss_matrix)
            im = ax9.imshow(loss_matrix, aspect='auto', cmap='hot', interpolation='nearest')
            ax9.set_xlabel("Training Step (last 1000)")
            ax9.set_ylabel("H_cycle Step")
            ax9.set_title("Loss Landscape (Recent)")
            ax9.set_yticks(range(len(step_labels)))
            ax9.set_yticklabels(step_labels)
            plt.colorbar(im, ax=ax9, label='Loss')
        else:
            ax9.text(0.5, 0.5, "No data yet", ha='center', va='center')
            ax9.set_title("Loss Landscape")

        # 10. Refinement progression
        ax10 = fig.add_subplot(gs[2, 1])
        # Show how losses converge to each other
        if len(loss_matrix) > 0:
            # Compute std across steps at each time point
            loss_std = np.std(loss_matrix, axis=0)
            ax10.plot(loss_std, linewidth=2, color='teal')
            ax10.set_xlabel("Training Step (last 1000)")
            ax10.set_ylabel("Loss Std Dev Across Steps")
            ax10.set_title("Step Convergence\n(Lower = more uniform)")
            ax10.grid(True, alpha=0.3)
        else:
            ax10.text(0.5, 0.5, "No data yet", ha='center', va='center')
            ax10.set_title("Step Convergence")

        # 11. Current step losses (bar chart)
        ax11 = fig.add_subplot(gs[2, 2])
        current_losses = []
        for i in range(self.num_steps):
            _, loss_i = self.get_scalar_data(f"train/deep_supervision/step_{i}_loss")
            if len(loss_i) > 0:
                current_losses.append(loss_i[-1])

        if current_losses:
            bars = ax11.bar(range(len(current_losses)), current_losses, color=colors)
            ax11.set_xlabel("H_cycle Step")
            ax11.set_ylabel("Current Loss")
            ax11.set_title("Latest Per-Step Losses")
            ax11.set_xticks(range(len(current_losses)))
            ax11.set_xticklabels([f"Step {i}" for i in range(len(current_losses))])
            ax11.grid(True, alpha=0.3, axis='y')

            # Annotate values
            for i, (bar, val) in enumerate(zip(bars, current_losses)):
                ax11.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                         f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        else:
            ax11.text(0.5, 0.5, "No data yet", ha='center', va='center')
            ax11.set_title("Current Step Losses")

        # 12. Summary statistics
        ax12 = fig.add_subplot(gs[2, 3])
        ax12.axis('off')

        # Compute summary
        if len(steps) > 0 and current_losses:
            total_steps = steps[-1]
            current_total_loss = total_loss[-1] if len(total_loss) > 0 else 0
            current_sup_loss = sup_loss[-1] if len(sup_loss) > 0 else 0
            refinement = current_losses[0] - current_losses[-1] if len(current_losses) > 1 else 0

            summary_text = f"""
    TRAINING SUMMARY
    {'='*40}

    Progress:
      • Total steps: {total_steps:,}
      • Current total loss: {current_total_loss:.4f}
      • Supervised loss: {current_sup_loss:.4f}

    Deep Supervision:
      • Active steps: {len(current_losses)}
      • Step 0 loss: {current_losses[0]:.4f}
      • Final loss: {current_losses[-1]:.4f}
      • Refinement: {refinement:.4f}

    Health Check:
      {'✓' if refinement > 0 else '✗'} Positive refinement
      {'✓' if np.all(np.diff(current_losses) <= 0) else '✗'} Monotonic improvement
      {'✓' if current_total_loss < 3.0 else '✗'} Loss in good range

    Status: {"🟢 Healthy" if refinement > 0 else "🟡 Check refinement"}
            """

            ax12.text(0.05, 0.95, summary_text, transform=ax12.transAxes,
                     fontsize=9, verticalalignment='top', family='monospace',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        else:
            ax12.text(0.5, 0.5, "Training not started\nor no data available",
                     ha='center', va='center', fontsize=12)

        # Main title
        fig.suptitle(f'Deep Supervision Training Dashboard\n{self.log_dir.name}',
                    fontsize=16, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")

        plt.show()


def compare_experiments(log_dirs: List[str], save_path: Optional[str] = None):
    """
    Compare multiple experiments side by side.

    Args:
        log_dirs: List of paths to tensorboard log directories
        save_path: Path to save comparison plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Experiment Comparison: Deep Supervision Configs", fontsize=14, fontweight='bold')

    colors = plt.cm.tab10(np.linspace(0, 1, len(log_dirs)))

    for i, log_dir in enumerate(log_dirs):
        monitor = DeepSupervisionMonitor(log_dir)
        if not monitor.load_tensorboard_logs():
            print(f"Skipping {log_dir} - no data")
            continue

        label = Path(log_dir).name

        # 1. Total loss
        steps, total_loss = monitor.get_scalar_data("train/total_loss")
        if len(steps) > 0:
            axes[0, 0].plot(steps, total_loss, label=label, color=colors[i], linewidth=2)

        # 2. Supervised loss
        steps, sup_loss = monitor.get_scalar_data("train/supervised_loss")
        if len(steps) > 0:
            axes[0, 1].plot(steps, sup_loss, label=label, color=colors[i], linewidth=2)

        # 3. Refinement quality (step 0 - final)
        steps_0, loss_0 = monitor.get_scalar_data("train/deep_supervision/step_0_loss")
        steps_final, loss_final = monitor.get_scalar_data("train/deep_supervision/step_2_loss")
        if len(steps_0) > 0 and len(steps_final) > 0:
            min_len = min(len(loss_0), len(loss_final))
            improvement = loss_0[:min_len] - loss_final[:min_len]
            axes[1, 0].plot(steps_0[:min_len], improvement, label=label, color=colors[i], linewidth=2)

        # 4. Learning efficiency (steps to reach loss < 2.0)
        if len(total_loss) > 0:
            threshold = 2.0
            below_threshold = np.where(total_loss < threshold)[0]
            if len(below_threshold) > 0:
                steps_to_threshold = steps[below_threshold[0]]
                axes[1, 1].bar(i, steps_to_threshold, color=colors[i], label=label)

    # Configure axes
    axes[0, 0].set_xlabel("Training Step")
    axes[0, 0].set_ylabel("Total Loss")
    axes[0, 0].set_title("Total Loss Comparison")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel("Training Step")
    axes[0, 1].set_ylabel("Supervised Loss")
    axes[0, 1].set_title("Deep Supervision Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_xlabel("Training Step")
    axes[1, 0].set_ylabel("Refinement (Loss Improvement)")
    axes[1, 0].set_title("Refinement Quality")
    axes[1, 0].axhline(0, color='red', linestyle='--', alpha=0.5)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_xlabel("Experiment")
    axes[1, 1].set_ylabel("Steps to Loss < 2.0")
    axes[1, 1].set_title("Convergence Speed")
    axes[1, 1].set_xticks(range(len(log_dirs)))
    axes[1, 1].set_xticklabels([Path(d).name for d in log_dirs], rotation=45, ha='right')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to {save_path}")

    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize deep supervision training")
    parser.add_argument("--log_dir", type=str, default="runs/deep_supervision_linear",
                       help="Path to tensorboard logs")
    parser.add_argument("--num_steps", type=int, default=3,
                       help="Number of H_cycle steps")
    parser.add_argument("--compare", nargs="+", default=None,
                       help="Compare multiple experiments")
    parser.add_argument("--save", type=str, default=None,
                       help="Save visualization to file")

    args = parser.parse_args()

    if args.compare:
        print(f"Comparing {len(args.compare)} experiments...")
        compare_experiments(args.compare, save_path=args.save)
    else:
        print(f"Visualizing training from: {args.log_dir}")
        monitor = DeepSupervisionMonitor(args.log_dir, num_steps=args.num_steps)
        monitor.visualize_comprehensive(save_path=args.save)
