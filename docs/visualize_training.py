#!/usr/bin/env python3
"""
Visualize Training Metrics from CSV Logs

This script reads the CSV log files generated during training
and creates visualization plots for analysis.

Usage:
    python visualize_training.py --log_dir ./outputs/llada_trm_warmup
    python visualize_training.py --log_dir ./outputs/llada_trm_warmup --smooth 10
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def smooth_curve(values, window_size=10):
    """Apply moving average smoothing"""
    if window_size <= 1:
        return values
    return pd.Series(values).rolling(window=window_size, min_periods=1).mean()


def plot_training_metrics(csv_path, output_dir, smooth_window=10):
    """Plot training metrics from CSV file"""

    # Read CSV
    df = pd.read_csv(csv_path)

    print(f"\nLoaded {len(df)} training steps from {csv_path}")
    print(f"Columns: {list(df.columns)}")

    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('LLaDA-TRM Hybrid Training Metrics', fontsize=16, fontweight='bold')

    # 1. Loss
    ax = axes[0, 0]
    if 'loss' in df.columns:
        ax.plot(df['global_step'], df['loss'], alpha=0.3, label='Raw', color='blue')
        smoothed = smooth_curve(df['loss'], smooth_window)
        ax.plot(df['global_step'], smoothed, label=f'Smoothed (window={smooth_window})', color='blue', linewidth=2)
        ax.set_xlabel('Global Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 2. Refinement Steps
    ax = axes[0, 1]
    if 'refinement_steps' in df.columns:
        ax.plot(df['global_step'], df['refinement_steps'], alpha=0.3, label='Raw', color='green')
        smoothed = smooth_curve(df['refinement_steps'], smooth_window)
        ax.plot(df['global_step'], smoothed, label=f'Smoothed (window={smooth_window})', color='green', linewidth=2)
        ax.set_xlabel('Global Step')
        ax.set_ylabel('Steps')
        ax.set_title('Refinement Steps per Chunk')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 3. Chunk Confidence
    ax = axes[1, 0]
    if 'chunk_confidence' in df.columns:
        ax.plot(df['global_step'], df['chunk_confidence'], alpha=0.3, label='Raw', color='orange')
        smoothed = smooth_curve(df['chunk_confidence'], smooth_window)
        ax.plot(df['global_step'], smoothed, label=f'Smoothed (window={smooth_window})', color='orange', linewidth=2)
        ax.set_xlabel('Global Step')
        ax.set_ylabel('Confidence')
        ax.set_title('Chunk Confidence')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 4. Learning Rate
    ax = axes[1, 1]
    if 'learning_rate' in df.columns:
        ax.plot(df['global_step'], df['learning_rate'], color='red', linewidth=2)
        ax.set_xlabel('Global Step')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.grid(True, alpha=0.3)

    # 5. Gradient Norm
    ax = axes[2, 0]
    if 'grad_norm' in df.columns:
        ax.plot(df['global_step'], df['grad_norm'], alpha=0.3, label='Raw', color='purple')
        smoothed = smooth_curve(df['grad_norm'], smooth_window)
        ax.plot(df['global_step'], smoothed, label=f'Smoothed (window={smooth_window})', color='purple', linewidth=2)
        ax.set_xlabel('Global Step')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Gradient Norm (after clipping)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 6. Average Loss (cumulative)
    ax = axes[2, 1]
    if 'avg_loss' in df.columns:
        ax.plot(df['global_step'], df['avg_loss'], color='cyan', linewidth=2)
        ax.set_xlabel('Global Step')
        ax.set_ylabel('Average Loss')
        ax.set_title('Cumulative Average Loss')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = output_dir / 'training_metrics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Training plot saved to: {output_path}")

    plt.close()


def plot_validation_metrics(csv_path, output_dir):
    """Plot validation metrics from CSV file"""

    # Read CSV
    df = pd.read_csv(csv_path)

    print(f"\nLoaded {len(df)} validation epochs from {csv_path}")
    print(f"Columns: {list(df.columns)}")

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('LLaDA-TRM Hybrid Validation Metrics', fontsize=16, fontweight='bold')

    # 1. Validation Loss
    ax = axes[0]
    if 'val_loss' in df.columns:
        ax.plot(df['epoch'], df['val_loss'], 'o-', color='blue', linewidth=2, markersize=8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation Loss')
        ax.set_title('Validation Loss per Epoch')
        ax.grid(True, alpha=0.3)

    # 2. Validation Refinement Steps
    ax = axes[1]
    if 'val_refinement_steps' in df.columns:
        ax.plot(df['epoch'], df['val_refinement_steps'], 'o-', color='green', linewidth=2, markersize=8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Refinement Steps')
        ax.set_title('Validation Refinement Steps')
        ax.grid(True, alpha=0.3)

    # 3. Validation Confidence
    ax = axes[2]
    if 'val_chunk_confidence' in df.columns:
        ax.plot(df['epoch'], df['val_chunk_confidence'], 'o-', color='orange', linewidth=2, markersize=8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Confidence')
        ax.set_title('Validation Chunk Confidence')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = output_dir / 'validation_metrics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Validation plot saved to: {output_path}")

    plt.close()


def generate_summary_stats(train_csv, val_csv, output_dir):
    """Generate summary statistics"""

    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)

    # Training stats
    if train_csv.exists():
        df_train = pd.read_csv(train_csv)
        print("\nTraining Metrics:")
        print(f"  Total steps: {len(df_train)}")
        if 'loss' in df_train.columns:
            print(f"  Final loss: {df_train['loss'].iloc[-1]:.4f}")
            print(f"  Min loss: {df_train['loss'].min():.4f}")
            print(f"  Mean loss: {df_train['loss'].mean():.4f}")
        if 'refinement_steps' in df_train.columns:
            print(f"  Avg refinement steps: {df_train['refinement_steps'].mean():.2f}")
        if 'chunk_confidence' in df_train.columns:
            print(f"  Avg chunk confidence: {df_train['chunk_confidence'].mean():.4f}")

    # Validation stats
    if val_csv.exists():
        df_val = pd.read_csv(val_csv)
        print("\nValidation Metrics:")
        print(f"  Total epochs: {len(df_val)}")
        if 'val_loss' in df_val.columns:
            print(f"  Final val loss: {df_val['val_loss'].iloc[-1]:.4f}")
            print(f"  Best val loss: {df_val['val_loss'].min():.4f}")
        if 'val_refinement_steps' in df_val.columns:
            print(f"  Avg refinement steps: {df_val['val_refinement_steps'].mean():.2f}")
        if 'val_chunk_confidence' in df_val.columns:
            print(f"  Avg confidence: {df_val['val_chunk_confidence'].mean():.4f}")

    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Visualize LLaDA-TRM training metrics')
    parser.add_argument('--log_dir', type=str, required=True,
                       help='Directory containing CSV log files')
    parser.add_argument('--smooth', type=int, default=10,
                       help='Smoothing window size for plots')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for plots (default: same as log_dir)')

    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    output_dir = Path(args.output_dir) if args.output_dir else log_dir

    if not log_dir.exists():
        print(f"Error: Log directory not found: {log_dir}")
        return

    # Paths to CSV files
    train_csv = log_dir / 'training_metrics.csv'
    val_csv = log_dir / 'validation_metrics.csv'

    print("="*70)
    print("LLaDA-TRM Training Metrics Visualization")
    print("="*70)
    print(f"Log directory: {log_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Smoothing window: {args.smooth}")

    # Check if files exist
    if not train_csv.exists():
        print(f"\n⚠ Warning: Training metrics not found: {train_csv}")
    else:
        plot_training_metrics(train_csv, output_dir, smooth_window=args.smooth)

    if not val_csv.exists():
        print(f"\n⚠ Warning: Validation metrics not found: {val_csv}")
    else:
        plot_validation_metrics(val_csv, output_dir)

    # Generate summary
    generate_summary_stats(train_csv, val_csv, output_dir)

    print("\n✓ Visualization complete!")


if __name__ == '__main__':
    main()
