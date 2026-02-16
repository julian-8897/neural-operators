"""
Evaluation script for trained neural operator models.

Usage:
    uv run python evaluate.py --checkpoint checkpoints/best_model.pt
    uv run python evaluate.py --checkpoint checkpoints/best_model.pt --resolution 32
    uv run python evaluate.py --checkpoint checkpoints/best_model.pt --save-viz
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data_loader import get_darcy_flow_dataloaders
from src.model import create_fno_model, get_loss_function


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    Compute comprehensive error metrics.

    Args:
        pred: Model predictions [batch, ...]
        target: Ground truth [batch, ...]

    Returns:
        Dictionary of metrics
    """
    # Flatten spatial dimensions
    batch_size = pred.size(0)
    pred_flat = pred.reshape(batch_size, -1)
    target_flat = target.reshape(batch_size, -1)

    # Relative L2 error
    diff_norms = torch.norm(pred_flat - target_flat, p=2, dim=1)
    target_norms = torch.norm(target_flat, p=2, dim=1)
    rel_l2 = (diff_norms / target_norms).mean().item()

    # Maximum pointwise error
    max_error = torch.abs(pred - target).max().item()

    # Mean absolute error
    mae = torch.abs(pred - target).mean().item()

    # Root mean squared error
    rmse = torch.sqrt(((pred - target) ** 2).mean()).item()

    # R² score
    ss_res = torch.sum((target_flat - pred_flat) ** 2)
    ss_tot = torch.sum((target_flat - target_flat.mean(dim=1, keepdim=True)) ** 2)
    r2 = (1 - ss_res / ss_tot).item()

    return {
        "relative_l2": rel_l2,
        "max_error": max_error,
        "mae": mae,
        "rmse": rmse,
        "r2_score": r2,
    }


def evaluate_model(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    data_processor,
    criterion,
    device: str,
) -> Dict:
    """
    Evaluate model on test set.

    Args:
        model: Neural operator model
        test_loader: Test data loader
        data_processor: Data processor for preprocessing
        criterion: Loss function
        device: Device to evaluate on

    Returns:
        Dictionary with evaluation results
    """
    model.eval()

    all_losses = []
    all_metrics = []
    predictions = []
    targets = []
    inputs = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Preprocess data
            if data_processor is not None:
                batch = data_processor.preprocess(batch, batched=True)

            x = batch["x"].to(device)
            y = batch["y"].to(device)

            # Forward pass
            pred = model(x)

            # Compute loss
            loss = criterion(pred, y)
            all_losses.append(loss.item())

            # Compute metrics
            metrics = compute_metrics(pred, y)
            all_metrics.append(metrics)

            # Store samples for visualization (first batch only)
            if batch_idx == 0:
                predictions = pred.cpu()
                targets = y.cpu()
                inputs = x.cpu()

    # Average metrics
    avg_metrics = {
        key: np.mean([m[key] for m in all_metrics]) for key in all_metrics[0].keys()
    }
    avg_metrics["loss"] = np.mean(all_losses)

    # Add std for key metrics
    avg_metrics["relative_l2_std"] = np.std([m["relative_l2"] for m in all_metrics])
    avg_metrics["mae_std"] = np.std([m["mae"] for m in all_metrics])

    return {
        "metrics": avg_metrics,
        "per_sample_metrics": all_metrics,
        "sample_predictions": predictions,
        "sample_targets": targets,
        "sample_inputs": inputs,
    }


def visualize_predictions(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    predictions: torch.Tensor,
    save_dir: Path,
    n_samples: int = 3,
):
    """
    Create visualization of model predictions.

    Args:
        inputs: Input fields
        targets: Ground truth outputs
        predictions: Model predictions
        save_dir: Directory to save visualizations
        n_samples: Number of samples to visualize
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    n_samples = min(n_samples, inputs.size(0))

    for idx in range(n_samples):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        # Extract sample
        input_np = inputs[idx, 0].numpy()
        target_np = targets[idx, 0].numpy()
        pred_np = predictions[idx, 0].numpy()
        error_np = np.abs(target_np - pred_np)

        # Plot input
        im0 = axes[0].imshow(input_np, cmap="viridis")
        axes[0].set_title("Input: Permeability", fontsize=12, fontweight="bold")
        axes[0].axis("off")
        plt.colorbar(im0, ax=axes[0], fraction=0.046)

        # Plot ground truth
        im1 = axes[1].imshow(target_np, cmap="RdBu_r")
        axes[1].set_title("Ground Truth: Pressure", fontsize=12, fontweight="bold")
        axes[1].axis("off")
        plt.colorbar(im1, ax=axes[1], fraction=0.046)

        # Plot prediction
        im2 = axes[2].imshow(
            pred_np, cmap="RdBu_r", vmin=target_np.min(), vmax=target_np.max()
        )
        axes[2].set_title("Prediction", fontsize=12, fontweight="bold")
        axes[2].axis("off")
        plt.colorbar(im2, ax=axes[2], fraction=0.046)

        # Plot error
        im3 = axes[3].imshow(error_np, cmap="hot")
        axes[3].set_title(
            f"Absolute Error\nMax: {error_np.max():.4f}",
            fontsize=11,
            fontweight="bold",
        )
        axes[3].axis("off")
        plt.colorbar(im3, ax=axes[3], fraction=0.046)

        plt.suptitle(
            f"Sample {idx + 1} - Model Evaluation", fontsize=14, fontweight="bold"
        )
        plt.tight_layout()

        save_path = save_dir / f"prediction_sample_{idx + 1}.png"
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()

        print(f"  Saved visualization: {save_path}")


def save_results(results: Dict, save_path: Path):
    """Save evaluation results to JSON."""
    # Convert to JSON-serializable format
    json_results = {
        "metrics": results["metrics"],
        "per_sample_metrics": results["per_sample_metrics"],
    }

    with open(save_path, "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"  Saved results: {save_path}")


def print_results(results: Dict):
    """Print evaluation results in a formatted table."""
    metrics = results["metrics"]

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"{'Metric':<25} {'Value':>15} {'Std':>15}")
    print("-" * 80)
    print(f"{'Loss':<25} {metrics['loss']:>15.6f} {'':<15}")
    print(
        f"{'Relative L2 Error':<25} {metrics['relative_l2']:>15.6f} {metrics['relative_l2_std']:>15.6f}"
    )
    print(f"{'Max Pointwise Error':<25} {metrics['max_error']:>15.6f} {'':<15}")
    print(
        f"{'Mean Absolute Error':<25} {metrics['mae']:>15.6f} {metrics['mae_std']:>15.6f}"
    )
    print(f"{'RMSE':<25} {metrics['rmse']:>15.6f} {'':<15}")
    print(f"{'R² Score':<25} {metrics['r2_score']:>15.6f} {'':<15}")
    print("=" * 80)

    # Summary
    print(f"\nNumber of test samples: {len(results['per_sample_metrics'])}")
    print(
        f"Average relative L2 error: {metrics['relative_l2']:.4f} ± {metrics['relative_l2_std']:.4f}"
    )


def load_checkpoint(checkpoint_path: Path, device: str):
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        model, config
    """
    print(f"\nLoading checkpoint from: {checkpoint_path}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint with weights_only=False for PyTorch 2.6+
    # This is safe for checkpoints from trusted sources
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract config
    if "config" in checkpoint:
        config = checkpoint["config"]
    else:
        print("Warning: Config not found in checkpoint, using default")
        from src.config import get_default_config

        config = get_default_config()

    # Create model
    model = create_fno_model(config)

    # Load state dict
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Try loading directly
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    print("✓ Model loaded successfully")
    print(f"  Device: {device}")

    # Print checkpoint info
    if "epoch" in checkpoint:
        print(f"  Trained for {checkpoint['epoch']} epochs")
    if "test_loss" in checkpoint:
        print(f"  Checkpoint test loss: {checkpoint['test_loss']:.6f}")

    return model, config


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained neural operator model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Test resolution (default: use config from checkpoint)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for evaluation (default: 100)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use (default: auto)",
    )
    parser.add_argument(
        "--save-viz",
        action="store_true",
        help="Save visualizations of predictions",
    )
    parser.add_argument(
        "--n-viz",
        type=int,
        default=3,
        help="Number of samples to visualize (default: 3)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Directory to save results (default: evaluation_results)",
    )

    args = parser.parse_args()

    # Setup device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print("=" * 80)
    print("NEURAL OPERATOR MODEL EVALUATION")
    print("=" * 80)

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    model, config = load_checkpoint(checkpoint_path, device)

    # Override resolution if specified
    if args.resolution is not None:
        config.test_resolutions = [args.resolution]
        config.eval_resolution = args.resolution
        print(f"\nUsing custom resolution: {args.resolution}x{args.resolution}")

    # Override batch size
    config.test_batch_size = args.batch_size

    # Load test data
    print("\nLoading test data...")
    print(f"  Resolution: {config.eval_resolution}x{config.eval_resolution}")
    print(f"  Test samples: {config.test_samples}")

    train_loader, test_loaders, data_processor = get_darcy_flow_dataloaders(config)
    test_loader = test_loaders[config.eval_resolution]

    print("✓ Data loaded successfully")

    # Get loss function
    criterion = get_loss_function(config)

    # Run evaluation
    print("\nRunning evaluation...")
    results = evaluate_model(model, test_loader, data_processor, criterion, device)

    # Print results
    print_results(results)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "evaluation_metrics.json"
    save_results(results, results_path)

    # Save visualizations if requested
    if args.save_viz:
        print("\nGenerating visualizations...")
        viz_dir = output_dir / "visualizations"
        visualize_predictions(
            results["sample_inputs"],
            results["sample_targets"],
            results["sample_predictions"],
            viz_dir,
            n_samples=args.n_viz,
        )

    print(f"\n{'=' * 80}")
    print(f"Evaluation completed! Results saved to: {output_dir}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
