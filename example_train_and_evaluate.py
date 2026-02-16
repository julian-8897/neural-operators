"""
Example: Complete workflow - Train and Evaluate

This example demonstrates:
1. Training a neural operator model
2. Saving checkpoints
3. Loading checkpoints for evaluation
4. Generating comprehensive evaluation metrics
"""

from src.config import get_default_config
from src.train import train_darcy_flow


def main():
    """Complete training and evaluation workflow."""

    # Step 1: Configure and train model
    print("=" * 80)
    print("STEP 1: Training Model")
    print("=" * 80)

    config = get_default_config()

    # Customize for quick training (for demonstration)
    config.epochs = 20  # Use 500+ for production
    config.train_samples = 100  # Use 1000+ for production
    config.test_samples = 50  # Use 100+ for production
    config.batch_size = 10

    # Train the model
    trainer, data_processor = train_darcy_flow(config)

    # Step 2: Show how to evaluate
    print("\n" + "=" * 80)
    print("STEP 2: Model Evaluation")
    print("=" * 80)

    checkpoint_path = config.save_dir / "best_model.pt"

    print(f"\nModel saved at: {checkpoint_path}")
    print("\nTo evaluate this model, run:")
    print(f"\n  uv run python evaluate.py --checkpoint {checkpoint_path}")
    print("\nWith visualizations:")
    print(
        f"  uv run python evaluate.py --checkpoint {checkpoint_path} --save-viz --n-viz 5"
    )
    print("\nAt different resolutions:")
    print(f"  uv run python evaluate.py --checkpoint {checkpoint_path} --resolution 32")
    print(f"  uv run python evaluate.py --checkpoint {checkpoint_path} --resolution 16")

    print("\n" + "=" * 80)
    print("Workflow Complete!")
    print("=" * 80)

    # Show what files were created
    print("\nGenerated files:")
    print(f"  ✓ Checkpoint: {checkpoint_path}")
    print(f"  ✓ Visualization: {config.log_dir / 'prediction_sample.png'}")

    print("\nNext steps:")
    print("  1. Run evaluation script to get detailed metrics")
    print("  2. Test at multiple resolutions")
    print("  3. Compare different hyperparameters")
    print("  4. Extend with new PDEs (see TEMPLATE_NEW_PDE.py)")


if __name__ == "__main__":
    main()
