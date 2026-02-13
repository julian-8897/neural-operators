"""Training script for 2D Darcy Flow with Neural Operators."""

import torch
from neuralop.training import Trainer
from neuralop.utils import count_model_params

from src.config import get_default_config
from src.data_loader import get_darcy_flow_dataloaders, visualize_sample
from src.model import create_fno_model, get_loss_function


def train_darcy_flow(config=None):
    """
    Main training function for Darcy Flow using neuralop's Trainer.

    Args:
        config: Configuration object (uses default if None)

    Returns:
        trainer: Trained Trainer object with model and history
        data_processor: Data processor for visualization
    """
    if config is None:
        config = get_default_config()

    print("=" * 80)
    print("Training Fourier Neural Operator on 2D Darcy Flow")
    print("=" * 80)

    # Set device
    if config.device == "cuda" and torch.cuda.is_available():
        device = "cuda"
        print(f"Using device: CUDA ({torch.cuda.get_device_name(0)})")
    elif config.device == "mps" and torch.backends.mps.is_available():
        device = "mps"
        print("Using device: MPS (Apple Silicon)")
    else:
        device = "cpu"
        print("Using device: CPU")

    # Load data
    print("\nLoading Darcy Flow dataset...")
    train_loader, test_loaders, data_processor = get_darcy_flow_dataloaders(config)
    print(f"Training samples: {config.train_samples}")
    print(f"Test samples: {config.test_samples}")
    print("Available test resolutions (config):", config.test_resolutions)

    # Create model
    print("\nCreating FNO model...")
    model = create_fno_model(config)
    model = model.to(device)

    # Count parameters
    n_params = count_model_params(model)
    print(f"Model parameters: {n_params:,}")
    print(f"Fourier modes: {config.n_modes}")
    print(f"Hidden channels: {config.hidden_channels}")
    print(f"Number of layers: {config.n_layers}")

    # Loss function
    train_loss = get_loss_function(config)
    eval_losses = {"l2": train_loss}

    print(f"\nLoss function: {config.loss_type.upper()}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Batch size: {config.batch_size}")
    print(f"Epochs: {config.epochs}")

    # Create Trainer using neuralop's built-in Trainer class
    print("\n" + "=" * 80)
    print("Initializing Trainer...")
    print("=" * 80)

    trainer = Trainer(
        model=model,
        n_epochs=config.epochs,
        device=device,
        verbose=True,
    )

    # Configure optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.scheduler_step, gamma=config.scheduler_gamma
    )

    # Train the model
    print("\nStarting training...")
    print("=" * 80)

    trainer.train(
        train_loader=train_loader,
        test_loaders=test_loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        regularizer=None,
        training_loss=train_loss,
        eval_losses=eval_losses,
    )

    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)

    # Save the trained model
    print("\nSaving models...")
    best_model_path = config.save_dir / "best_model.pt"
    torch.save(
        {
            "model_state_dict": trainer.model.state_dict(),
            "config": config,
        },
        best_model_path,
    )
    print(f"Model saved to: {best_model_path}")

    # Visualize a test sample
    print("\nGenerating visualization...")
    model.eval()
    with torch.no_grad():
        # Get a test sample from the dataset and preprocess it
        test_sample = test_loaders[config.eval_resolution].dataset[0]
        test_sample = data_processor.preprocess(test_sample, batched=False)

        # Add batch dimension and move to device
        x_test = test_sample["x"].unsqueeze(0).to(device)
        y_test = test_sample["y"].unsqueeze(0).to(device)

        # Model prediction
        pred = model(x_test)

        vis_path = config.log_dir / "prediction_sample.png"
        visualize_sample(
            x_test[0],
            y_test[0],
            pred[0],
            data_processor=data_processor,
            save_path=vis_path,
        )

    return trainer, data_processor


if __name__ == "__main__":
    # Train with default configuration
    trainer, data_processor = train_darcy_flow()
