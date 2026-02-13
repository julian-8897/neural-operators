"""Training script for 2D Darcy Flow with Neural Operators."""

import json
import time

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from src.config import get_default_config
from src.data_loader import get_darcy_flow_dataloaders, visualize_sample
from src.model import create_fno_model, get_loss_function


def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """
    Train for one epoch.

    Args:
        model: Neural operator model
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        epoch: Current epoch number

    Returns:
        Average training loss
    """
    model.train()
    train_loss = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        # Get data
        x = batch["x"].to(device)  # Input: permeability field
        y = batch["y"].to(device)  # Output: pressure field

        # Forward pass
        optimizer.zero_grad()
        out = model(x)

        # Compute loss
        loss = criterion(out, y)

        # Backward pass
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        n_batches += 1

    avg_loss = train_loss / n_batches
    return avg_loss


def evaluate(model, test_loader, criterion, device):
    """
    Evaluate the model.

    Args:
        model: Neural operator model
        test_loader: Test data loader
        criterion: Loss function
        device: Device to evaluate on

    Returns:
        Average test loss
    """
    model.eval()
    test_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in test_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            out = model(x)
            loss = criterion(out, y)

            test_loss += loss.item()
            n_batches += 1

    avg_loss = test_loss / n_batches
    return avg_loss


def train_darcy_flow(config=None):
    """
    Main training function for Darcy Flow.

    Args:
        config: Configuration object (uses default if None)
    """
    if config is None:
        config = get_default_config()

    print("=" * 80)
    print("Training Fourier Neural Operator on 2D Darcy Flow")
    print("=" * 80)

    # Set device
    if config.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: CUDA ({torch.cuda.get_device_name(0)})")
    elif config.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    # Load data
    print("\nLoading Darcy Flow dataset...")
    train_loader, test_loader, data_processor = get_darcy_flow_dataloaders(config)
    print(f"Training samples: {config.train_samples}")
    print(f"Test samples: {config.test_samples}")
    print(f"Resolution: {config.train_resolution}x{config.train_resolution}")

    # Create model
    print("\nCreating FNO model...")
    model = create_fno_model(config)
    model = model.to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    print(f"Fourier modes: {config.n_modes}")
    print(f"Hidden channels: {config.hidden_channels}")
    print(f"Number of layers: {config.n_layers}")

    # Loss function and optimizer
    criterion = get_loss_function(config)
    optimizer = Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = StepLR(
        optimizer, step_size=config.scheduler_step, gamma=config.scheduler_gamma
    )

    print(f"\nLoss function: {config.loss_type.upper()}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Batch size: {config.batch_size}")
    print(f"Epochs: {config.epochs}")

    # Training history
    history = {"train_loss": [], "test_loss": [], "learning_rate": []}

    # Training loop
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)

    best_test_loss = float("inf")
    start_time = time.time()

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )

        # Evaluate
        test_loss = evaluate(model, test_loader, criterion, device)

        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Record history
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["learning_rate"].append(current_lr)

        epoch_time = time.time() - epoch_start

        # Print progress
        if epoch % config.log_interval == 0 or epoch == 1:
            print(
                f"Epoch {epoch:4d}/{config.epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Test Loss: {test_loss:.6f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {epoch_time:.2f}s"
            )

        # Save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            checkpoint_path = config.save_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "test_loss": test_loss,
                    "train_loss": train_loss,
                    "config": config,
                },
                checkpoint_path,
            )
            print(f"  → Saved best model (test loss: {test_loss:.6f})")

    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"Training completed in {total_time / 60:.2f} minutes")
    print(f"Best test loss: {best_test_loss:.6f}")
    print("=" * 80)

    # Save final model
    final_checkpoint_path = config.save_dir / "final_model.pt"
    final_train_loss = history["train_loss"][-1] if history["train_loss"] else 0.0
    final_test_loss = history["test_loss"][-1] if history["test_loss"] else 0.0
    torch.save(
        {
            "epoch": config.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "test_loss": final_test_loss,
            "train_loss": final_train_loss,
            "config": config,
        },
        final_checkpoint_path,
    )
    print(f"\nFinal model saved to: {final_checkpoint_path}")

    # Save training history
    history_path = config.log_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {history_path}")

    # Visualize a test sample
    print("\nGenerating visualization...")
    model.eval()
    with torch.no_grad():
        test_batch = next(iter(test_loader))
        x_test = test_batch["x"][:1].to(device)
        y_test = test_batch["y"][:1].to(device)
        pred = model(x_test)

        vis_path = config.log_dir / "prediction_sample.png"
        visualize_sample(
            x_test[0],
            y_test[0],
            pred[0],
            data_processor=data_processor,
            save_path=vis_path,
        )

    return model, history, data_processor


if __name__ == "__main__":
    # Train with default configuration
    model, history, data_processor = train_darcy_flow()
