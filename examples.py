"""Example usage of the 2D Darcy Flow neural operator training."""

from src.config import get_default_config
from src.train import train_darcy_flow


def quick_example():
    """Quick training example with reduced epochs for testing."""
    print("Running quick example (10 epochs)...\n")

    config = get_default_config()
    config.epochs = 10
    config.train_samples = 100
    config.test_samples = 20
    config.log_interval = 2

    trainer, data_processor = train_darcy_flow(config)

    print("\n✅ Quick example completed!")
    print(f"Model trained for {trainer.n_epochs} epochs")
    print("Training history available in: trainer.losses")


def custom_config_example():
    """Example with custom configuration."""
    print("Running with custom configuration...\n")

    config = get_default_config()

    # Customize model architecture
    config.n_modes = (16, 16)  # More Fourier modes
    config.hidden_channels = 128  # Larger hidden dimension
    config.n_layers = 6  # Deeper network

    # Training settings
    config.epochs = 100
    config.batch_size = 32
    config.learning_rate = 5e-4

    # Loss function
    config.loss_type = "h1"  # Use H1 loss instead of L2

    trainer, data_processor = train_darcy_flow(config)

    print("\n✅ Custom training completed!")
    print("Best model saved to: checkpoints/best_model.pt")


if __name__ == "__main__":
    # Run quick example
    quick_example()

    # Uncomment to run custom config example:
    # custom_config_example()
