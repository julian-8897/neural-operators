"""Main entry point for 2D Darcy Flow Neural Operator training."""

from src.config import get_default_config
from src.train import train_darcy_flow


def main():
    """Main function to train neural operator on 2D Darcy Flow."""
    print("🚀 Neural Operator for 2D Darcy Flow\n")

    # Get default configuration
    config = get_default_config()

    # You can customize configuration here
    # config.epochs = 100
    # config.learning_rate = 5e-4
    # config.batch_size = 32

    # Train the model using neuralop's Trainer
    trainer, data_processor = train_darcy_flow(config)

    print(
        "\n✅ Training completed successfully!"
        "\nAccess model: trainer.model"
        "\nTraining history: trainer.losses"
    )


if __name__ == "__main__":
    main()
