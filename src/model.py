"""Model definitions for neural operators."""

from neuralop.losses import H1Loss, LpLoss
from neuralop.models import FNO


def create_fno_model(config):
    """
    Create a Fourier Neural Operator (FNO) model for Darcy Flow.

    Args:
        config: Configuration object with model parameters
        in_channels (int): Number of input channels (default: 3 for permeability + 2D coordinates)
        out_channels (int): Number of output channels (default: 1 for pressure field)

    Returns:
        FNO model
    """
    model = FNO(
        n_modes=config.n_modes,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        hidden_channels=config.hidden_channels,
        n_layers=config.n_layers,
    )

    return model


def get_loss_function(config):
    """
    Get loss function based on configuration using neuralop's built-in losses.

    Args:
        config: Configuration object
s
    Returns:
        Loss function from neuralop.losses
    """
    if config.loss_type.lower() == "l2":
        return LpLoss(d=2, p=2)
    elif config.loss_type.lower() == "h1":
        return H1Loss(d=2)
    else:
        raise ValueError(f"Unknown loss type: {config.loss_type}")
