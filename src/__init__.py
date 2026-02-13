"""Neural Operators for 2D Darcy Flow."""

from src.config import DarcyFlowConfig, get_default_config
from src.data_loader import get_darcy_flow_dataloaders, visualize_sample
from src.model import H1Loss, LpLoss, create_fno_model, get_loss_function
from src.train import train_darcy_flow

__all__ = [
    "DarcyFlowConfig",
    "get_default_config",
    "get_darcy_flow_dataloaders",
    "visualize_sample",
    "create_fno_model",
    "get_loss_function",
    "LpLoss",
    "H1Loss",
    "train_darcy_flow",
]
