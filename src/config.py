"""Configuration for 2D Darcy Flow Neural Operator training."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class DarcyFlowConfig:
    """Configuration for Darcy Flow experiment."""

    # Data parameters
    train_resolution: int = 85  # Resolution of input/output grid
    grid_boundaries: Optional[list] = None  # [[x_min, x_max], [y_min, y_max]]
    train_samples: int = 1000
    test_samples: int = 100
    batch_size: int = 20
    test_batch_size: int = 100

    # Model parameters (FNO)
    n_modes: tuple = (12, 12)  # Number of Fourier modes
    hidden_channels: int = 64
    n_layers: int = 4

    # Training parameters
    epochs: int = 500
    learning_rate: float = 1e-3
    scheduler_step: int = 100
    scheduler_gamma: float = 0.5
    weight_decay: float = 1e-4

    # Loss parameters
    loss_type: str = "l2"  # 'l2' or 'h1'

    # Device
    device: str = "cuda"  # or "cpu" or "mps" for Mac

    # Logging and checkpoints
    log_interval: int = 10
    save_dir: Path = Path("checkpoints")
    log_dir: Path = Path("logs")

    def __post_init__(self):
        if self.grid_boundaries is None:
            self.grid_boundaries = [[0, 1], [0, 1]]
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)


def get_default_config():
    """Get default configuration for Darcy Flow."""
    return DarcyFlowConfig()
