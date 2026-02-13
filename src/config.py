"""Configuration for 2D Darcy Flow Neural Operator training."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DarcyFlowConfig:
    """Configuration for Darcy Flow experiment."""

    # Data parameters
    test_resolutions: list[int] = field(
        default_factory=lambda: [16, 32]
    )  # test resolutions
    # Selected evaluation resolution (must be one of `test_resolutions`).
    # If None, will default to the first entry in `test_resolutions`.
    eval_resolution: Optional[int] = None
    train_samples: int = 1000
    test_samples: int = 100
    batch_size: int = 20
    test_batch_size: int = 100

    # Model parameters (FNO)
    n_modes: tuple = (8, 8)  # Number of Fourier modes
    in_channels: int = 1
    out_channels: int = 1
    hidden_channels: int = 24
    n_layers: int = 4

    # Training parameters
    epochs: int = 1
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
        # Ensure selected eval resolution is set and valid
        if self.eval_resolution is None:
            # default to first available test resolution
            self.eval_resolution = self.test_resolutions[0]
        elif self.eval_resolution not in self.test_resolutions:
            raise ValueError(
                f"eval_resolution={self.eval_resolution} is not in test_resolutions={self.test_resolutions}"
            )

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)


def get_default_config() -> DarcyFlowConfig:
    """Get default configuration for Darcy Flow."""
    return DarcyFlowConfig()
