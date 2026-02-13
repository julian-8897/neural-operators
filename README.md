# Neural Operators for 2D Darcy Flow

A complete implementation using **neuraloperator library's built-in Trainer class** for solving the 2D Darcy Flow equation with Fourier Neural Operators (FNO).

## Overview

This project implements neural operators to learn the mapping from permeability fields to pressure fields in the 2D Darcy Flow problem. The Darcy Flow equation models fluid flow through porous media and is a fundamental PDE in subsurface flow modeling.

**Problem**: Given a permeability field $a(x)$, find the pressure field $u(x)$ that satisfies:
```
-∇·(a(x)∇u(x)) = f(x)  in Ω = [0,1]²
u(x) = 0  on ∂Ω
```

## Features

- ✅ **Uses neuraloperator's built-in Trainer class** - simplified training pipeline
- ✅ **Built-in loss functions** from neuralop.losses (LpLoss, H1Loss)
- ✅ Fourier Neural Operator (FNO) from neuralop.models
- ✅ Built-in Darcy Flow dataset (no external downloads needed)
- ✅ Automatic checkpointing and logging
- ✅ Configurable hyperparameters
- ✅ Visualization of predictions

## Project Structure

```
neural-operators/
├── src/
│   ├── __init__.py          # Package initialization
│   ├── config.py            # Configuration and hyperparameters
│   ├── data_loader.py       # Data loading and visualization
│   ├── model.py             # FNO model creation and loss selection
│   └── train.py             # Training with neuralop.Trainer
├── main.py                  # Main entry point
├── examples.py              # Example scripts
├── pyproject.toml           # Project dependencies
└── README.md                # This file
```

## Installation

This project uses `uv` for dependency management. Dependencies are already specified in `pyproject.toml`:

```toml
dependencies = [
    "torch>=2.10.0",
    "neuraloperator>=2.0.0",
    "matplotlib>=3.10.8",
]
```

All dependencies should already be installed. If not, run:
```bash
uv sync
```

## Quick Start

### Basic Training

Simply run the main script:

```bash
python main.py
```

This will:
1. Load the built-in Darcy Flow dataset (automatically downloaded and cached)
2. Create an FNO model with default hyperparameters
3. Train for 500 epochs
4. Save checkpoints to `checkpoints/`
5. Save training logs and visualizations to `logs/`

### Custom Configuration

Modify the configuration in [main.py](main.py):

```python
from src.config import get_default_config
from src.train import train_darcy_flow

config = get_default_config()

# Customize hyperparameters
config.epochs = 100
config.learning_rate = 5e-4
config.batch_size = 32
config.n_modes = (16, 16)
config.hidden_channels = 128

# Train using neuralop's Trainer
trainer, data_processor = train_darcy_flow(config)

# Access the trained model and history
model = trainer.model
training_losses = trainer.losses
```

## Configuration Options

Key hyperparameters in [src/config.py](src/config.py):

### Data Parameters
- `train_samples`: Number of training samples (default: 1000)
- `test_samples`: Number of test samples (default: 100)
- `train_resolution`: Grid resolution (default: 85×85)
- `batch_size`: Training batch size (default: 20)

### Model Parameters (FNO)
- `n_modes`: Number of Fourier modes (default: (12, 12))
- `hidden_channels`: Hidden channel dimension (default: 64)
- `n_layers`: Number of FNO layers (default: 4)
- `lifting_channels`: Lifting layer channels (default: 256)
- `projection_channels`: Projection layer channels (default: 256)

### Training Parameters
- `epochs`: Number of training epochs (default: 500)
- `learning_rate`: Initial learning rate (default: 1e-3)
- `scheduler_step`: LR decay step size (default: 100)
- `scheduler_gamma`: LR decay factor (default: 0.5)
- `weight_decay`: Weight decay for regularization (default: 1e-4)

### Loss Parameters
- `loss_type`: Loss function type ('l2' or 'h1')

## Model Architecture

The Fourier Neural Operator (FNO) architecture:

1. **Lifting Layer**: Projects input to higher dimension
2. **Fourier Layers**: Spectral convolutions in Fourier space
   - Linear transformation in frequency domain
   - Skip connections in spatial domain
3. **Projection Layer**: Maps to output dimension

Key advantages:
- Resolution-invariant (can evaluate at different resolutions)
- Fast computation via FFT
- Captures global patterns efficiently

## Dataset

The built-in Darcy Flow dataset from neuraloperator:
- Automatically downloaded on first run
- Cached locally for subsequent runs
- Contains permeability-pressure field pairs
- Supports multiple resolutions

## Output

### Checkpoints
- `checkpoints/best_model.pt`: Best model based on test loss
- `checkpoints/final_model.pt`: Final model after all epochs

### Logs
- `logs/training_history.json`: Loss curves and learning rates
- `logs/prediction_sample.png`: Visualization of a test prediction

## Example Output

```
================================================================================
Training Fourier Neural Operator on 2D Darcy Flow
================================================================================
Using device: MPS (Apple Silicon)

Loading Darcy Flow dataset...
Training samples: 1000
Test samples: 100
Resolution: 85x85

Creating FNO model...
Model parameters: 2,459,009
Fourier modes: (12, 12)
Hidden channels: 64
Number of layers: 4

Loss function: L2
Learning rate: 0.001
Batch size: 20
Epochs: 500

================================================================================
Starting training...
================================================================================
Epoch    1/500 | Train Loss: 0.156432 | Test Loss: 0.142156 | LR: 1.00e-03 | Time: 2.45s
  → Saved best model (test loss: 0.142156)
Epoch   10/500 | Train Loss: 0.087234 | Test Loss: 0.082341 | LR: 1.00e-03 | Time: 2.31s
...
```Using neuralop's Built-in Components

The code uses neuraloperator's built-in classes for clean, maintainable code:

```python
from neuralop.training import Trainer
from neuralop.losses import LpLoss, H1Loss
from neuralop.models import FNO
from src.config import get_default_config
from src.data_loader import get_darcy_flow_dataloaders

config = get_default_config()
train_loader, test_loader, data_processor = get_darcy_flow_dataloaders(config)

# Create model
model = FNO(
    n_modes=(12, 12),
    hidden_channels=64,
    in_channels=3,
    out_channels=1,
    n_layers=4
)

# Create trainer
trainer = Trainer(
    model=model,
    n_epochs=config.epochs,
    device='cuda',
    verbose=True
)

# Train
trainer.train(
    train_loader=train_loader,
    test_loaders={85: test_loader},
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
    training_loss=LpLoss(d=2, p=2),
    eval_losses={'l2': LpLoss(d=2, p=2)}
)processor = get_darcy_flow_dataloaders(config)
model = create_fno_model(config)
criterion = get_loss_function(config)

# Your custom training loop here
```

### Visualization
Neuraloperator Documentation**: [Docs](https://neuraloperator.github.io/neuraloperator/)
- **Darcy Flow**: Classical problem in subsurface flow modeling

## Key Neuraloperator Components Used

- `neuralop.models.FNO`: Fourier Neural Operator model
- `neuralop.training.Trainer`: Built-in training loop with logging
- `neuralop.losses.LpLoss`: Relative Lp norm loss
- `neuralop.losses.H1Loss`: H1 Sobolev norm loss
- `neuralop.datasets.load_darcy_flow_small`: Built-in Darcy Flow dataset
```python
from src.data_loader import visualize_sample

# Visualize a sample
visualize_sample(
    input_tensor,
    output_tensor,
    prediction=model_output,
    data_processor=data_processor,
    save_path="output.png"
)
```

## Performance Tips

1. **GPU Usage**: Set `config.device = "cuda"` if you have NVIDIA GPU
2. **Apple Silicon**: Use `config.device = "mps"` for M1/M2/M3 Macs
3. **Batch Size**: Increase if you have more memory
4. **Fourier Modes**: More modes = more capacity but slower training
5. **Resolution**: Start with lower resolution for faster iteration

## References

- **FNO Paper**: [Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895)
- **Neuraloperator Library**: [GitHub](https://github.com/neuraloperator/neuraloperator)
- **Darcy Flow**: Classical problem in subsurface flow modeling

## License

See [LICENSE](LICENSE) file for details.
