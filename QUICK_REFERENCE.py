"""
Quick Reference: Using Neuraloperator's Built-in Components
=============================================================

This codebase now uses neuraloperator's built-in Trainer class and loss functions
for a cleaner, more maintainable implementation.

Key Changes
-----------

1. **Trainer Class** (from neuralop.training)
   - Replaces custom training loop
   - Handles training/evaluation automatically
   - Built-in logging and progress tracking

2. **Loss Functions** (from neuralop.losses)
   - LpLoss: Relative Lp norm (p=2 for L2)
   - H1Loss: H1 Sobolev norm
   - No need for custom implementations

3. **FNO Model** (from neuralop.models)
   - Already available in neuraloperator
   - Clean API with sensible defaults

Usage Examples
--------------

Basic Training:
    ```python
    from src.train import train_darcy_flow
    from src.config import get_default_config

    config = get_default_config()
    trainer, data_processor = train_darcy_flow(config)

    # Access trained model
    model = trainer.model

    # Access training history
    losses = trainer.losses
    ```

Custom Configuration:
    ```python
    config = get_default_config()
    config.epochs = 100
    config.n_modes = (16, 16)
    config.hidden_channels = 128
    config.loss_type = "h1"  # or "l2"

    trainer, data_processor = train_darcy_flow(config)
    ```

Direct Trainer Usage:
    ```python
    from neuralop.training import Trainer
    from neuralop.losses import LpLoss
    from neuralop.models import FNO

    model = FNO(n_modes=(12, 12), hidden_channels=64, ...)
    trainer = Trainer(model=model, n_epochs=500, device='cuda')

    trainer.train(
        train_loader=train_loader,
        test_loaders={85: test_loader},
        optimizer=torch.optim.Adam(model.parameters()),
        training_loss=LpLoss(d=2, p=2),
        eval_losses={'l2': LpLoss(d=2, p=2)}
    )
    ```

Neuraloperator Components Used
-------------------------------

- neuralop.models.FNO
  Fourier Neural Operator with spectral convolutions

- neuralop.training.Trainer
  Unified training interface with logging

- neuralop.losses.LpLoss
  Relative Lp norm loss (default p=2)

- neuralop.losses.H1Loss
  H1 Sobolev norm (L2 + gradient penalty)

- neuralop.datasets.load_darcy_flow_small
  Built-in 2D Darcy Flow dataset

Benefits
--------

✅ Less code to maintain (removed ~100 lines of custom training logic)
✅ Better tested implementations from neuraloperator team
✅ Consistent API across different neural operator models
✅ Built-in logging and progress tracking
✅ Easy to extend with new features from neuraloperator updates

File Structure
--------------

src/
├── config.py         - Configuration dataclass
├── data_loader.py    - Dataset loading + visualization
├── model.py          - Model creation + loss selection (simplified)
└── train.py          - Training with Trainer class (refactored)

main.py               - Entry point
examples.py           - Usage examples

References
----------

- Neuraloperator Docs: https://neuraloperator.github.io/neuraloperator/
- GitHub: https://github.com/neuraloperator/neuraloperator
- FNO Paper: https://arxiv.org/abs/2010.08895
"""

if __name__ == "__main__":
    print(__doc__)
