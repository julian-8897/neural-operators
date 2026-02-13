"""Model definitions for neural operators."""

import torch
import torch.nn as nn
from neuralop.models import FNO


def create_fno_model(config, in_channels=3, out_channels=1):
    """
    Create a Fourier Neural Operator (FNO) model for Darcy Flow.

    Args:
        config: Configuration object with model parameters
        in_channels: Number of input channels (default: 3 for permeability + 2D coordinates)
        out_channels: Number of output channels (default: 1 for pressure field)

    Returns:
        FNO model
    """
    model = FNO(
        n_modes=config.n_modes,
        hidden_channels=config.hidden_channels,
        in_channels=in_channels,
        out_channels=out_channels,
        n_layers=config.n_layers,
    )

    return model


class LpLoss(nn.Module):
    """
    Lp loss for neural operator training.

    Args:
        d: Dimension of the domain
        p: Order of the norm (default: 2 for L2)
        reduction: Reduction method ('mean' or 'sum')
    """

    def __init__(self, d=2, p=2, reduction="mean"):
        super().__init__()
        self.d = d
        self.p = p
        self.reduction = reduction

    def forward(self, pred, target):
        """
        Compute Lp loss.

        Args:
            pred: Predictions [batch, ...]
            target: Ground truth [batch, ...]
        """
        # Flatten spatial dimensions
        batch_size = pred.size(0)
        pred_flat = pred.reshape(batch_size, -1)
        target_flat = target.reshape(batch_size, -1)

        # Compute relative Lp norm
        diff_norms = torch.norm(pred_flat - target_flat, p=self.p, dim=1)
        target_norms = torch.norm(target_flat, p=self.p, dim=1)

        loss = diff_norms / target_norms

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class H1Loss(nn.Module):
    """
    H1 loss (combines L2 loss and gradient loss).
    """

    def __init__(self, d=2, reduction="mean"):
        super().__init__()
        self.d = d
        self.reduction = reduction

    def forward(self, pred, target):
        """
        Compute H1 loss.

        Args:
            pred: Predictions [batch, channels, height, width]
            target: Ground truth [batch, channels, height, width]
        """
        # L2 part
        batch_size = pred.size(0)
        pred_flat = pred.reshape(batch_size, -1)
        target_flat = target.reshape(batch_size, -1)

        diff_l2 = torch.norm(pred_flat - target_flat, p=2, dim=1)
        target_l2 = torch.norm(target_flat, p=2, dim=1)
        l2_loss = diff_l2 / target_l2

        # Gradient part (approximate using finite differences)
        if pred.dim() == 4:  # [batch, channels, height, width]
            # Compute gradients
            pred_dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
            pred_dy = pred[:, :, :, 1:] - pred[:, :, :, :-1]
            target_dx = target[:, :, 1:, :] - target[:, :, :-1, :]
            target_dy = target[:, :, :, 1:] - target[:, :, :, :-1]

            # Flatten and compute norms
            pred_grad = torch.cat(
                [pred_dx.reshape(batch_size, -1), pred_dy.reshape(batch_size, -1)],
                dim=1,
            )
            target_grad = torch.cat(
                [target_dx.reshape(batch_size, -1), target_dy.reshape(batch_size, -1)],
                dim=1,
            )

            diff_grad = torch.norm(pred_grad - target_grad, p=2, dim=1)
            target_grad_norm = torch.norm(target_grad, p=2, dim=1)
            grad_loss = diff_grad / (target_grad_norm + 1e-8)

            loss = l2_loss + grad_loss
        else:
            loss = l2_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def get_loss_function(config):
    """
    Get loss function based on configuration.

    Args:
        config: Configuration object

    Returns:
        Loss function
    """
    if config.loss_type.lower() == "l2":
        return LpLoss(d=2, p=2)
    elif config.loss_type.lower() == "h1":
        return H1Loss(d=2)
    else:
        raise ValueError(f"Unknown loss type: {config.loss_type}")
