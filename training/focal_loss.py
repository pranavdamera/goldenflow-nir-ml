"""Focal loss implementation for imbalanced adulteration classes."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    """Compute focal loss for multi-class classification.

    Focal loss down-weights easy majority examples and focuses training on
    hard minority classes. This is useful for adulteration datasets where
    `pure` honey often dominates label distribution.

    Formula:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean") -> None:
        """Initialize focal loss hyperparameters."""
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Return focal loss over class logits and integer class targets."""
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal
