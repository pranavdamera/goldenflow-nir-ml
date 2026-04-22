"""HoneyNIRNet architecture for NIR adulteration inference.

Design rationale:
- 1D convolutions operate over the spectral dimension directly; no 2D reshaping needed.
- Squeeze-Excitation attention captures inter-channel (frequency-band) correlations,
  which is the dominant structure in NIR spectra. Multi-head self-attention would add
  positional relationships that are not meaningful across a smoothed spectral axis and
  would push the ONNX export well past the 2MB deployment target.
- Three Conv blocks (32→64→128 channels) give sufficient representational depth while
  keeping parameter count low enough for the size constraint.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn


class SqueezeExcitation(nn.Module):
    """Channel attention block that reweights informative feature channels.

    The module performs global average pooling over the spectral dimension,
    projects channel descriptors to a bottleneck (`C/16`), and restores channel
    dimensionality with a sigmoid gate to scale each channel response.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        """Initialize a Squeeze-Excitation block.

        Args:
            channels: Number of convolutional channels to recalibrate.
            reduction: Bottleneck reduction factor for channel excitation.
        """
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        """Apply channel recalibration to a feature map tensor."""
        batch_size, channels, _ = x.shape
        pooled = self.pool(x).view(batch_size, channels)
        scale = self.fc2(self.relu(self.fc1(pooled)))
        scale = self.sigmoid(scale).view(batch_size, channels, 1)
        return x * scale


class ConvSEBlock(nn.Module):
    """Convolutional feature extraction block with SE channel attention.

    Each block applies 1D convolution, batch normalization, and ReLU for
    stable representation learning, followed by Squeeze-Excitation to focus
    on adulteration-relevant spectral channels.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5) -> None:
        """Build a Conv-BN-ReLU-SE stack for 1D spectral signals."""
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.se = SqueezeExcitation(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        """Return block output after convolutional and attention operations."""
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.se(x)
        return x


class HoneyNIRNet(nn.Module):
    """1D-CNN with Squeeze-Excitation attention for honey adulteration analysis.

    Input shape is `(batch, 1, 256)` where 256 points map to the 900-1700nm
    measurement span. The network emits:
    - 7-class classification logits for adulteration profile detection
    - 1-unit regression output (sigmoid) for adulteration percentage in [0, 1]
    """

    def __init__(self) -> None:
        """Construct the full HoneyNIRNet inference architecture."""
        super().__init__()
        self.block1 = ConvSEBlock(1, 32)
        self.block2 = ConvSEBlock(32, 64)
        self.block3 = ConvSEBlock(64, 128)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(128, 7)
        self.regressor = nn.Linear(128, 1)
        self.regression_activation = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Run a forward pass and return classification + regression outputs."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_pool(x).squeeze(-1)
        class_logits = self.classifier(x)
        adulteration_pct = self.regression_activation(self.regressor(x))
        return class_logits, adulteration_pct
