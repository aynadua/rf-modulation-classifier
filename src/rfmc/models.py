"""The neural classifier: a small 1D CNN over raw IQ.

Design constraints, in priority order:

* It must run on a CPU in minutes, because that is the hardware this repo is
  reproducible on.
* It must be small enough to export and quantise for an SDR front end, so
  roughly 100k parameters rather than millions.
* It must be phase-agnostic. Global average pooling over time gives shift
  tolerance; the network has to learn rotation tolerance from the data, which
  carries a uniformly random carrier phase on every example.
"""

from __future__ import annotations

import torch
from torch import nn


class IQCNN(nn.Module):
    """Four strided convolution blocks over the (I, Q) channel pair."""

    def __init__(self, n_classes: int = 6, width: int = 64, dropout: float = 0.3) -> None:
        super().__init__()
        w = width

        def block(c_in: int, c_out: int, kernel: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv1d(c_in, c_out, kernel, padding=kernel // 2, bias=False),
                nn.BatchNorm1d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
            )

        self.features = nn.Sequential(
            block(2, w, 7),
            block(w, w, 5),
            block(w, 2 * w, 5),
            block(2 * w, 2 * w, 3),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * w, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(batch, 2, length)`` IQ to class logits."""
        z = self.features(x)
        z = self.pool(z).flatten(1)
        return self.head(z)


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
