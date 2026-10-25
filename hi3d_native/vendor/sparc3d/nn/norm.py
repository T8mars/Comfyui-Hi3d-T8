"""Numerically stable normalization layers for the sparse transformer."""

# Adapted from Microsoft TRELLIS under the MIT License.
# Modified for SPARC3D to retain float32 layer normalization only.

from __future__ import annotations

import torch
from torch import nn


class LayerNorm32(nn.LayerNorm):
    """Layer normalization with float32 statistics and input-dtype output."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize ``x`` in float32 and restore its dtype."""
        return super().forward(x.float()).to(x.dtype)
