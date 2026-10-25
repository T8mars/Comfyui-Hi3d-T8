"""Feed-forward building block for the sparse transformer."""

# Adapted from Microsoft TRELLIS under the MIT License.
# Modified for SPARC3D to retain the sparse feed-forward path only.

from __future__ import annotations

from torch import nn

from ..basic import SparseTensor
from ..linear import SparseLinear
from ..nonlinearity import SparseGELU


class SparseFeedForwardNet(nn.Module):
    """Two-layer sparse MLP with tanh-approximated GELU."""

    def __init__(self, channels: int, mlp_ratio: float = 4.0):
        """
        Args:
            channels: Input and output channels.
            mlp_ratio: Hidden width as a multiple of ``channels``.
        """
        super().__init__()
        hidden_channels = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            SparseLinear(channels, hidden_channels),
            SparseGELU(approximate="tanh"),
            SparseLinear(hidden_channels, channels),
        )

    def forward(self, x: SparseTensor) -> SparseTensor:
        """Transform sparse features with the MLP."""
        return self.mlp(x)
