"""Nonlinear activations over sparse feature rows."""

# Adapted from Microsoft TRELLIS under the MIT License.
# Modified for SPARC3D to retain GELU only.

from __future__ import annotations

from torch import nn

from .basic import SparseTensor


class SparseGELU(nn.GELU):
    """GELU activation preserving sparse coordinates and caches."""

    def forward(self, input: SparseTensor) -> SparseTensor:
        """Activate ``input`` features."""
        return input.replace(super().forward(input.feats))
