"""Linear projection over sparse feature rows."""

# Adapted from Microsoft TRELLIS under the MIT License.
# Modified for SPARC3D's private sparse tensor.

from __future__ import annotations

from torch import nn

from .basic import SparseTensor


class SparseLinear(nn.Linear):
    """Linear layer preserving sparse coordinates and caches."""

    def forward(self, input: SparseTensor) -> SparseTensor:
        """Project ``input`` features."""
        return input.replace(super().forward(input.feats))
