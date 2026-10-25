"""Sparse submanifold convolution over spconv."""

# Adapted from Microsoft TRELLIS under the MIT License.
# Modified for SPARC3D to route convolution through ``ops.sparse_backend``.

from __future__ import annotations

import torch
from torch import nn

from sparc3d.ops import sparse_backend

from .basic import SparseTensor


class _DeferredConv3d(nn.Module):
    """State-compatible placeholder used when spconv is not installed."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        bias: bool,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(
                out_channels,
                kernel_size,
                kernel_size,
                kernel_size,
                in_channels,
            )
        )
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, _input):
        """Raise the deferred sparse-backend import error."""
        raise ImportError("sparse convolution requires spconv")


class SparseConv3d(nn.Module):
    """Coordinate-preserving three-dimensional sparse convolution."""

    _sparc3d_convertible = True

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        bias: bool = True,
    ):
        """
        Args:
            in_channels: Input feature channels.
            out_channels: Output feature channels.
            kernel_size: Cubic kernel extent.
            bias: Add a bias term.
        """
        super().__init__()
        try:
            self.conv = sparse_backend.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                bias=bias,
            )
        except ImportError:
            # Construction and state-dict validation must work without spconv.
            # Forward still raises; this is not an inference fallback.
            self.conv = _DeferredConv3d(in_channels, out_channels, kernel_size, bias)
        self.out_channels = out_channels

    def forward(self, x: SparseTensor) -> SparseTensor:
        """Convolve ``x`` while preserving its coordinate lineage."""
        output = self.conv(x.backend_data())
        return SparseTensor(
            data=output,
            shape=x.shape[:1] + (self.out_channels,),
            layout=x.layout,
            scale=x.scale,
            spatial_cache=x._spatial_cache,
        )
