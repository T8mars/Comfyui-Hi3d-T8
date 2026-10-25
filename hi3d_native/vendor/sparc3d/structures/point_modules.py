"""Module base and sequential container for layers that operate on a ``Point``."""

# Adapted from Pointcept / Point Transformer V3 under the MIT License.
# Modified for SPARC3D to drop the Lightning hyperparameters mixin and to
# dispatch sparse-convolution children through ``ops.sparse_backend``.

from __future__ import annotations

from typing import Optional

from torch import nn

from sparc3d.ops import sparse_backend
from sparc3d.structures.point import Point


class PointModule(nn.Module):
    """Marker base class for modules whose ``forward`` takes and returns a ``Point``."""


class PointSequential(PointModule):
    """
    Sequential container that dispatches on each child's kind.

    A ``PointModule`` child receives the whole ``Point``; a sparse convolution consumes
    and replaces ``sparse_conv_feat``; any other module is applied to ``feat``.
    """

    def __init__(self, *args: nn.Module):
        """
        Args:
            *args: Child modules, named by their index to match the checkpoint
                key layout.
        """
        super().__init__()
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def add(self, module: nn.Module, name: Optional[str] = None) -> None:
        """Append ``module``, naming it by position when ``name`` is ``None``."""
        if name is None:
            name = str(len(self._modules))
            if name in self._modules:
                raise KeyError("name exists")
        self.add_module(name, module)

    def forward(self, input: Point) -> Point:
        """Run each child in order, dispatching on how it consumes the point.

        A :class:`PointModule` receives the whole point; a sparse convolution receives
        ``sparse_conv_feat`` and its output is mirrored back onto ``feat``; anything
        else receives ``feat`` alone.
        """
        for module in self._modules.values():
            if isinstance(module, PointModule):
                input = module(input)
            elif sparse_backend.is_sparse_conv(module):
                assert isinstance(input, Point)
                input.sparse_conv_feat = module(input.sparse_conv_feat)
                input.feat = sparse_backend.get_feats(input.sparse_conv_feat)
            else:
                assert isinstance(input, Point)
                input.update_feat(module(input.feat))
        return input
