"""Sparse-convolution primitives built on spconv."""

from __future__ import annotations

from math import prod
from typing import Any, Optional, Tuple

import torch
from torch import nn

_SPARSE_CONV_FLAG = "_sparc3d_sparse_conv"
"""Instance attribute marking a module that consumes and returns a sparse tensor."""

_SPCONV_INT32_BYTE_LIMIT = 2**31 - 1
"""Largest input matrix spconv's implicit-GEMM indexing can address."""

_spconv_conv_class: Optional[type] = None


def _spconv():
    """Import spconv's PyTorch namespace, with an actionable error if it is absent."""
    try:
        import spconv.pytorch as spconv
    except ImportError as error:
        raise ImportError(
            "the sparse autoencoder needs spconv; install sparc3d[inference]"
        ) from error
    return spconv


def _submanifold_conv_class() -> type:
    """Build (once) a submanifold convolution that ignores autocast and preserves dtype.

    The released weights require a convolution that returns whatever precision its input
    features carry and casts its kernel to match, regardless of any enclosing autocast
    context. Stock spconv does the opposite: it downcasts under autocast, and outside one
    it has no kernel for half features at all. The subclass restores the required
    behaviour, at the cost of running the GEMM in the weight's precision.
    """
    global _spconv_conv_class
    if _spconv_conv_class is not None:
        return _spconv_conv_class

    spconv = _spconv()

    class SubMConv3d(spconv.SubMConv3d):
        def _large_forward(self, input: Any) -> Any:
            """Evaluate a submanifold convolution without cumm's int32 GEMM limit."""
            from spconv.pytorch import ops
            from spconv.pytorch.core import IndiceData

            indice_key = (
                f"{self.indice_key}_large" if self.indice_key is not None else None
            )
            data = input.find_indice_pair(indice_key)
            if data is None:
                out_indices, pairs, pair_counts = ops.get_indice_pairs(
                    input.indices,
                    input.batch_size,
                    input.spatial_shape,
                    spconv.ConvAlgo.Native,
                    self.kernel_size,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.output_padding,
                    True,
                    False,
                )
                data = IndiceData(
                    out_indices,
                    input.indices,
                    pairs,
                    pair_counts,
                    input.spatial_shape,
                    input.spatial_shape,
                    is_subm=True,
                    algo=spconv.ConvAlgo.Native,
                    ksize=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                )
            else:
                out_indices = data.out_indices
                pairs = data.indice_pairs
                pair_counts = data.indice_pair_num

            features = input.features
            kernel_volume = prod(self.kernel_size)
            weights = self.weight.reshape(
                self.out_channels, kernel_volume, self.in_channels
            )
            center = kernel_volume // 2
            output = torch.empty(
                (features.shape[0], self.out_channels),
                dtype=features.dtype,
                device=features.device,
            )
            chunk_rows = 250_000
            for start in range(0, len(features), chunk_rows):
                stop = min(start + chunk_rows, len(features))
                output[start:stop] = features[start:stop] @ weights[:, center, :].T

            def accumulate(
                source: torch.Tensor,
                target: torch.Tensor,
                weight: torch.Tensor,
            ) -> None:
                for start in range(0, len(source), chunk_rows):
                    stop = min(start + chunk_rows, len(source))
                    contribution = features[source[start:stop]] @ weight.T
                    output.index_add_(0, target[start:stop], contribution)

            for offset in range(center):
                count = int(pair_counts[offset])
                source = pairs[0, offset, :count].long()
                target = pairs[1, offset, :count].long()
                accumulate(source, target, weights[:, offset, :])
                accumulate(target, source, weights[:, kernel_volume - 1 - offset, :])
            if self.bias is not None:
                output.add_(self.bias)

            result = input.replace_feature(output)
            if indice_key is not None:
                result.indice_dict[indice_key] = data
            result.indices = out_indices
            return result

        def forward(self, input: Any) -> Any:
            feats_dtype = input.features.dtype
            matrix_element_size = max(
                input.features.element_size(), self.weight.element_size()
            )
            input_bytes = input.features.numel() * matrix_element_size
            with torch.autocast("cuda", enabled=False):
                if feats_dtype != self.weight.dtype:
                    input = input.replace_feature(input.features.to(self.weight.dtype))
                if input_bytes >= _SPCONV_INT32_BYTE_LIMIT:
                    # Both cumm convolution algorithms flatten the feature matrix
                    # with int32 indices. Chunked PyTorch GEMMs retain the exact
                    # native pair traversal for final 1536-scale decoding.
                    out = self._large_forward(input)
                else:
                    out = super().forward(input)
                if out.features.dtype != feats_dtype:
                    out = out.replace_feature(out.features.to(feats_dtype))
            return out

    _spconv_conv_class = SubMConv3d
    return _spconv_conv_class


def Conv3d(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: Optional[int] = None,
    bias: bool = True,
) -> nn.Module:
    """Build a submanifold sparse convolution.

    A submanifold convolution keeps the input's occupied coordinates and their row
    order, which is what lets the surrounding code index features positionally.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: Cubic kernel extent; must be odd.
        stride: Only ``1`` is supported; a strided convolution would change the
            occupied set and is not part of this architecture.
        padding: Only the submanifold value ``kernel_size // 2`` (or ``None``, which
            means the same) is supported.
        bias: Whether to add a per-output-channel bias.

    Returns:
        A module taking and returning an spconv sparse tensor.

    Raises:
        ValueError: ``kernel_size`` is even, or ``stride``/``padding`` are not the
            submanifold configuration.
    """
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd, got {kernel_size}")
    if stride != 1:
        raise ValueError(f"only stride 1 is supported, got {stride}")
    if padding is not None and padding != kernel_size // 2:
        raise ValueError(
            f"only submanifold padding {kernel_size // 2} is supported, got {padding}"
        )

    conv = _submanifold_conv_class()(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        bias=bias,
        # Submanifold output coordinates always equal the input's, so every
        # convolution over one tensor lineage can share the cached index pairs.
        indice_key=f"subm{kernel_size}",
    )
    setattr(conv, _SPARSE_CONV_FLAG, True)
    return conv


def is_sparse_conv(module: nn.Module) -> bool:
    """Return whether ``module`` was built by :func:`Conv3d`."""
    return getattr(module, _SPARSE_CONV_FLAG, False)


def sparse_tensor(
    feats: torch.Tensor,
    coords: torch.Tensor,
    spatial_shape: Optional[Tuple[int, int, int]] = None,
) -> Any:
    """Wrap features and coordinates in an spconv sparse tensor.

    Args:
        feats: Features ``[N, C]``.
        coords: Integer coordinates ``[N, 4]`` laid out as ``[batch, x, y, z]``.
        spatial_shape: Grid extent per axis; derived from ``coords`` when ``None``.

    Returns:
        The spconv sparse tensor.

    Raises:
        ValueError: ``coords`` is not ``[N, 4]``, holds negative entries, or exceeds
            ``spatial_shape``.
    """
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(f"expected coords [N, 4], got {tuple(coords.shape)}")
    coords = coords.int().contiguous()

    spconv = _spconv()
    grid = coords[:, 1:]
    if int(grid.min()) < 0:
        raise ValueError("grid coordinates must be non-negative")
    extent = (grid.max(dim=0).values + 1).tolist()
    if spatial_shape is None:
        spatial_shape = tuple(extent)
    elif any(bound < size for bound, size in zip(spatial_shape, extent)):
        raise ValueError(
            f"grid coordinates reach {extent}, outside spatial_shape {spatial_shape}"
        )
    return spconv.SparseConvTensor(
        features=feats,
        indices=coords,
        spatial_shape=list(spatial_shape),
        batch_size=int(coords[:, 0].max()) + 1,
    )


def get_feats(tensor: Any) -> torch.Tensor:
    """Return the feature matrix of a sparse tensor."""
    return tensor.features


def replace_feats(tensor: Any, feats: torch.Tensor) -> Any:
    """Return a sparse tensor with the same coordinates and new features.

    spconv rejects assignment to ``features``, so the result may be a different
    object than ``tensor``; callers must use the returned value.

    Args:
        tensor: Sparse tensor.
        feats: Replacement features ``[N, C]``, one row per coordinate.

    Returns:
        The updated sparse tensor.

    Raises:
        ValueError: ``feats`` has a different number of rows than the tensor holds.
    """
    current = get_feats(tensor)
    if feats.shape[0] != current.shape[0]:
        raise ValueError(
            f"expected {current.shape[0]} rows to match the coordinates, "
            f"got {feats.shape[0]}"
        )
    return tensor.replace_feature(feats)
