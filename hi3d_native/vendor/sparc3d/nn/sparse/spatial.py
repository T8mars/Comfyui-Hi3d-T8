"""Paired average downsampling and cached nearest-neighbor upsampling."""

# Adapted from Microsoft TRELLIS under the MIT License.
# Modified for SPARC3D to retain only paired downsample and upsample operations.

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .basic import SparseTensor


def _factor_tuple(factor: int | Sequence[int], dimensions: int) -> tuple[int, ...]:
    values = tuple(factor) if isinstance(factor, Sequence) else (factor,) * dimensions
    if len(values) != dimensions or any(value <= 0 for value in values):
        raise ValueError("factor must provide one positive value per spatial dimension")
    return values


class SparseDownsample(nn.Module):
    """Average features that collapse onto the same downsampled coordinate."""

    def __init__(self, factor: int | Sequence[int]):
        """
        Args:
            factor: Per-axis downsampling factor, or one factor for every axis.
        """
        super().__init__()
        self.factor = tuple(factor) if isinstance(factor, Sequence) else factor

    def forward(self, input: SparseTensor) -> SparseTensor:
        """Downsample ``input`` and register the inverse mapping."""
        dimensions = input.coords.shape[-1] - 1
        factor = _factor_tuple(self.factor, dimensions)
        coordinate_columns = list(input.coords.unbind(dim=-1))
        for axis, value in enumerate(factor):
            coordinate_columns[axis + 1] = coordinate_columns[axis + 1] // value

        extents = [
            int(coordinate_columns[axis + 1].max().item()) + 1
            for axis in range(dimensions)
        ]
        offsets = torch.cumprod(torch.tensor(extents[::-1]), 0).tolist()[::-1] + [1]
        codes = sum(
            coordinate * offset
            for coordinate, offset in zip(coordinate_columns, offsets)
        )
        unique_codes, inverse = codes.unique(return_inverse=True)
        new_feats = torch.scatter_reduce(
            torch.zeros(
                unique_codes.shape[0],
                input.feats.shape[1],
                device=input.device,
                dtype=input.dtype,
            ),
            dim=0,
            index=inverse[:, None].expand(-1, input.feats.shape[1]),
            src=input.feats,
            reduce="mean",
        )
        new_coords = torch.stack(
            [unique_codes // offsets[0]]
            + [
                (unique_codes // offsets[axis + 1]) % extents[axis]
                for axis in range(dimensions)
            ],
            dim=-1,
        )
        output = SparseTensor(
            new_feats,
            new_coords,
            shape=input.shape,
            scale=tuple(scale * value for scale, value in zip(input.scale, factor)),
            spatial_cache=input._spatial_cache,
        )
        output.register_spatial_cache(f"upsample_{factor}_coords", input.coords)
        output.register_spatial_cache(f"upsample_{factor}_layout", input.layout)
        output.register_spatial_cache(f"upsample_{factor}_idx", inverse)
        return output


class SparseUpsample(nn.Module):
    """Restore coordinates from a paired downsample cache."""

    def __init__(self, factor: int | Sequence[int]):
        """
        Args:
            factor: Per-axis factor, matching the paired :class:`SparseDownsample`.
        """
        super().__init__()
        self.factor = tuple(factor) if isinstance(factor, Sequence) else factor

    def forward(self, input: SparseTensor) -> SparseTensor:
        """Upsample ``input`` through the recorded inverse mapping.

        Raises:
            ValueError: ``input`` is not at a scale this factor could have produced,
                or no paired :class:`SparseDownsample` registered a cache for it.
        """
        dimensions = input.coords.shape[-1] - 1
        factor = _factor_tuple(self.factor, dimensions)
        if any(scale % value for scale, value in zip(input.scale, factor)):
            raise ValueError(
                f"cannot upsample by {factor} from scale {input.scale}; "
                "SparseUpsample must mirror its paired SparseDownsample"
            )
        coords = input.get_spatial_cache(f"upsample_{factor}_coords")
        layout = input.get_spatial_cache(f"upsample_{factor}_layout")
        inverse = input.get_spatial_cache(f"upsample_{factor}_idx")
        if coords is None or layout is None or inverse is None:
            raise ValueError(
                "upsample cache not found; SparseUpsample must follow its "
                "paired SparseDownsample"
            )
        return SparseTensor(
            input.feats[inverse],
            coords,
            shape=input.shape,
            layout=layout,
            scale=tuple(scale // value for scale, value in zip(input.scale, factor)),
            spatial_cache=input._spatial_cache,
        )
