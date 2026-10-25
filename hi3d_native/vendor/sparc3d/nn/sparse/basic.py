"""Private sparse tensor wrapper with contiguous batches and spatial caches."""

# Adapted from Microsoft TRELLIS under the MIT License.
# Modified for SPARC3D to use the local sparse backend and paired resampling cache.

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from sparc3d.ops import sparse_backend


class SparseTensor:
    """Sparse features with batch-first coordinates and shared spatial caches."""

    def __init__(
        self,
        feats: torch.Tensor | None = None,
        coords: torch.Tensor | None = None,
        shape: torch.Size | None = None,
        layout: list[slice] | None = None,
        *,
        data: Any | None = None,
        scale: tuple[int, ...] = (1, 1, 1),
        spatial_cache: dict[str, dict[str, Any]] | None = None,
    ):
        """
        Args:
            feats: Features ``[N, C]``, one row per occupied coordinate.
            coords: Coordinates ``[N, 4]`` as ``(batch, x, y, z)``, sorted by batch.
            shape: Logical shape; derived from ``coords`` and ``feats`` when omitted.
            layout: Per-batch row slices; recomputed from ``coords`` when omitted.
            data: Backend sparse tensor to adopt instead of ``feats`` and ``coords``.
            scale: Cumulative downsampling factor relative to the original grid, so
                ``(1, 1, 1)`` is the original resolution and each downsample
                multiplies it. It also keys :attr:`_spatial_cache`, so it has to
                differ between nesting levels.
            spatial_cache: Cache shared with tensors over the same geometry, which is
                how a downsample hands its inverse mapping to the paired upsample.

        Raises:
            ValueError: Both ``data`` and explicit features or coordinates were given,
                either is missing, the shapes disagree, or the tensor would be empty.
        """
        if data is not None:
            if feats is not None or coords is not None:
                raise ValueError("provide either data or feats and coords")
            feats = sparse_backend.get_feats(data)
            coords = data.indices
        if feats is None or coords is None:
            raise ValueError("feats and coords are required")
        if coords.ndim != 2 or coords.shape[1] != 4:
            raise ValueError(f"expected coords [N, 4], got {tuple(coords.shape)}")
        if feats.shape[0] != coords.shape[0]:
            raise ValueError("features and coordinates must have the same row count")
        if coords.shape[0] == 0:
            raise ValueError("a SparseTensor cannot be empty")
        self._feats = feats
        self._coords = coords.int().contiguous()
        self.data = data
        batch_size = int(self._coords[:, 0].max().item()) + 1
        self._shape = shape or torch.Size((batch_size, *feats.shape[1:]))
        self._layout = layout or self._calculate_layout(self._coords, batch_size)
        self._scale = tuple(scale)
        self._spatial_cache = spatial_cache if spatial_cache is not None else {}
        self._validate_layout()

    @staticmethod
    def _calculate_layout(coords: torch.Tensor, batch_size: int) -> list[slice]:
        counts = torch.bincount(coords[:, 0].long(), minlength=batch_size)
        offsets = torch.cumsum(counts, dim=0)
        return [
            slice(int(offsets[i] - counts[i]), int(offsets[i]))
            for i in range(batch_size)
        ]

    def _validate_layout(self) -> None:
        for batch, span in enumerate(self._layout):
            if not torch.all(self._coords[span, 0] == batch):
                raise ValueError("coordinates for each batch must be contiguous")

    def backend_data(self) -> Any:
        """Return the sparse-backend tensor, creating it lazily."""
        if self.data is None:
            self.data = sparse_backend.sparse_tensor(self._feats, self._coords)
        return self.data

    @property
    def shape(self) -> torch.Size:
        """Logical shape ``[B, *feature_shape]``."""
        return self._shape

    @property
    def layout(self) -> list[slice]:
        """Contiguous feature slice for each batch element."""
        return self._layout

    @property
    def feats(self) -> torch.Tensor:
        """Feature values in coordinate-row order."""
        return self._feats

    @property
    def coords(self) -> torch.Tensor:
        """Batch-first integer coordinates in ``[batch, x, y, z]`` order."""
        return self._coords

    @property
    def dtype(self) -> torch.dtype:
        """Feature dtype."""
        return self.feats.dtype

    @property
    def device(self) -> torch.device:
        """Feature device."""
        return self.feats.device

    @property
    def scale(self) -> tuple[int, ...]:
        """Current spatial scale tracked by paired resampling."""
        return self._scale

    def replace(self, feats: torch.Tensor) -> "SparseTensor":
        """Return a tensor sharing coordinates and caches with new features."""
        if feats.shape[0] != self.feats.shape[0]:
            raise ValueError("replacement features must preserve the sparse row count")
        data = None
        if self.data is not None and feats.ndim == 2:
            data = sparse_backend.replace_feats(self.data, feats)
        return SparseTensor(
            feats=feats if data is None else None,
            coords=self.coords if data is None else None,
            data=data,
            shape=torch.Size((self.shape[0], *feats.shape[1:])),
            layout=self.layout,
            scale=self._scale,
            spatial_cache=self._spatial_cache,
        )

    def reshape(self, *shape: int) -> "SparseTensor":
        """Reshape feature dimensions while retaining the sparse coordinates."""
        return self.replace(self.feats.reshape(self.feats.shape[0], *shape))

    def type(self, dtype: torch.dtype) -> "SparseTensor":
        """Cast features to ``dtype``."""
        return self.replace(self.feats.to(dtype=dtype))

    def register_spatial_cache(self, key: str, value: Any) -> None:
        """Store ``value`` under ``key`` for the current scale."""
        self._spatial_cache.setdefault(str(self._scale), {})[key] = value

    def get_spatial_cache(self, key: str) -> Any | None:
        """Return a cached value for the current scale."""
        return self._spatial_cache.get(str(self._scale), {}).get(key)

    def _elementwise(
        self,
        other: torch.Tensor | "SparseTensor" | float,
        operation: Callable[[torch.Tensor, Any], torch.Tensor],
    ) -> "SparseTensor":
        if isinstance(other, SparseTensor):
            if not torch.equal(self.coords, other.coords):
                raise ValueError("sparse operands must have identical coordinates")
            other = other.feats
        elif isinstance(other, torch.Tensor) and other.ndim >= 1:
            if other.shape[0] == self.shape[0] and other.shape != self.feats.shape:
                other = sparse_batch_broadcast(self, other)
        return self.replace(operation(self.feats, other))

    def __add__(self, other):
        return self._elementwise(other, torch.add)

    def __mul__(self, other):
        return self._elementwise(other, torch.mul)


def sparse_batch_broadcast(input: SparseTensor, other: torch.Tensor) -> torch.Tensor:
    """Broadcast a batch-first tensor over sparse feature rows."""
    output = torch.empty_like(input.feats)
    for batch, span in enumerate(input.layout):
        output[span] = other[batch]
    return output
