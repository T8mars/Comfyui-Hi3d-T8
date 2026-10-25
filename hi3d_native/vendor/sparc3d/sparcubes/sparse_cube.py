"""Sparse voxel-grid to cube-vertex primitives for decoded mesh extraction."""

from __future__ import annotations

from typing import Tuple

import torch

CUBE_CORNERS = torch.tensor(
    [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ],
    dtype=torch.int,
)
"""Corner offsets of a unit cube, in the order FlexiCubes indexes cube vertices."""


def construct_voxel_grid(coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Expand voxel coordinates into their unique cube vertices.

    Args:
        coords: Voxel coordinates ``[V, 3]``.

    Returns:
        ``(verts_unique, cubes)`` where ``verts_unique`` is ``[N, 3]`` sorted unique
        cube-corner coordinates and ``cubes`` is ``[V, 8]`` indices into it.
    """
    verts = (CUBE_CORNERS.unsqueeze(0).to(coords) + coords.unsqueeze(1)).reshape(-1, 3)
    verts_unique, inverse_indices = torch.unique(verts, dim=0, return_inverse=True)
    cubes = inverse_indices.reshape(-1, 8)
    return verts_unique, cubes


def cubes_to_verts(
    num_verts: int,
    cubes: torch.Tensor,
    value: torch.Tensor,
    reduce: str = "mean",
) -> torch.Tensor:
    """Scatter per-cube-corner values onto shared cube vertices.

    Args:
        num_verts: Number of unique cube vertices.
        cubes: Vertex index per cube corner ``[V, 8]``.
        value: Values to scatter ``[V, 8, M]``.
        reduce: ``torch.scatter_reduce`` reduction applied to colliding corners.

    Returns:
        Reduced per-vertex values ``[num_verts, M]``.
    """
    M = value.shape[2]  # number of channels
    reduced = torch.zeros(num_verts, M, device=cubes.device, dtype=value.dtype)
    return torch.scatter_reduce(
        reduced,
        0,
        cubes.unsqueeze(-1).expand(-1, -1, M).flatten(0, 1),
        value.flatten(0, 1),
        reduce=reduce,
        include_self=False,
    )


def sparse_cube2verts(
    coords: torch.Tensor,
    feats: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Average per-cube corner features onto the shared sparse cube vertices.

    Args:
        coords: Voxel coordinates ``[V, 3]``.
        feats: Per-corner features ``[V, 8, M]``.

    Returns:
        ``(new_coords, new_feats, cubes)``.
    """
    new_coords, cubes = construct_voxel_grid(coords)
    new_feats = cubes_to_verts(new_coords.shape[0], cubes, feats)
    return new_coords, new_feats, cubes
