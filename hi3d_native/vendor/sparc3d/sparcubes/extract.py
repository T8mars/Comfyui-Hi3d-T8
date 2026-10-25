"""Isosurface extraction from a decoded sparse ``Point``."""

from __future__ import annotations

import functools

import torch

from sparc3d.sparcubes.sparse_cube import sparse_cube2verts
from sparc3d.ops.flexicubes import FlexiCubes
from sparc3d.structures.point import Mesh, Point


@functools.lru_cache(maxsize=None)
def _extractor(device: torch.device) -> FlexiCubes:
    """Return the per-device FlexiCubes instance, building its tables once."""
    return FlexiCubes(device=device)


def point_to_mesh(point: Point) -> Mesh:
    """Extract a triangle mesh from a decoded, activated ``Point``.

    Expects the point's ``feat`` to hold per-corner SDF and deformation channels, as
    produced by the decoder followed by :meth:`Point.activate`. When the decoder also
    emitted ``sdf_sign`` it overrides the sign of the SDF magnitude, which is how the
    network expresses occupancy independently of distance.

    Args:
        point: Single-batch decoded point cloud.

    Returns:
        Mesh in the point's normalized coordinate frame. ``Mesh.success`` is ``False``
        when the isosurface is empty.

    Raises:
        ValueError: ``point`` holds more than one batch element.
    """
    if int(point.batch[-1]) != 0:
        raise ValueError(
            f"sparse_marching_cubes supports a single batch element, got "
            f"{int(point.batch[-1]) + 1}"
        )
    feat, grid_coord = point.feat, point.grid_coord
    resolution = point.resolution

    # FlexiCubes' sparse ambiguity resolution assumes lexicographically ordered cubes.
    index = (grid_coord[:, 0] * resolution + grid_coord[:, 1]) * resolution + grid_coord[:, 2]
    order = torch.argsort(index)
    feat = feat[order]
    grid_coord = grid_coord[order]

    vert_feat, vert_layouts = point.get_vert_feat(feat)

    # Carry the sign logits through the corner-to-vertex averaging as an extra channel.
    has_sdf_sign = "sdf_sign" in point
    if has_sdf_sign:
        sdf_sign = torch.sigmoid(point.sdf_sign).unsqueeze(2)[order]
        vert_feat = torch.cat([sdf_sign, vert_feat], dim=2)

    vert_coord, vert_feat, cubes = sparse_cube2verts(grid_coord, vert_feat)

    if has_sdf_sign:
        sdf_sign, vert_feat = vert_feat[:, 0:1], vert_feat[:, 1:]

    sdf = Point.get_attr_from_feat(vert_feat, vert_layouts, "sdf")
    deform = Point.get_attr_from_feat(vert_feat, vert_layouts, "deform")

    vert_coord = Point.grid_to_coord(vert_coord + deform, point.matrix)

    if has_sdf_sign:
        sdf = torch.where(sdf_sign >= 0.5, 1, -1) * sdf.abs()

    with torch.amp.autocast(device_type="cuda", enabled=False):
        vertices, faces = _extractor(vert_feat.device)(
            voxelgrid_vertices=vert_coord,
            scalar_field=sdf,
            cube_idx=cubes,
            resolution=resolution,
            cube_index_map=grid_coord,
        )
    return Mesh(vertices=vertices, faces=faces)
