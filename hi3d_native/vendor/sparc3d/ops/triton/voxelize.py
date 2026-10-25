"""Triton surface-voxelization kernels and CSR construction."""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _axis_overlaps(
    v0x,
    v0y,
    v0z,
    v1x,
    v1y,
    v1z,
    v2x,
    v2y,
    v2z,
    ax,
    ay,
    az,
):
    p0 = v0x * ax + v0y * ay + v0z * az
    p1 = v1x * ax + v1y * ay + v1z * az
    p2 = v2x * ax + v2y * ay + v2z * az
    lower = tl.minimum(tl.minimum(p0, p1), p2)
    upper = tl.maximum(tl.maximum(p0, p1), p2)
    radius = 0.5 * (tl.abs(ax) + tl.abs(ay) + tl.abs(az))
    return (lower <= radius) & (upper >= -radius)


@triton.jit
def _triangle_overlaps_voxel(
    a0,
    a1,
    a2,
    b0,
    b1,
    b2,
    c0,
    c1,
    c2,
    x,
    y,
    z,
):
    center_x = x.to(tl.float32) + 0.5
    center_y = y.to(tl.float32) + 0.5
    center_z = z.to(tl.float32) + 0.5
    v0x, v0y, v0z = a0 - center_x, a1 - center_y, a2 - center_z
    v1x, v1y, v1z = b0 - center_x, b1 - center_y, b2 - center_z
    v2x, v2y, v2z = c0 - center_x, c1 - center_y, c2 - center_z

    overlaps = (
        (tl.minimum(tl.minimum(v0x, v1x), v2x) <= 0.5)
        & (tl.maximum(tl.maximum(v0x, v1x), v2x) >= -0.5)
        & (tl.minimum(tl.minimum(v0y, v1y), v2y) <= 0.5)
        & (tl.maximum(tl.maximum(v0y, v1y), v2y) >= -0.5)
        & (tl.minimum(tl.minimum(v0z, v1z), v2z) <= 0.5)
        & (tl.maximum(tl.maximum(v0z, v1z), v2z) >= -0.5)
    )

    e0x, e0y, e0z = v1x - v0x, v1y - v0y, v1z - v0z
    e1x, e1y, e1z = v2x - v1x, v2y - v1y, v2z - v1z
    e2x, e2y, e2z = v0x - v2x, v0y - v2y, v0z - v2z

    overlaps &= _axis_overlaps(
        v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z, 0.0, e0z, -e0y
    )
    overlaps &= _axis_overlaps(
        v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z, -e0z, 0.0, e0x
    )
    overlaps &= _axis_overlaps(
        v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z, e0y, -e0x, 0.0
    )
    overlaps &= _axis_overlaps(
        v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z, 0.0, e1z, -e1y
    )
    overlaps &= _axis_overlaps(
        v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z, -e1z, 0.0, e1x
    )
    overlaps &= _axis_overlaps(
        v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z, e1y, -e1x, 0.0
    )
    overlaps &= _axis_overlaps(
        v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z, 0.0, e2z, -e2y
    )
    overlaps &= _axis_overlaps(
        v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z, -e2z, 0.0, e2x
    )
    overlaps &= _axis_overlaps(
        v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z, e2y, -e2x, 0.0
    )

    nx = e0y * e1z - e0z * e1y
    ny = e0z * e1x - e0x * e1z
    nz = e0x * e1y - e0y * e1x
    plane_offset = -(nx * v0x + ny * v0y + nz * v0z)
    plane_radius = 0.5 * (tl.abs(nx) + tl.abs(ny) + tl.abs(nz))
    return overlaps & (tl.abs(plane_offset) <= plane_radius)


@triton.jit
def _load_triangle(vertices_ptr, faces_ptr, face_id):
    index0 = tl.load(faces_ptr + face_id * 3)
    index1 = tl.load(faces_ptr + face_id * 3 + 1)
    index2 = tl.load(faces_ptr + face_id * 3 + 2)
    a0 = tl.load(vertices_ptr + index0 * 3)
    a1 = tl.load(vertices_ptr + index0 * 3 + 1)
    a2 = tl.load(vertices_ptr + index0 * 3 + 2)
    b0 = tl.load(vertices_ptr + index1 * 3)
    b1 = tl.load(vertices_ptr + index1 * 3 + 1)
    b2 = tl.load(vertices_ptr + index1 * 3 + 2)
    c0 = tl.load(vertices_ptr + index2 * 3)
    c1 = tl.load(vertices_ptr + index2 * 3 + 1)
    c2 = tl.load(vertices_ptr + index2 * 3 + 2)
    return a0, a1, a2, b0, b1, b2, c0, c1, c2


@triton.jit
def _count_intersections_kernel(
    vertices_ptr,
    faces_ptr,
    counts_ptr,
    resolution: tl.constexpr,
):
    face_id = tl.program_id(0)
    a0, a1, a2, b0, b1, b2, c0, c1, c2 = _load_triangle(
        vertices_ptr, faces_ptr, face_id
    )
    start_x = tl.maximum(
        tl.floor(tl.minimum(tl.minimum(a0, b0), c0)).to(tl.int32), 0
    )
    start_y = tl.maximum(
        tl.floor(tl.minimum(tl.minimum(a1, b1), c1)).to(tl.int32), 0
    )
    start_z = tl.maximum(
        tl.floor(tl.minimum(tl.minimum(a2, b2), c2)).to(tl.int32), 0
    )
    end_x = tl.minimum(
        tl.floor(tl.maximum(tl.maximum(a0, b0), c0)).to(tl.int32),
        resolution - 1,
    )
    end_y = tl.minimum(
        tl.floor(tl.maximum(tl.maximum(a1, b1), c1)).to(tl.int32),
        resolution - 1,
    )
    end_z = tl.minimum(
        tl.floor(tl.maximum(tl.maximum(a2, b2), c2)).to(tl.int32),
        resolution - 1,
    )

    count = 0
    for z in tl.range(start_z, end_z + 1):
        for y in tl.range(start_y, end_y + 1):
            for x in tl.range(start_x, end_x + 1):
                count += _triangle_overlaps_voxel(
                    a0, a1, a2, b0, b1, b2, c0, c1, c2, x, y, z
                ).to(tl.int32)
    tl.store(counts_ptr + face_id, count)


@triton.jit
def _emit_intersections_kernel(
    vertices_ptr,
    faces_ptr,
    offsets_ptr,
    keys_ptr,
    face_ids_ptr,
    resolution: tl.constexpr,
):
    face_id = tl.program_id(0)
    a0, a1, a2, b0, b1, b2, c0, c1, c2 = _load_triangle(
        vertices_ptr, faces_ptr, face_id
    )
    start_x = tl.maximum(
        tl.floor(tl.minimum(tl.minimum(a0, b0), c0)).to(tl.int32), 0
    )
    start_y = tl.maximum(
        tl.floor(tl.minimum(tl.minimum(a1, b1), c1)).to(tl.int32), 0
    )
    start_z = tl.maximum(
        tl.floor(tl.minimum(tl.minimum(a2, b2), c2)).to(tl.int32), 0
    )
    end_x = tl.minimum(
        tl.floor(tl.maximum(tl.maximum(a0, b0), c0)).to(tl.int32),
        resolution - 1,
    )
    end_y = tl.minimum(
        tl.floor(tl.maximum(tl.maximum(a1, b1), c1)).to(tl.int32),
        resolution - 1,
    )
    end_z = tl.minimum(
        tl.floor(tl.maximum(tl.maximum(a2, b2), c2)).to(tl.int32),
        resolution - 1,
    )

    output = tl.load(offsets_ptr + face_id)
    for z in tl.range(start_z, end_z + 1):
        for y in tl.range(start_y, end_y + 1):
            for x in tl.range(start_x, end_x + 1):
                hit = _triangle_overlaps_voxel(
                    a0, a1, a2, b0, b1, b2, c0, c1, c2, x, y, z
                )
                key = (
                    (z.to(tl.int64) << 22)
                    | (y.to(tl.int64) << 11)
                    | x.to(tl.int64)
                )
                tl.store(keys_ptr + output, key, mask=hit)
                tl.store(face_ids_ptr + output, face_id, mask=hit)
                output += hit.to(tl.int32)


def coarse_voxelize(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    resolution: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Voxelize triangles and return voxel-to-face CSR tensors.

    Args:
        vertices: CUDA float coordinates with shape ``[V, 3]`` in voxel space.
        faces: CUDA triangle indices with shape ``[F, 3]``.
        resolution: Side length of the voxel grid.

    Returns:
        Encoded coordinates, grouped face IDs, CSR starts, CSR counts, and
        per-face voxel counts.

    Raises:
        ValueError: Inputs do not satisfy the low-level voxelizer contract.
    """
    if not vertices.is_cuda or not faces.is_cuda:
        raise ValueError("vertices and faces must be CUDA tensors")
    if vertices.device != faces.device:
        raise ValueError("vertices and faces must be on the same device")
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape [num_vertices, 3]")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape [num_faces, 3]")
    if resolution < 1 or resolution > 2048:
        raise ValueError(f"resolution must be in [1, 2048], got {resolution}")

    vertices = vertices.to(torch.float32).contiguous()
    faces = faces.to(torch.int32).contiguous()
    num_faces = faces.shape[0]
    counts = torch.zeros(num_faces, dtype=torch.int32, device=faces.device)
    if num_faces == 0:
        empty_u64 = torch.empty(0, dtype=torch.uint64, device=faces.device)
        empty_u32 = torch.empty(0, dtype=torch.uint32, device=faces.device)
        return empty_u64, empty_u32, counts, counts, counts

    _count_intersections_kernel[(num_faces,)](
        vertices, faces, counts, resolution, num_warps=1
    )
    inclusive = torch.cumsum(counts, dim=0, dtype=torch.int32)
    offsets = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=faces.device), inclusive[:-1]]
    )
    total = int(inclusive[-1].item())
    if total == 0:
        empty_u64 = torch.empty(0, dtype=torch.uint64, device=faces.device)
        empty_u32 = torch.empty(0, dtype=torch.uint32, device=faces.device)
        empty_i32 = torch.empty(0, dtype=torch.int32, device=faces.device)
        return empty_u64, empty_u32, empty_i32, empty_i32, counts

    keys = torch.empty(total, dtype=torch.int64, device=faces.device)
    face_ids = torch.empty(total, dtype=torch.int32, device=faces.device)
    _emit_intersections_kernel[(num_faces,)](
        vertices,
        faces,
        offsets,
        keys,
        face_ids,
        resolution,
        num_warps=1,
    )

    sorted_keys, order = torch.sort(keys, stable=True)
    grouped_faces = face_ids[order]
    unique_keys, unique_counts = torch.unique_consecutive(
        sorted_keys, return_counts=True
    )
    unique_counts = unique_counts.to(torch.int32)
    unique_starts = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=faces.device),
            torch.cumsum(unique_counts, dim=0, dtype=torch.int32)[:-1],
        ]
    )
    return (
        unique_keys.to(torch.uint64),
        grouped_faces.to(torch.uint32),
        unique_starts,
        unique_counts,
        counts,
    )
