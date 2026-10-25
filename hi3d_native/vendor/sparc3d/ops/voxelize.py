"""Mesh voxelization operations on the GPU.

Ported from pytorch-voxelization (Zhihao Li).
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Sequence, Union

import torch

from sparc3d.ops._triton import load_triton_ops

try:
    import trimesh
except ImportError:  # pragma: no cover
    trimesh = None  # type: ignore

DEFAULT_PAD = 3
"""Default empty voxel margin per side."""

MIN_RESOLUTION = 2 * DEFAULT_PAD + 1
"""Smallest grid with a positive voxelization scale at :data:`DEFAULT_PAD`."""

MAX_RESOLUTION = 2048
"""Largest grid the kernel can encode. Voxel coordinates are packed into an
11-bit field per axis, so a coordinate of 2048 or more collides with its
neighbours instead of failing."""


class MeshVoxelizer:
    """
    GPU mesh voxelizer with centering, uniform scale, optional margin (pad), and phase shift.

    The object's center (from the eps-expanded bounding box) maps to
    (resolution/2, resolution/2, resolution/2) plus an optional voxel-space ``shift``.
    Default ``pad=3`` leaves a 3-voxel margin on each side, so the usable extent is
    ``resolution - 2*pad - 2*eps``.

    Reversible transforms: ``vertex_to_voxel`` and ``voxel_to_vertex``.
    ``shift`` changes lattice phase without changing scale; ``pad`` changes scale.
    """

    def __init__(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        resolution: int,
        eps: float = 1e-6,
        *,
        pad: int = DEFAULT_PAD,
        shift: Optional[Union[torch.Tensor, Sequence[float], float]] = None,
    ):
        """
        Args:
            vertices: Tensor of shape [num_vertices, 3] with xyz coordinates.
            faces: Tensor of shape [num_faces, 3] with vertex indices.
            resolution: Voxel grid resolution (e.g. 64 for a 64³ grid).
            eps: Epsilon to expand the bounding box for numerical stability.
            pad: Empty margin in voxel units on each side of the grid.
            shift: Optional voxel-space translation after centering and scaling.
                Scalar or length-3; typical values are in [-0.5, 0.5] per axis.

        Raises:
            ValueError: ``resolution`` is outside ``[1, MAX_RESOLUTION]``, or
                ``vertices`` / ``faces`` are not CUDA tensors.
        """
        self.resolution = int(resolution)
        self.pad = int(pad)
        self.eps = float(eps)

        if not 1 <= self.resolution <= MAX_RESOLUTION:
            raise ValueError(
                f"resolution must be in [1, {MAX_RESOLUTION}] for the 11-bit voxel "
                f"key packing, got {self.resolution}"
            )
        if not vertices.is_cuda:
            raise ValueError("vertices must be a CUDA tensor")
        if not faces.is_cuda:
            raise ValueError("faces must be a CUDA tensor")

        self.faces = faces
        self.original_vertices = vertices

        self.vertices_min = self.original_vertices.min(dim=0)[0]
        self.vertices_max = self.original_vertices.max(dim=0)[0]

        self.expanded_min = self.vertices_min - self.eps
        self.expanded_max = self.vertices_max + self.eps
        self.center = (self.expanded_min + self.expanded_max) / 2.0

        max_range = (self.expanded_max - self.expanded_min).max()
        self.scale = (self.resolution - self.pad * 2 - 2 * self.eps) / max_range

        self.shift = self._normalize_shift(shift)
        self.norm_vertices = self.vertex_to_voxel()

    def _normalize_shift(self, shift) -> torch.Tensor:
        device = self.original_vertices.device
        dtype = self.original_vertices.dtype

        if shift is None:
            return torch.zeros((3,), device=device, dtype=dtype)

        if isinstance(shift, (int, float)):
            return torch.full((3,), float(shift), device=device, dtype=dtype)

        if isinstance(shift, torch.Tensor):
            shift_t = shift.to(device=device, dtype=dtype).reshape(-1)
        else:
            shift_t = torch.tensor(list(shift), device=device, dtype=dtype).reshape(-1)

        if shift_t.numel() != 3:
            raise ValueError(
                f"shift must be scalar or len-3, got shape={tuple(shift_t.shape)}"
            )
        return shift_t

    def vertex_to_voxel(self, vertices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Map original vertex coordinates to voxel coordinates (float).

        The object center maps to (resolution/2, resolution/2, resolution/2) + shift.
        """
        if vertices is None:
            vertices = self.original_vertices
        centered_vertices = vertices - self.center
        return centered_vertices * self.scale + (self.resolution / 2.0) + self.shift

    def voxel_to_vertex(self, voxel_coords: torch.Tensor) -> torch.Tensor:
        """Map voxel coordinates back to original vertex coordinates."""
        return (
            (voxel_coords - (self.resolution / 2.0) - self.shift) / self.scale
            + self.center
        )

    def voxelize(self) -> Dict[str, torch.Tensor]:
        """
        Voxelize the mesh on normalized vertices.

        Returns:
            Dictionary with ``voxel_coords``, ``voxel_face_idx``, ``voxel_face_start``,
            ``voxel_face_count``, and ``face_voxel_count``.
        """
        norm_vertices = self.norm_vertices.to(torch.float32)
        faces = self.faces.to(torch.int32)

        (
            encoded_coords,
            voxel_face_idx,
            voxel_face_start_idx,
            voxel_face_count,
            face_voxel_count,
        ) = load_triton_ops().coarse_voxelize(
            norm_vertices, faces, self.resolution
        )

        mask = (1 << 11) - 1
        encoded_coords = encoded_coords.to(torch.int64)
        x = encoded_coords & mask
        y = (encoded_coords >> 11) & mask
        z = (encoded_coords >> 22) & mask
        voxel_coords = torch.stack([x, y, z], dim=-1).to(torch.uint16)

        return {
            "voxel_coords": voxel_coords,
            "voxel_face_idx": voxel_face_idx,
            "voxel_face_start": voxel_face_start_idx,
            "voxel_face_count": voxel_face_count,
            "face_voxel_count": face_voxel_count,
        }


def adaptive_vox(
    mesh,
    resolution: int = 64,
    target_num: int = 80000,
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    """
    Shrink-to-fit surface voxelization.

    Voxelize at ``int(resolution * scale)`` and, while the occupied-voxel count
    exceeds ``target_num``, multiply ``scale`` by 0.95 and retry.

    Args:
        mesh: A ``trimesh.Trimesh`` (or any object with ``.vertices`` / ``.faces``).
        resolution: Nominal grid resolution.
        target_num: Maximum number of occupied voxels to accept.
        device: CUDA device used for voxelization.

    Returns:
        Int32 tensor of shape ``[N, 3]`` with unique voxel coordinates.

    Raises:
        ValueError: ``target_num`` is not positive, ``resolution`` is below
            :data:`MIN_RESOLUTION`, or the retries shrank the grid below that bound,
            where the voxelization scale turns non-positive, without fitting
            ``target_num``.

    Warns:
        UserWarning: The accepted grid is coarser than ``resolution``.
    """
    if target_num < 1:
        raise ValueError(f"target_num must be positive, got {target_num}")
    if resolution < MIN_RESOLUTION:
        raise ValueError(
            f"resolution must be at least {MIN_RESOLUTION} for a positive voxelization "
            f"scale, got {resolution}"
        )

    if trimesh is not None and isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(f"adaptive_vox requires a CUDA device, got {device}")
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(mesh.faces, dtype=torch.int32, device=device)

    scale = 1.0
    while True:
        cur_res = int(resolution * scale)
        if cur_res < MIN_RESOLUTION:
            raise ValueError(
                f"shrinking from resolution {resolution} reached {cur_res}, below the "
                f"smallest grid with a positive voxelization scale ({MIN_RESOLUTION}); "
                f"target_num={target_num} is unreachable for this mesh"
            )
        vox = MeshVoxelizer(vertices, faces, resolution=cur_res)
        coords = vox.voxelize()["voxel_coords"].to(device=device, dtype=torch.int32)
        if coords.shape[0] <= target_num:
            break
        scale = scale * 0.95

    if cur_res != resolution:
        warnings.warn(
            f"adaptive_vox settled at resolution {cur_res} instead of the requested "
            f"{resolution} to fit target_num={target_num}; the returned coordinates "
            "are on the coarser grid",
            stacklevel=2,
        )

    return coords
