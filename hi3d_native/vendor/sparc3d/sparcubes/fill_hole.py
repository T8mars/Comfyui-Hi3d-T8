"""Boundary-loop detection and ear-clipping hole filling for extracted meshes."""

from __future__ import annotations

from typing import List

import numpy as np
import torch

from sparc3d.structures.point import Mesh


def fill_mesh_holes(mesh: Mesh, length_thresh: int = 10000) -> Mesh:
    """Close every boundary loop of a triangle mesh by ear clipping.

    Boundary half-edges are found on the GPU, then each loop is triangulated on the
    CPU with numpy. Loops longer than ``length_thresh`` are left open, since ear
    clipping is quadratic in the loop length.

    Args:
        mesh: Mesh to repair. Vertices are reused unchanged; only faces are added.
        length_thresh: Maximum boundary-loop length to fill.

    Returns:
        New mesh with the fill triangles appended.
    """
    vertices = mesh.vertices
    faces = mesh.faces
    device = vertices.device

    # A directed half-edge is on the boundary when its undirected edge occurs once.
    v0, v1, v2 = faces.unbind(dim=1)
    directed = torch.cat(
        [
            torch.stack([v0, v1], dim=1),
            torch.stack([v1, v2], dim=1),
            torch.stack([v2, v0], dim=1),
        ],
        dim=0,
    )
    undirected = torch.stack(
        [
            torch.minimum(directed[:, 0], directed[:, 1]),
            torch.maximum(directed[:, 0], directed[:, 1]),
        ],
        dim=1,
    )
    _, inv_idx, counts = torch.unique(
        undirected, return_inverse=True, return_counts=True, dim=0
    )
    boundary_directed = directed[counts[inv_idx] == 1]

    loops = _boundary_loops(boundary_directed.cpu().numpy().astype(int))

    new_faces = [
        _ear_clip(vertices, loop) for loop in loops if 3 <= len(loop) <= length_thresh
    ]
    if new_faces:
        appended = torch.from_numpy(np.vstack(new_faces)).long().to(device)
        faces = torch.cat([faces, appended], dim=0)

    return Mesh(vertices=vertices, faces=faces)


def _boundary_loops(boundary: np.ndarray) -> List[List[int]]:
    """Chain boundary half-edges ``[B, 2]`` into closed vertex loops.

    Half-edges are consumed at most once, so loops that share a vertex are still
    separated instead of merging into one traversal.
    """
    outgoing: dict[int, List[int]] = {}
    for idx, (start, _) in enumerate(boundary):
        outgoing.setdefault(start, []).append(idx)
    visited = [False] * len(boundary)

    loops: List[List[int]] = []
    for i in range(len(boundary)):
        if visited[i]:
            continue
        visited[i] = True
        start_v = boundary[i][0]
        loop = [start_v, boundary[i][1]]

        while loop[-1] != start_v:
            for next_idx in outgoing.get(loop[-1], []):
                if not visited[next_idx]:
                    visited[next_idx] = True
                    loop.append(boundary[next_idx][1])
                    break
            else:
                break  # dead end: not a closed loop

        if loop[0] == loop[-1] and len(loop) > 3:
            loop.pop()  # drop the duplicated start vertex
            loops.append(loop)
    return loops


def _ear_clip(vertices: torch.Tensor, boundary: List[int]) -> np.ndarray:
    """Triangulate one boundary loop, repeatedly clipping its sharpest corner."""
    boundline = vertices[torch.tensor(boundary, device=vertices.device)].cpu().numpy()
    boundline_idx = np.array(boundary, dtype=int)

    faces_hole = []
    while True:
        if len(boundline_idx) == 3:
            faces_hole.append(boundline_idx[[2, 1, 0]])
            break

        next_pts = np.roll(boundline, -1, axis=0)
        T = next_pts - boundline
        T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-6)
        T_prev = np.roll(T, 1, axis=0)
        N = np.cross(-T_prev, T)

        # Signed interior angle at each corner, relative to the loop centroid.
        center = boundline.mean(axis=0, keepdims=True)
        P = next_pts - center
        Q = boundline - center
        sin_sign = np.sign((np.cross(P, Q) * N).sum(axis=1, keepdims=True))
        sin_val = np.linalg.norm(N, axis=1, keepdims=True) * sin_sign
        cos_val = -(T_prev * T).sum(axis=1, keepdims=True)
        A = np.arctan2(sin_val, cos_val).reshape(-1)

        # Clip the sharpest corner: 1/A ranks the smallest positive angle first and
        # every reflex corner last. Scoring exp(1/A) instead ranks corners the same
        # way, but a folded corner has exactly antiparallel edges, hence A of exactly
        # 0, and dividing by it is fatal rather than merely a warning wherever
        # floating-point traps are unmasked. Such a corner is clipped first, which is
        # where the ranking wants it. Ties among comparably sharp corners can break
        # differently than under exp, which rounds large scores together.
        collapsed = A == 0.0
        score = np.where(collapsed, np.inf, 1.0 / np.where(collapsed, 1.0, A))
        ear = int(np.argmax(score))

        i0 = (ear + 1) % len(boundline_idx)
        i2 = (ear - 1) % len(boundline_idx)
        faces_hole.append(boundline_idx[[i0, ear, i2]])

        boundline = np.delete(boundline, ear, axis=0)
        boundline_idx = np.delete(boundline_idx, ear)

    return np.vstack(faces_hole)
