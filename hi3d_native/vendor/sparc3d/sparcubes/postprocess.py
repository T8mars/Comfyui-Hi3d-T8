"""Geometry-only mesh validation, filtering, simplification, and normalization."""

from __future__ import annotations

from numbers import Integral

import numpy as np
import trimesh

COMPONENT_FACE_RATIO = 0.005
"""Minimum fraction of total faces retained for a non-largest component."""


def validate_mesh(mesh: trimesh.Trimesh | trimesh.Scene) -> trimesh.Trimesh:
    """Validate and flatten a triangle mesh or transformed scene.

    Args:
        mesh: Triangle mesh or scene whose instance transforms should be applied.

    Returns:
        Material-free triangle mesh with scene geometries concatenated.

    Raises:
        TypeError: ``mesh`` is neither a ``Trimesh`` nor a ``Scene``.
        ValueError: Geometry is empty, non-triangular, non-finite, or has zero
            extent.
    """
    if isinstance(mesh, trimesh.Scene):
        instances: list[trimesh.Trimesh] = []
        for node_name in mesh.graph.nodes_geometry:
            transform, geometry_name = mesh.graph[node_name]
            geometry = mesh.geometry[geometry_name]
            if not isinstance(geometry, trimesh.Trimesh):
                raise ValueError(
                    f"scene geometry {geometry_name!r} is not a triangle mesh"
                )
            instance = geometry.copy()
            instance.apply_transform(transform)
            instances.append(instance)
        if not instances:
            raise ValueError("mesh scene contains no triangle geometry")
        mesh = trimesh.util.concatenate(instances)
    elif not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(
            f"mesh must be trimesh.Trimesh or trimesh.Scene, got {type(mesh).__name__}"
        )

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) == 0:
        raise ValueError("mesh must contain vertices shaped [V, 3]")
    if faces.ndim != 2 or faces.shape[1:] != (3,) or len(faces) == 0:
        raise ValueError("mesh must contain triangular faces shaped [F, 3]")
    if not np.isfinite(vertices).all():
        raise ValueError("mesh vertices must all be finite")
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError("mesh faces contain out-of-range vertex indices")
    if float(np.max(np.ptp(vertices, axis=0))) == 0.0:
        raise ValueError("mesh must have non-zero extent")

    return trimesh.Trimesh(
        vertices=vertices.copy(),
        faces=faces.astype(np.int64, copy=True),
        process=False,
    )


def postprocess_mesh(
    mesh: trimesh.Trimesh | trimesh.Scene,
    *,
    target_face_count: int,
    component_face_ratio: float = COMPONENT_FACE_RATIO,
) -> trimesh.Trimesh:
    """Filter, simplify, center, and unit-scale mesh geometry.

    Args:
        mesh: Triangle mesh or transformed scene to process.
        target_face_count: Caller-selected pyfqmr face target.
        component_face_ratio: Minimum total-face ratio retained for components
            other than the largest.

    Returns:
        Material-free mesh centered by vertex mean with unit maximum radial norm.

    Raises:
        ValueError: A parameter or intermediate mesh is invalid.
    """
    if (
        isinstance(target_face_count, bool)
        or not isinstance(target_face_count, Integral)
        or int(target_face_count) <= 0
    ):
        raise ValueError(
            f"target_face_count must be a positive integer, got {target_face_count!r}"
        )
    if not 0.0 <= component_face_ratio <= 1.0:
        raise ValueError(
            f"component_face_ratio must be in [0, 1], got {component_face_ratio!r}"
        )

    result = validate_mesh(mesh)
    result = _filter_components(result, float(component_face_ratio))
    result = _simplify(result, int(target_face_count))
    result = validate_mesh(result)

    vertices = np.asarray(result.vertices, dtype=np.float64)
    vertices = vertices - vertices.mean(axis=0, keepdims=True)
    max_radius = float(np.linalg.norm(vertices, axis=1).max())
    if not np.isfinite(max_radius) or max_radius <= 0.0:
        raise ValueError("mesh cannot be normalized to a positive finite radius")
    vertices = vertices / max_radius

    return trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(result.faces, dtype=np.int64).copy(),
        process=False,
    )


def _filter_components(
    mesh: trimesh.Trimesh, component_face_ratio: float
) -> trimesh.Trimesh:
    face_count = len(mesh.faces)
    components = trimesh.graph.connected_components(
        mesh.face_adjacency,
        nodes=np.arange(face_count, dtype=np.int64),
        min_len=1,
    )
    if not components:
        raise ValueError("mesh contains no connected face components")

    largest = max(components, key=len)
    kept = [
        np.asarray(component, dtype=np.int64)
        for component in components
        if component is largest
        or len(component) / face_count >= component_face_ratio
    ]
    selected_faces = np.sort(np.concatenate(kept))
    faces = np.asarray(mesh.faces, dtype=np.int64)[selected_faces]
    vertex_indices, inverse = np.unique(faces.reshape(-1), return_inverse=True)
    compact_faces = inverse.reshape(-1, 3)
    compact_vertices = np.asarray(mesh.vertices, dtype=np.float64)[vertex_indices]
    return trimesh.Trimesh(
        vertices=compact_vertices,
        faces=compact_faces,
        process=False,
    )


def _simplify(mesh: trimesh.Trimesh, target_face_count: int) -> trimesh.Trimesh:
    try:
        import pyfqmr
    except ImportError as exc:
        raise ImportError(
            "Mesh simplification requires the core `pyfqmr` dependency."
        ) from exc

    simplifier = pyfqmr.Simplify()
    simplifier.setMesh(mesh.vertices, mesh.faces)
    simplifier.simplify_mesh(
        target_count=target_face_count,
        aggressiveness=5,
        preserve_border=True,
        verbose=False,
    )
    vertices, faces, _ = simplifier.getMesh()
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
