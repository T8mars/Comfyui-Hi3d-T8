"""Immutable public output returned by the SPARC3D pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import trimesh


@dataclass(frozen=True)
class Sparc3DPipelineOutput:
    """Completed in-memory mesh refinement result."""

    mesh: trimesh.Trimesh
    """Geometry-only ``trimesh.Trimesh`` result."""

    resolution: int
    """Output resolution the mesh was produced at."""
