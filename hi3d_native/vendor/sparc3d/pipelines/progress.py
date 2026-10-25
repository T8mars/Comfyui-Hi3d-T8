"""Immutable progress events emitted during sparse refinement."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Sparc3DProgressEvent:
    """Progress snapshot for one completed scheduler update."""

    pass_index: int
    """Zero-based pass index in the current call."""

    num_passes: int
    """Total number of refinement passes in the current call."""

    structure_resolution: int
    """Sparse structure resolution of the active pass."""

    step: int
    """One-based number of the completed scheduler update."""

    num_inference_steps: int
    """Total scheduler updates in the active pass."""
