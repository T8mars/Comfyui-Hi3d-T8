"""Lazy public package for the SPARC3D pipeline, output, and progress types."""

from __future__ import annotations

from typing import Any

__version__ = "0.1.0"

__all__ = ["Sparc3DPipeline", "Sparc3DPipelineOutput", "Sparc3DProgressEvent"]


def __getattr__(name: str) -> Any:
    """Load one of the public symbols on first access."""
    if name == "Sparc3DPipeline":
        from sparc3d.pipelines.sparc3d_pipeline import Sparc3DPipeline

        return Sparc3DPipeline
    if name == "Sparc3DPipelineOutput":
        from sparc3d.pipelines.output import Sparc3DPipelineOutput

        return Sparc3DPipelineOutput
    if name == "Sparc3DProgressEvent":
        from sparc3d.pipelines.progress import Sparc3DProgressEvent

        return Sparc3DProgressEvent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return module names including the lazy public symbols."""
    return sorted({*globals(), *__all__})
