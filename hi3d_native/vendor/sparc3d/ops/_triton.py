"""Lazy loader for the Triton operation backend."""

from __future__ import annotations

from types import ModuleType


def load_triton_ops() -> ModuleType:
    """Import and return ``sparc3d.ops.triton``, or raise a clear ``ImportError``.

    Raises:
        ImportError: Triton is unavailable in the running environment.
    """
    try:
        from sparc3d.ops import triton as triton_ops
    except ImportError as exc:
        raise ImportError(
            "The SPARC3D operations require Triton. Install the verified Linux "
            "CUDA PyTorch stack, which supplies it."
        ) from exc
    return triton_ops
