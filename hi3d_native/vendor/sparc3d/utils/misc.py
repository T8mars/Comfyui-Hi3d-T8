"""Batch/offset conversions and structure cloning helpers."""

# Adapted from Pointcept / Point Transformer V3 under the MIT License.
# Modified for SPARC3D to retain the batch/offset trio used by the decoder.

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch


@torch.inference_mode()
def offset2bincount(offset: torch.Tensor) -> torch.Tensor:
    """Return the per-batch point counts implied by a cumulative ``offset``."""
    return torch.diff(
        offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long)
    )


@torch.inference_mode()
def offset2batch(offset: torch.Tensor) -> torch.Tensor:
    """Expand a cumulative ``offset`` into a per-point batch index."""
    bincount = offset2bincount(offset)
    return torch.arange(
        len(bincount), device=offset.device, dtype=torch.long
    ).repeat_interleave(bincount)


@torch.inference_mode()
def batch2offset(batch: torch.Tensor) -> torch.Tensor:
    """Reduce a per-point batch index into a cumulative ``offset``."""
    return torch.cumsum(batch.bincount(), dim=0).long()


def deep_clone(value: Any) -> Any:
    """Copy a nested container, sharing tensors rather than duplicating them.

    Tensors are passed through by reference: the point structures this backs are
    cloned per network stage and copying multi-million-element features at every
    stage would dominate decoder memory.
    """
    if isinstance(value, dict):
        return {key: deep_clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [deep_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(deep_clone(item) for item in value)
    if isinstance(value, torch.Tensor):
        return value
    return deepcopy(value)
