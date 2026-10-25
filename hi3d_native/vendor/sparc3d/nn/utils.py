"""Parameter precision and initialization helpers for neural network modules."""

# Adapted from Microsoft TRELLIS under the MIT License.
# Modified for SPARC3D to retain the sparse-transformer precision closure.

from __future__ import annotations

from torch import nn


def _is_convertible(module: nn.Module) -> bool:
    return isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)) or bool(
        getattr(module, "_sparc3d_convertible", False)
    )


def convert_module_to_f16(module: nn.Module) -> None:
    """Convert primitive module parameters to float16."""
    if _is_convertible(module):
        recurse = bool(getattr(module, "_sparc3d_convertible", False))
        for parameter in module.parameters(recurse=recurse):
            parameter.data = parameter.data.half()


def zero_module(module: nn.Module) -> nn.Module:
    """Zero ``module`` parameters and return the module."""
    for parameter in module.parameters():
        parameter.detach().zero_()
    return module
