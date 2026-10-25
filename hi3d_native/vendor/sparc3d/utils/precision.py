"""Autocast-proof precision control for modules that must run in float32."""

# Adapted from Pointcept / Point Transformer V3 under the MIT License.
# Modified for SPARC3D: an explicit mode enum in place of the upstream flags.

from __future__ import annotations

import functools
from enum import Enum
from typing import Callable, TypeVar

import torch

_F = TypeVar("_F", bound=Callable)


class PrecisionMode(Enum):
    """How :func:`force_fp32` reconciles an input's dtype with float32."""

    ENFORCE = "enforce"
    """Raise unless the input already has the target dtype."""

    CONVERT = "convert"
    """Convert the input and leave the result in the target dtype."""

    CONVERT_BACK = "convert_back"
    """Convert the input, then restore the original dtype on the result."""


class DtypeMismatchError(Exception):
    """Raised when a module is handed a point in an unexpected dtype."""


def force_fp32(mode: str | PrecisionMode = "convert") -> Callable[[_F], _F]:
    """Run a ``Point``-taking forward in float32 with autocast disabled.

    Args:
        mode: One of the :class:`PrecisionMode` values, by name or by member.

    Returns:
        Decorator for a ``forward(self, point, *args, **kwargs)`` method.

    Raises:
        ValueError: ``mode`` is not a known precision mode.
    """
    if isinstance(mode, str):
        try:
            mode = PrecisionMode(mode)
        except ValueError:
            valid_modes = [candidate.value for candidate in PrecisionMode]
            raise ValueError(f"Invalid mode: {mode}. Must be one of {valid_modes}")

    def decorator(func):

        @functools.wraps(func)
        def wrapper(self, point, *args, **kwargs):
            module_name = self.__class__.__name__
            original_dtype = None
            converted = False

            if mode == PrecisionMode.ENFORCE:
                if point.dtype != torch.float32:
                    raise DtypeMismatchError(
                        f"{module_name} requires point.dtype to be float32, "
                        f"but got {point.dtype}."
                    )
            elif mode == PrecisionMode.CONVERT:
                if point.dtype != torch.float32:
                    point = point.float()
                    converted = True
            elif mode == PrecisionMode.CONVERT_BACK:
                if point.dtype != torch.float32:
                    original_dtype = point.dtype
                    point = point.float()
                    converted = True

            with torch.amp.autocast(device_type="cuda", enabled=False):
                result = func(self, point, *args, **kwargs)

            if (
                mode == PrecisionMode.CONVERT_BACK
                and converted
                and original_dtype is not None
            ):
                result = result.to(original_dtype)

            return result

        return wrapper

    return decorator
