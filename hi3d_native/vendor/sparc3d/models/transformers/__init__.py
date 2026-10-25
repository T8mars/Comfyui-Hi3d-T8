"""Sparse latent transformer and its validated runtime configuration."""

from .config import Sparc3DTransformerConfig
from .sparc3d_transformer import Sparc3DTransformer

__all__ = ["Sparc3DTransformer", "Sparc3DTransformerConfig"]
