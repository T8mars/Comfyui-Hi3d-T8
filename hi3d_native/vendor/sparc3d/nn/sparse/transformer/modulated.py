"""Shared-modulation sparse transformer cross-attention block."""

# Adapted from Microsoft TRELLIS under the MIT License.
# Modified for SPARC3D: shared timestep modulation with full attention.

from __future__ import annotations

import torch
from torch import nn

from ...norm import LayerNorm32
from ..attention import SparseMultiHeadAttention
from ..basic import SparseTensor
from .blocks import SparseFeedForwardNet


class ModulatedSparseTransformerCrossBlock(nn.Module):
    """Sparse self/cross-attention block with shared adaptive normalization."""

    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.self_attn = SparseMultiHeadAttention(
            channels,
            num_heads,
            attention_type="self",
        )
        self.cross_attn = SparseMultiHeadAttention(
            channels,
            num_heads,
            ctx_channels=ctx_channels,
            attention_type="cross",
        )
        self.mlp = SparseFeedForwardNet(channels, mlp_ratio)
        self.modulation = nn.Parameter(torch.randn(1, 6 * channels) / channels**0.5)

    def forward(
        self, x: SparseTensor, mod: torch.Tensor, context: torch.Tensor
    ) -> SparseTensor:
        """Transform ``x`` with timestep modulation and image context."""
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = (self.modulation + mod).chunk(6, dim=1)
        h = x.replace(self.norm1(x.feats))
        h = h * (1 + scale_msa) + shift_msa
        h = self.self_attn(h) * gate_msa
        x = x + h
        h = x.replace(self.norm2(x.feats))
        x = x + self.cross_attn(h, context)
        h = x.replace(self.norm3(x.feats))
        h = h * (1 + scale_mlp) + shift_mlp
        return x + self.mlp(h) * gate_mlp
