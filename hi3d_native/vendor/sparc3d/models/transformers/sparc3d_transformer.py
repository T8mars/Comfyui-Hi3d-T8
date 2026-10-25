"""Sparse latent flow transformer for image-conditioned mesh refinement."""

# Adapted from Microsoft TRELLIS under the MIT License.
# Modified for SPARC3D: the released sparse inference architecture.

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from sparc3d.nn import LayerNorm32
from sparc3d.nn.sparse import (
    SparseConv3d,
    SparseDownsample,
    SparseLinear,
    SparseTensor,
    SparseUpsample,
)
from sparc3d.nn.sparse.transformer import ModulatedSparseTransformerCrossBlock
from sparc3d.nn.utils import convert_module_to_f16, zero_module

from .config import Sparc3DTransformerConfig


class TimestepEmbedder(nn.Module):
    """Sinusoidal scalar timestep embedding followed by a two-layer MLP."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        """
        Args:
            hidden_size: Width of the embedding the MLP produces.
            frequency_embedding_size: Sinusoidal features formed before the MLP.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(
        timestep: torch.Tensor, dim: int, max_period: int = 10000
    ) -> torch.Tensor:
        """Create sinusoidal embeddings for scalar timesteps.

        Args:
            timestep: Fractional or integral timesteps ``[B]``.
            dim: Embedding width.
            max_period: Minimum-frequency denominator.

        Returns:
            Embeddings ``[B, dim]``.
        """
        half = dim // 2
        frequencies = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32) / half
        ).to(timestep.device)
        arguments = timestep[:, None].float() * frequencies[None]
        embedding = torch.cat((torch.cos(arguments), torch.sin(arguments)), dim=-1)
        if dim % 2:
            embedding = torch.cat(
                (embedding, torch.zeros_like(embedding[:, :1])), dim=-1
            )
        return embedding

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        """Embed ``timestep`` into the transformer hidden width."""
        frequencies = self.timestep_embedding(timestep, self.frequency_embedding_size)
        return self.mlp(frequencies)


class SparseResBlock3d(nn.Module):
    """Timestep-modulated sparse residual block with optional paired resampling."""

    def __init__(
        self,
        channels: int,
        emb_channels: int,
        out_channels: int | None = None,
        *,
        downsample: bool = False,
        upsample: bool = False,
    ):
        """
        Args:
            channels: Input feature channels.
            emb_channels: Timestep embedding channels consumed by the modulation.
            out_channels: Output channels; defaults to ``channels``.
            downsample: Halve the spatial resolution across the block.
            upsample: Double the spatial resolution across the block.

        Raises:
            ValueError: Both ``downsample`` and ``upsample`` are set.
        """
        super().__init__()
        if downsample and upsample:
            raise ValueError("cannot downsample and upsample in one block")
        self.channels = channels
        self.emb_channels = emb_channels
        self.out_channels = out_channels or channels
        self.hidden_channels = self.out_channels // 2
        self.downsample = downsample
        self.upsample = upsample
        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(
            self.hidden_channels, elementwise_affine=False, eps=1e-6
        )
        self.conv1 = SparseConv3d(channels, self.hidden_channels, 3)
        self.conv2 = SparseConv3d(self.hidden_channels, self.hidden_channels, 3)
        self.linear = zero_module(SparseLinear(self.hidden_channels, self.out_channels))
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, 2 * self.hidden_channels, bias=True),
        )
        self.skip_connection = (
            SparseLinear(channels, self.out_channels)
            if channels != self.out_channels
            else nn.Identity()
        )
        self.updown: nn.Module | None = None
        if downsample:
            self.updown = SparseDownsample(2)
        elif upsample:
            self.updown = SparseUpsample(2)

    def forward(self, x: SparseTensor, embedding: torch.Tensor) -> SparseTensor:
        """Apply residual convolution and optional paired resampling."""
        scale, shift = self.emb_layers(embedding).to(x.dtype).chunk(2, dim=1)
        if self.updown is not None:
            x = self.updown(x)
        h = x.replace(self.norm1(x.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv1(h)
        h = h.replace(self.norm2(h.feats)) * (1 + scale) + shift
        h = h.replace(F.silu(h.feats))
        h = self.conv2(h)
        h = h.replace(F.silu(h.feats))
        return self.linear(h) + self.skip_connection(x)


class Sparc3DTransformer(nn.Module):
    """Released sparse latent flow transformer with a validated narrow config."""

    def __init__(
        self,
        config: Sparc3DTransformerConfig | None = None,
        **config_values: Any,
    ):
        """
        Args:
            config: Validated configuration. Mutually exclusive with ``config_values``.
            **config_values: Configuration fields to validate into a config instead.

        Raises:
            ValueError: Both a config and configuration values were given, or a value
                fails validation.
        """
        super().__init__()
        if config is not None and config_values:
            raise ValueError("pass either config or configuration keyword values")
        self.config = config or Sparc3DTransformerConfig.from_dict(config_values)
        config = self.config
        self.dtype = torch.float16 if config.use_fp16 else torch.float32

        channels = list(config.io_block_channels)
        self.t_embedder = TimestepEmbedder(config.model_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.model_channels, 6 * config.model_channels, bias=True),
        )
        self.input_layer = SparseLinear(config.in_channels, channels[0])
        self.input_blocks = nn.ModuleList()
        for current, following in zip(channels, channels[1:] + [config.model_channels]):
            self.input_blocks.extend(
                SparseResBlock3d(current, config.model_channels)
                for _ in range(config.num_io_res_blocks - 1)
            )
            self.input_blocks.append(
                SparseResBlock3d(
                    current,
                    config.model_channels,
                    out_channels=following,
                    downsample=True,
                )
            )

        self.blocks = nn.ModuleList(
            ModulatedSparseTransformerCrossBlock(
                config.model_channels,
                config.cond_channels,
                config.num_heads,
                mlp_ratio=config.mlp_ratio,
            )
            for _ in range(config.num_blocks)
        )

        self.out_blocks = nn.ModuleList()
        previous_channels = [config.model_channels] + list(reversed(channels[1:]))
        for current, previous in zip(reversed(channels), previous_channels):
            self.out_blocks.append(
                SparseResBlock3d(
                    previous * 2,
                    config.model_channels,
                    out_channels=current,
                    upsample=True,
                )
            )
            self.out_blocks.extend(
                SparseResBlock3d(
                    current * 2,
                    config.model_channels,
                    out_channels=current,
                )
                for _ in range(config.num_io_res_blocks - 1)
            )
        self.out_layer = SparseLinear(channels[0], config.out_channels)

        self.initialize_weights()
        if config.use_fp16:
            self.convert_to_fp16()

    def initialize_weights(self) -> None:
        """Initialize linear, timestep, modulation, and output projections."""

        def initialize_linear(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(initialize_linear)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    def convert_to_fp16(self) -> None:
        """Convert the I/O and transformer torso to float16."""
        self.input_blocks.apply(convert_module_to_f16)
        self.blocks.apply(convert_module_to_f16)
        self.out_blocks.apply(convert_module_to_f16)
        self.dtype = torch.float16

    def forward(
        self, x: SparseTensor, timestep: torch.Tensor, cond: torch.Tensor
    ) -> SparseTensor:
        """Predict latent flow for sparse ``x`` under image ``cond``."""
        if x.shape[0] != timestep.shape[0] or x.shape[0] != cond.shape[0]:
            raise ValueError("sparse input, timestep, and condition batches must match")
        if x.feats.shape[1] != self.config.in_channels:
            raise ValueError(
                f"expected {self.config.in_channels} input channels, "
                f"got {x.feats.shape[1]}"
            )
        h = self.input_layer(x).type(self.dtype)
        timestep_embedding = self.t_embedder(timestep)
        convolution_embedding = timestep_embedding.to(self.dtype)
        skips = []
        for block in self.input_blocks:
            h = block(h, convolution_embedding)
            skips.append(h.feats)

        modulation = self.adaLN_modulation(timestep_embedding).to(self.dtype)
        context = cond.to(self.dtype)
        for block in self.blocks:
            h = block(h, modulation, context)

        for block, skip in zip(self.out_blocks, reversed(skips)):
            h = h.replace(torch.cat((h.feats, skip), dim=1))
            h = block(h, convolution_embedding)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        return self.out_layer(h.type(x.dtype))
