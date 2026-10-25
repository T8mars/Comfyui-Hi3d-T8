"""Validated configuration for the released sparse latent transformer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping


_LATENT_MEAN = (
    -0.00155994,
    0.03314649,
    -0.00307796,
    0.00630751,
    -0.00835457,
    -0.03860179,
    0.02528263,
    0.03740620,
    -0.06209833,
    -0.00539356,
    -0.02110271,
    0.05126001,
    0.00753371,
    0.02866821,
    0.02772475,
    -0.02131735,
    -0.02799912,
    0.01084617,
    0.03433723,
    -0.00903100,
    -0.00121899,
    -0.01944244,
    0.01641873,
    -0.01063657,
    -0.01990618,
    0.04856062,
    0.01747232,
    -0.02387775,
    0.03216169,
    -0.03732105,
    -0.03730354,
    -0.01948937,
)
"""Per-channel mean of the released structured latent."""

_LATENT_STD = (
    0.61686146,
    0.80115320,
    0.85489200,
    0.96384716,
    0.77005910,
    0.62859850,
    0.61541015,
    0.74368310,
    0.62762610,
    1.01842280,
    0.77113867,
    0.62627990,
    0.68749195,
    0.62698966,
    0.75491240,
    0.61577400,
    0.63780660,
    0.80406000,
    0.83846325,
    0.69120390,
    0.69227580,
    0.79321617,
    0.63475930,
    0.63848300,
    0.67194235,
    0.73519380,
    0.84781533,
    0.93012910,
    0.62691440,
    0.67906080,
    0.65198770,
    0.77692040,
)
"""Per-channel standard deviation of the released structured latent."""


@dataclass(frozen=True)
class Sparc3DTransformerConfig:
    """Narrow architecture configuration for :class:`Sparc3DTransformer`.

    Only the dimensions that actually vary between the two released checkpoints are
    configurable. The architecture choices the weights were trained under -- rotary
    position encoding, full flash attention, self-attention QK RMS normalization,
    shared timestep modulation, and I/O skip connections -- are fixed in the model.
    """

    resolution: int = 64
    """Structure-grid resolution represented by input coordinates."""

    in_channels: int = 32
    """Input latent feature channels."""

    out_channels: int = 32
    """Predicted latent feature channels."""

    model_channels: int = 2112
    """Transformer hidden width."""

    cond_channels: int = 2048
    """Image-conditioning width."""

    num_blocks: int = 24
    """Number of shared-modulation transformer blocks."""

    num_heads: int = 22
    """Number of attention heads."""

    mlp_ratio: float = 4.0
    """Expansion ratio of each feed-forward layer."""

    patch_size: int = 2
    """Sparse I/O downsampling factor."""

    num_io_res_blocks: int = 2
    """Residual blocks in each I/O stage."""

    io_block_channels: tuple[int, ...] = (256,)
    """Channels in each sparse I/O stage."""

    use_fp16: bool = True
    """Whether the transformer torso uses float16 parameters."""

    latent_mean: tuple[float, ...] = _LATENT_MEAN
    """Per-channel latent mean used by pipeline normalization."""

    latent_std: tuple[float, ...] = _LATENT_STD
    """Per-channel latent standard deviation used by pipeline normalization."""

    def __post_init__(self) -> None:
        positive = {
            "resolution": self.resolution,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "model_channels": self.model_channels,
            "cond_channels": self.cond_channels,
            "num_blocks": self.num_blocks,
            "num_heads": self.num_heads,
            "patch_size": self.patch_size,
            "num_io_res_blocks": self.num_io_res_blocks,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"positive dimensions required for {invalid}")
        if self.model_channels % self.num_heads:
            raise ValueError("model_channels must be divisible by num_heads")
        head_dim = self.model_channels // self.num_heads
        if head_dim % 2 or head_dim < 6:
            raise ValueError("attention head dimension must be even and at least 6")
        if self.patch_size & (self.patch_size - 1):
            raise ValueError("patch_size must be a power of two")
        if self.patch_size != 2 ** len(self.io_block_channels):
            raise ValueError("I/O stage count must equal log2(patch_size)")
        if any(channels <= 1 for channels in self.io_block_channels):
            raise ValueError("I/O stage channels must be greater than one")
        if len(self.latent_mean) != self.in_channels:
            raise ValueError("latent_mean length must match in_channels")
        if len(self.latent_std) != self.in_channels:
            raise ValueError("latent_std length must match in_channels")
        if any(value <= 0 for value in self.latent_std):
            raise ValueError("latent_std values must be positive")

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "Sparc3DTransformerConfig":
        """Build a config from an allowlisted mapping.

        Args:
            values: Parsed transformer configuration.

        Returns:
            Validated config.

        Raises:
            ValueError: Unknown keys are present.
        """
        values = dict(values)
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"unknown transformer config keys: {unknown}")
        for key in ("io_block_channels", "latent_mean", "latent_std"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible configuration mapping."""
        values = asdict(self)
        for key in ("io_block_channels", "latent_mean", "latent_std"):
            values[key] = list(values[key])
        return values
