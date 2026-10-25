"""Explicit JSON-to-module construction for the sparse mesh autoencoder."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from sparc3d.models.autoencoders.sparconv_vae import SparconvVAE
from sparc3d.models.autoencoders.ptv3 import PTv3MeshDecoder
from sparc3d.structures.point import Point

_SUPPORTED_TYPES: dict[str, type] = {
    "SparconvVAE": SparconvVAE,
    "PTv3MeshDecoder": PTv3MeshDecoder,
}
"""The only ``type`` discriminators a runtime config may name."""

_SUB_CONFIG_TYPES: dict[str, str] = {"decoder_cfg": "PTv3MeshDecoder"}
"""The ``type`` each ``SparconvVAE`` sub-config must name."""


def load_config(config: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Read a runtime config and return its model block.

    Args:
        config: Path to a JSON file, or an already-parsed mapping. Either the whole
            document (with a ``model`` key) or the model block itself is accepted.

    Returns:
        The model block as a plain dict.

    Raises:
        ValueError: The document has no model block carrying a ``type``.
    """
    if isinstance(config, (str, Path)):
        with Path(config).open("r", encoding="utf-8") as f:
            document = json.load(f)
    else:
        document = dict(config)

    model = document.get("model", document)
    if not isinstance(model, Mapping) or "type" not in model:
        raise ValueError("config has no 'model' block with a 'type' discriminator")
    return dict(model)


def build_sparconv_vae(config: str | Path | Mapping[str, Any]) -> SparconvVAE:
    """Instantiate a :class:`SparconvVAE` from a runtime config, without loading weights.

    Args:
        config: Runtime config, as accepted by :func:`load_config`.

    Returns:
        Model on the CPU in eval mode.

    Raises:
        ValueError: An unsupported ``type``, a missing sub-config, or an unknown
            constructor key.
    """
    model_cfg = load_config(config)
    _resolve_type(model_cfg, "model", "SparconvVAE")
    # Training sections (losses, optimizer, scheduler) are rejected rather than
    # ignored, so a training config can never be mistaken for a runtime artifact.
    extra = sorted(set(model_cfg) - set(_SUB_CONFIG_TYPES))
    if extra:
        raise ValueError(f"model config has non-architecture keys: {extra}")

    expected_attributes = sum(
        Point.ALL_LAYOUTS[key]["size"] for key in ("sdf", "deform")
    )
    if "decoder_cfg" not in model_cfg:
        raise ValueError("model config is missing 'decoder_cfg'")
    sub_cfg = dict(model_cfg["decoder_cfg"])
    cls = _resolve_type(sub_cfg, "decoder_cfg", "PTv3MeshDecoder")
    kwargs = _validated_kwargs(cls, sub_cfg, "decoder_cfg")
    output_channels = kwargs.get(
        "out_channels", inspect.signature(cls).parameters["out_channels"].default
    )
    if output_channels != expected_attributes:
        raise ValueError(
            f"decoder out_channels must be {expected_attributes}, "
            f"got {output_channels}"
        )

    model = SparconvVAE(decoder=cls(**kwargs))
    model.eval()
    return model


def load_sparconv_vae(
    weights: str | Path,
    config: str | Path | Mapping[str, Any] | None = None,
    *,
    device: str | torch.device = "cuda",
) -> SparconvVAE:
    """Build a :class:`SparconvVAE` and load its weights, requiring an exact key match.

    Args:
        weights: Path to a ``.safetensors`` file, or to a directory holding
            ``model.safetensors`` and ``config.json``.
        config: Runtime config; defaults to ``config.json`` beside the weights.
        device: Device to move the model to.

    Returns:
        Model in eval mode on ``device``.

    Raises:
        FileNotFoundError: The weights, or the implied ``config.json``, are missing.
        RuntimeError: The checkpoint keys do not match the model's parameters exactly.
    """
    from safetensors.torch import load_file

    weights_path = Path(weights)
    if weights_path.is_dir():
        weights_path = weights_path / "model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(f"no weights at {weights_path}")

    if config is None:
        config = weights_path.parent / "config.json"
        if not Path(config).is_file():
            raise FileNotFoundError(
                f"no config given and no config.json beside {weights_path}"
            )

    model = build_sparconv_vae(config)
    model.load_state_dict(load_file(str(weights_path)), strict=True)
    return model.to(device).eval()


def _resolve_type(cfg: dict[str, Any], where: str, expected: str) -> type:
    """Pop the ``type`` discriminator and resolve it to its class."""
    type_name = cfg.pop("type", None)
    if type_name != expected:
        raise ValueError(
            f"{where}: unsupported type {type_name!r}; expected {expected!r}"
        )
    return _SUPPORTED_TYPES[type_name]


def _validated_kwargs(cls: type, cfg: dict[str, Any], where: str) -> dict[str, Any]:
    """Reject config keys that are not constructor parameters of ``cls``."""
    accepted = set(inspect.signature(cls).parameters)
    unknown = sorted(set(cfg) - accepted)
    if unknown:
        raise ValueError(f"{where}: unknown keys for {cls.__name__}: {unknown}")
    return cfg
