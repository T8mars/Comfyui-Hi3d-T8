"""Shared DINOv2 and CLIP image conditioner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn
from transformers import AutoModel, CLIPVisionModel

DINO_MODEL_ID = "facebook/dinov2-large"
"""Moving Hugging Face repository used for DINOv2 conditioning."""

CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
"""Moving Hugging Face repository used for CLIP conditioning."""

_DINO_SIZE = 756
_CLIP_SIZE = 224


class ImageConditioner(nn.Module):
    """Frozen fp32 image conditioner combining DINOv2 tokens and CLIP context."""

    def __init__(
        self,
        *,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        dino_model_id: str | Path = DINO_MODEL_ID,
        clip_model_id: str | Path = CLIP_MODEL_ID,
    ) -> None:
        """Load and freeze the shared Hugging Face vision encoders.

        Args:
            cache_dir: Optional Hugging Face cache directory. This is the directory
                that holds the ``models--*`` trees, i.e. ``$HF_HOME/hub`` rather
                than ``$HF_HOME``.
            local_files_only: Whether encoder loading must use cached files only.
            dino_model_id: DINOv2 repository ID, or a local directory holding the
                encoder. A directory needs no Hub cache layout, which is the
                simplest way to run without network access.
            clip_model_id: CLIP repository ID, or a local directory.
        """
        super().__init__()
        load_kwargs: dict[str, Any] = {}
        if cache_dir is not None:
            load_kwargs["cache_dir"] = str(cache_dir)
        if local_files_only:
            load_kwargs["local_files_only"] = True
        self.dino = AutoModel.from_pretrained(str(dino_model_id), **load_kwargs).float()
        self.clip = CLIPVisionModel.from_pretrained(
            str(clip_model_id), **load_kwargs
        ).float()

        for parameter in self.parameters():
            parameter.requires_grad_(False)

        self.register_buffer(
            "_dino_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_dino_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_clip_mean",
            torch.tensor(
                [0.48145466, 0.4578275, 0.40821073], dtype=torch.float32
            ).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_clip_std",
            torch.tensor(
                [0.26862954, 0.26130258, 0.27577711], dtype=torch.float32
            ).view(1, 3, 1, 1),
            persistent=False,
        )
        self.eval()

    def train(self, mode: bool = True) -> "ImageConditioner":
        """Keep the conditioner and both frozen encoders in evaluation mode."""
        del mode
        return super().train(False)

    @torch.no_grad()
    def forward(self, image: Image.Image) -> Tensor:
        """Encode one prepared PIL image into batch-1 conditioning tokens.

        Args:
            image: Prepared RGB PIL image. Only a single image is accepted.

        Returns:
            L2-normalized DINOv2 tokens concatenated with a broadcast CLIP
            embedding, shaped ``[1, 2917, 2048]``.

        Raises:
            TypeError: ``image`` is not a PIL image.
            ValueError: The encoders return an unexpected token or channel shape.
        """
        if not isinstance(image, Image.Image):
            raise TypeError(
                f"image must be one prepared PIL image, got {type(image).__name__}"
            )

        device = self._dino_mean.device
        base_input = self._pil_tensor(image, device)
        dino_input = F.interpolate(
            base_input,
            size=(_DINO_SIZE, _DINO_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        clip_input = self._image_tensor(
            image, _CLIP_SIZE, Image.Resampling.LANCZOS, device
        )
        dino_input = (dino_input - self._dino_mean) / self._dino_std
        clip_input = (clip_input - self._clip_mean) / self._clip_std

        # Disable an enclosing mixed-precision context because conditioner numerics
        # are fixed to fp32 independently of transformer inference precision.
        with torch.autocast(device_type=device.type, enabled=False):
            dino_output: Any = self.dino(pixel_values=dino_input)
            clip_output: Any = self.clip(pixel_values=clip_input)

        dino_tokens = F.normalize(
            dino_output.last_hidden_state.float(), p=2, dim=-1
        )
        clip_embedding = F.normalize(clip_output.pooler_output.float(), p=2, dim=-1)

        if dino_tokens.shape != (1, 2917, 1024):
            raise ValueError(
                "DINOv2 output must have shape [1, 2917, 1024], "
                f"got {list(dino_tokens.shape)}"
            )
        if clip_embedding.shape != (1, 1024):
            raise ValueError(
                "CLIP output must have shape [1, 1024], "
                f"got {list(clip_embedding.shape)}"
            )

        clip_tokens = clip_embedding[:, None, :].expand(
            -1, dino_tokens.shape[1], -1
        )
        return torch.cat((dino_tokens, clip_tokens), dim=-1)

    @staticmethod
    def _image_tensor(
        image: Image.Image,
        size: int,
        resample: Image.Resampling,
        device: torch.device,
    ) -> Tensor:
        rgb = image.convert("RGB").resize((size, size), resample=resample)
        return ImageConditioner._pil_tensor(rgb, device)

    @staticmethod
    def _pil_tensor(image: Image.Image, device: torch.device) -> Tensor:
        rgb = image.convert("RGB")
        array = np.asarray(rgb, dtype=np.float32)
        tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(device=device, dtype=torch.float32).div_(255.0)
