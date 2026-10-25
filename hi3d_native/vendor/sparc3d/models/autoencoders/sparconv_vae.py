"""Sparse decoder over voxel-corner mesh attributes (``SparconvVAE``)."""

from __future__ import annotations

from typing import Mapping

from torch import nn

from sparc3d.models.autoencoders.ptv3 import PTv3MeshDecoder
from sparc3d.structures.point import Point


class SparconvVAE(nn.Module):
    """Sparse latent decoder over per-corner mesh attributes.

    Named for the autoencoder the weights were trained as, but this is the decoder
    alone: latents arrive from the diffusion transformer, so no encoder or Gaussian
    bottleneck is reachable at inference and neither is included here.
    """

    def __init__(self, decoder: PTv3MeshDecoder):
        """
        Args:
            decoder: Latent-to-attribute decoder.
        """
        super().__init__()
        self.decoder = decoder

    def decode(self, z_q: Point | Mapping) -> Point:
        """Decode a latent point cloud into activated per-corner attributes.

        Args:
            z_q: Latent point cloud, or a diffusion output mapping with ``grid_coord``, ``feat``, ``resolution``, and a
                batch descriptor.

        Returns:
            Decoded point cloud whose ``feat`` holds activated attributes and whose
            ``sdf_sign`` holds the per-corner sign logits.
        """
        if not isinstance(z_q, Point):
            data = dict(z_q)
            data.setdefault("sdf", None)
            data.setdefault("deform", None)
            z_q = Point(data)
            z_q.initialization()
        elif "layouts" not in z_q or "matrix" not in z_q:
            z_q.initialization()

        if not {"sdf", "deform"}.issubset(z_q.layouts):
            start = 0
            layouts = {}
            for key in ("sdf", "deform"):
                entry = Point.ALL_LAYOUTS[key]
                layouts[key] = {
                    **entry,
                    "range": (start, start + entry["size"]),
                }
                start += entry["size"]
            z_q["layouts"] = layouts

        x_hat = self.decoder(z_q)
        x_hat.activate()
        return x_hat
