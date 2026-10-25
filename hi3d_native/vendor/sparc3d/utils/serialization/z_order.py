# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see THIRD_PARTY_LICENSES.md for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------
# Modified for SPARC3D to retain encoding only: the decode tables and the
# batch-index field of the key are removed.
# --------------------------------------------------------
"""Z-order (Morton) encoding of integer grid coordinates."""

import torch


class KeyLUT:
    """Per-device lookup tables mapping each byte of a coordinate to its Morton bits."""

    def __init__(self):
        r256 = torch.arange(256, dtype=torch.int64)
        zero = torch.zeros(256, dtype=torch.int64)
        device = torch.device("cpu")

        self._encode = {
            device: (
                self.xyz2key(r256, zero, zero, 8),
                self.xyz2key(zero, r256, zero, 8),
                self.xyz2key(zero, zero, r256, 8),
            )
        }

    def encode_lut(self, device=torch.device("cpu")):
        """Return the ``(x, y, z)`` tables for ``device``, copying them there once."""
        if device not in self._encode:
            cpu = torch.device("cpu")
            self._encode[device] = tuple(
                e.to(device) for e in self._encode[cpu])
        return self._encode[device]

    def xyz2key(self, x, y, z, depth):
        """Interleave the low ``depth`` bits of each coordinate, one bit at a time.

        Only used to build the tables; :func:`xyz2key` below is the fast path.
        """
        key = torch.zeros_like(x)
        for i in range(depth):
            mask = 1 << i
            key = (key
                   | ((x & mask) << (2 * i + 2))
                   | ((y & mask) << (2 * i + 1))
                   | ((z & mask) << (2 * i + 0)))
        return key


_key_lut = KeyLUT()


def xyz2key(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    depth: int = 16,
):
    r"""Encodes :attr:`x`, :attr:`y`, :attr:`z` coordinates to the shuffled keys
    based on pre-computed look up tables. The speed of this function is much
    faster than the method based on for-loop.

    Args:
      x (torch.Tensor): The x coordinate.
      y (torch.Tensor): The y coordinate.
      z (torch.Tensor): The z coordinate.
      depth (int): The depth of the shuffled key, and must be smaller than 17 (< 17).
    """

    EX, EY, EZ = _key_lut.encode_lut(x.device)
    x, y, z = x.long(), y.long(), z.long()

    mask = 255 if depth > 8 else (1 << depth) - 1
    key = EX[x & mask] | EY[y & mask] | EZ[z & mask]
    if depth > 8:
        mask = (1 << (depth - 8)) - 1
        key16 = EX[(x >> 8) & mask] | EY[(y >> 8) & mask] | EZ[(z >> 8) & mask]
        key = key16 << 24 | key

    return key
