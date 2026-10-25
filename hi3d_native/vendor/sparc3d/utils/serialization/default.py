"""Z-order serialization of grid coordinates into sortable integer codes."""

# Adapted from Pointcept / Point Transformer V3 under the MIT License.
# Modified for SPARC3D to retain the z-order curve only; the Hilbert order and
# the decode direction are removed.

import torch

from .z_order import xyz2key as z_order_encode_


@torch.inference_mode()
def encode(grid_coord, batch=None, depth=16, order="z"):
    """Encode grid coordinates as z-order codes, optionally keyed by batch.

    Args:
        grid_coord: Integer coordinates ``[N, 3]``.
        batch: Optional batch index per point, packed above the coordinate bits.
        depth: Bits per axis.
        order: ``"z"``, or ``"z-trans"`` to swap the first two axes first.

    Returns:
        One int64 code per point, sortable along the chosen curve.

    Raises:
        ValueError: ``order`` is not a supported curve.
    """
    if order not in {"z", "z-trans"}:
        raise ValueError(f"unsupported serialization order: {order!r}")
    if order == "z":
        code = z_order_encode(grid_coord, depth=depth)
    else:
        code = z_order_encode(grid_coord[:, [1, 0, 2]], depth=depth)
    if batch is not None:
        batch = batch.long()
        code = batch << depth * 3 | code
    return code


def z_order_encode(grid_coord: torch.Tensor, depth: int = 16):
    """Interleave the bits of each coordinate triple into one z-order code."""
    x, y, z = grid_coord[:, 0].long(), grid_coord[:, 1].long(), grid_coord[:, 2].long()
    # Batching is handled by the caller so Point owns the batched code layout.
    return z_order_encode_(x, y, z, depth=depth)
