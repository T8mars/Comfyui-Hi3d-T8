"""Variable-length full flash attention for sparse feature sequences."""

# Adapted from Microsoft TRELLIS under the MIT License.
# Modified for SPARC3D: variable-length flash attention only.

from __future__ import annotations

import torch

from ..basic import SparseTensor


def _flash_attn():
    """Import flash-attn when sparse attention is first executed."""
    try:
        import flash_attn
    except ImportError as exc:
        raise ImportError(
            "Sparc3DTransformer attention requires flash-attn; "
            "install sparc3d[inference]"
        ) from exc
    return flash_attn


def _lengths(tensor: SparseTensor) -> list[int]:
    return [span.stop - span.start for span in tensor.layout]


def _cumulative(lengths: list[int], device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [0, *lengths], device=device, dtype=torch.int32
    ).cumsum(0, dtype=torch.int32)


def sparse_scaled_dot_product_attention(
    query: SparseTensor, kv: torch.Tensor | None = None
) -> SparseTensor:
    """Apply variable-length full flash attention over sparse queries.

    Args:
        query: Packed sparse ``qkv`` of shape ``[N, 3, H, D]`` for self-attention,
            or sparse queries ``[N, H, D]`` when ``kv`` is given.
        kv: Dense packed keys and values ``[B, L, 2, H, D]`` for cross-attention;
            ``None`` selects self-attention over the packed ``query``.

    Returns:
        Sparse output over the query's coordinates.

    Raises:
        TypeError: ``query`` is not sparse.
        ValueError: ``kv`` does not share the query's batch size.
    """
    flash_attn = _flash_attn()
    if not isinstance(query, SparseTensor):
        raise TypeError("Sparc3D sparse attention requires sparse queries")
    query_lengths = _lengths(query)
    cumulative_query = _cumulative(query_lengths, query.device)

    if kv is None:
        output = flash_attn.flash_attn_varlen_qkvpacked_func(
            query.feats, cumulative_query, max(query_lengths)
        )
        return query.replace(output)

    batch, length = kv.shape[:2]
    if batch != query.shape[0]:
        raise ValueError("query and context batch sizes must match")
    key_lengths = [length] * batch
    output = flash_attn.flash_attn_varlen_kvpacked_func(
        query.feats,
        kv.reshape(batch * length, *kv.shape[2:]),
        cumulative_query,
        _cumulative(key_lengths, query.device),
        max(query_lengths),
        max(key_lengths),
    )
    return query.replace(output)
