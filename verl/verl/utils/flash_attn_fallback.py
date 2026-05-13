"""Fallback implementations of flash_attn.bert_padding functions using PyTorch."""

import torch
from einops import rearrange


def index_first_axis(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Select elements along the first axis by indices."""
    return x[indices]


def unpad_input(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
):
    """Remove padding from input sequences.

    Args:
        hidden_states: (batch, seqlen, ...)
        attention_mask: (batch, seqlen), 1=keep, 0=pad

    Returns:
        output: (total_nnz, ...)
        indices: (total_nnz,)
        cu_seqlens: (batch + 1,)
        max_seqlen_in_batch: int
    """
    seqlens = attention_mask.sum(dim=1).int()
    indices = attention_mask.bool().flatten().nonzero(as_tuple=True)[0]
    output = hidden_states.flatten(0, 1)[indices]
    max_seqlen_in_batch = int(seqlens.max().item())
    cu_seqlens = torch.nn.functional.pad(
        seqlens.cumsum(dim=0), pad=(1, 0), value=0
    )
    return output, indices, cu_seqlens, max_seqlen_in_batch


def pad_input(
    hidden_states: torch.Tensor,
    indices: torch.Tensor,
    batch: int,
    seqlen: int,
):
    """Pad hidden_states back to (batch, seqlen, ...) using indices.

    Args:
        hidden_states: (total_nnz, ...)
        indices: (total_nnz,)
        batch: int
        seqlen: int

    Returns:
        output: (batch, seqlen, ...)
    """
    dim = hidden_states.shape[-1]
    output = torch.zeros(
        (batch * seqlen, dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    output[indices] = hidden_states
    return output.view(batch, seqlen, dim)
