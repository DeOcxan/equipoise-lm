"""Causal attention of the text models: rotary positions and the two layouts of the causal EBT.

The causal EBT reads P prefix tokens, S observed tokens and S predictions (method.md sections
3.1 and 18.3). The prefix is the EBT's step-index token (P = 1); the baseline has none (P = 0).
The observed stream is the prefix followed by the observed tokens; each of its positions
attends causally within it. The prediction at position j (the candidate for the character
after observed token j) attends to the prefix, to observed tokens 0, ..., j and to itself only.
Two layouts compute this pattern:

- `two_stream_mask`: one sequence of length P + 2S with a square mask (§C.4);
- `prediction_attention`: the prediction-to-observed-stream scores and each prediction's own
  score as one extra column, without the square score matrix (§C.3).

Both use the explicit attention of `ebt.models.attention`, which can be differentiated twice.
"""

import math

import torch
from torch import Tensor

ROTARY_BASE = 10_000.0


def rotary(x: Tensor, positions: Tensor) -> Tensor:
    """Rotate each pair (x_2i, x_2i+1) of the last dimension by the angle p · base^(−2i / d).

    `x` has shape (..., T, d) with d even; `positions` has shape (T,). The score between a
    rotated query at position a and a rotated key at position b depends on a − b only.
    """
    width = x.shape[-1]
    exponents = torch.arange(0, width, 2, device=x.device, dtype=x.dtype) / width
    angles = positions.to(x.dtype).unsqueeze(-1) * ROTARY_BASE**-exponents
    cos, sin = angles.cos(), angles.sin()
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


def causal_mask(size: int, device: torch.device, prefix: int = 0) -> Tensor:
    """(size, prefix + size) boolean mask: query i may attend to key j ≤ prefix + i."""
    rows = torch.arange(size, device=device)
    columns = torch.arange(prefix + size, device=device)
    return rows.unsqueeze(-1) + prefix >= columns


def two_stream_mask(size: int, device: torch.device, prefix: int = 0) -> Tensor:
    """(P + 2S) square mask of the causal EBT. Rows and columns 0, ..., P + S − 1 are the
    observed stream, causal within itself; prediction row j (index P + S + j) sees observed
    stream columns 0, ..., P + j and its own column."""
    observed = prefix + size
    mask = torch.zeros(observed + size, observed + size, dtype=torch.bool, device=device)
    mask[:observed, :observed] = causal_mask(observed, device)
    mask[observed:, :observed] = causal_mask(size, device, prefix)
    mask[observed:, observed:] = torch.eye(size, dtype=torch.bool, device=device)
    return mask


def stream_positions(size: int, device: torch.device, prefix: int = 0) -> tuple[Tensor, Tensor]:
    """Rotary positions of the observed stream (0, ..., P + S − 1: the prefix, then observed
    token i at P + i) and of the predictions (P + 1, ..., P + S): the prediction for the
    character after observed token j is at P + j + 1."""
    observed = torch.arange(prefix + size, device=device)
    return observed, observed[prefix:] + 1


def prediction_attention(
    query: Tensor, observed_key: Tensor, observed_value: Tensor, key: Tensor, value: Tensor
) -> Tensor:
    """Attention of each prediction to observed stream positions 0, ..., P + j and to itself.

    `query`, `key` and `value` belong to the S predictions, (..., S, d); `observed_key` and
    `observed_value` to the observed stream of P + S positions, (..., P + S, d). The scores
    against the observed stream are masked causally with offset P; each prediction's score
    against its own key is one extra column of the softmax (§C.3).
    """
    size = query.shape[-2]
    prefix = observed_key.shape[-2] - size
    scale = math.sqrt(query.shape[-1])
    scores = query @ observed_key.transpose(-2, -1) / scale
    scores = scores.masked_fill(~causal_mask(size, query.device, prefix), -torch.inf)
    own = (query * key).sum(-1, keepdim=True) / scale
    weights = torch.cat((scores, own), dim=-1).softmax(dim=-1)
    return weights[..., :-1] @ observed_value + weights[..., -1:] * value
