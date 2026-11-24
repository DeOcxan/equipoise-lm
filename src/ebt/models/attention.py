"""Scaled dot-product attention written out, so that it can be differentiated twice.

PyTorch's fused `scaled_dot_product_attention` has no double backward on the CPU: on torch
2.15 its second derivative fails with "derivative for
aten::_scaled_dot_product_flash_attention_for_cpu_backward is not implemented". The thinking
loop differentiates the energy with respect to the candidate and then, in training, that
gradient with respect to the parameters, so every attention in an energy uses this function
(method.md section 17.3).
"""

import math

import torch
from torch import Tensor


def attention(query: Tensor, key: Tensor, value: Tensor, mask: Tensor | None = None) -> Tensor:
    """softmax(Q Kᵀ / √d + M) V over the last two dimensions, (..., T, d) -> (..., T, d).

    `mask` is a boolean tensor broadcastable to (..., T_query, T_key); False entries are
    excluded from the softmax, as in a causal mask.
    """
    scores = query @ key.transpose(-2, -1) / math.sqrt(query.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, -torch.inf)
    return scores.softmax(dim=-1) @ value
