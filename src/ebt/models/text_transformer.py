"""Causal text transformer and causal text EBT on the same backbone (method.md 3.1, 4.1, 18.3).

Backbone, as the authors' text models (§I.1.2, §D.1): pre-norm blocks with RMSNorm, attention
with rotary positions and a SwiGLU feed-forward of hidden width 4D (the authors' code default;
§D.1.1 says no multiplier, method.md D11), bias-free linear layers and a final RMSNorm. Head
width 64. Initialization (method.md section 18.3): Xavier-normal for every linear layer, with
the query, key and value maps initialized as three separate D × D blocks; N(0, 0.02²) for the
token embedding; RMSNorm weights 1. The EBT's step-index embedding keeps PyTorch's default
N(0, 1), as upstream, which does not re-initialize it.

- `CausalTransformer`, the baseline: token embedding, the blocks with a causal mask and a
  bias-free D → V output layer; trained with cross-entropy.
- `TextEnergy`, the EBT: one learned step-index embedding at position 0, the observed tokens
  and the candidate of every next position (B × S × V) in the two-stream layout of
  `ebt.models.causal_attention` (method.md sections 11 and 18.3). The thinking variable is the
  softmax over V of the noisy candidate (`transform_candidate`, method.md section 4.1); a
  bias-free V → D map encodes it, a bias-free D → 1 head gives one energy per predicted
  position, and `after_step` applies a learnable RMSNorm over V after every thinking step
  (`norm_pred`, with PyTorch's default ε, as upstream).
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ebt.core.energy import EnergyModel
from ebt.models.attention import attention
from ebt.models.causal_attention import (
    causal_mask,
    prediction_attention,
    rotary,
    stream_positions,
    two_stream_mask,
)

HEAD_WIDTH = 64
FFN_MULTIPLIER = 4
"""SwiGLU hidden width = FFN_MULTIPLIER · D, without rounding (method.md D11)."""
NORM_EPS = 1e-5
EMBEDDING_STD = 0.02
STEP_TOKENS = 1
"""Step-index tokens of the EBT: one, since S2 thinks on one energy landscape (section 11)."""
LAYOUTS = ("full", "split")
"""Attention layout of the EBT: the 2S × 2S mask (§C.4) or the split computation (§C.3)."""


@dataclass(frozen=True)
class TextTransformerConfig:
    """Size of the text models and the attention layout of the EBT (method.md section 18.3)."""

    vocabulary: int
    dim: int
    depth: int
    layout: str = "split"
    head_width: int = HEAD_WIDTH

    def __post_init__(self) -> None:
        if self.dim % self.head_width or self.head_width % 2:
            raise ValueError("dim must be a multiple of an even head width")
        if self.layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {LAYOUTS}")

    @property
    def heads(self) -> int:
        return self.dim // self.head_width


class Block(nn.Module):
    """x ← x + attention(RMSNorm(x)); x ← x + SwiGLU(RMSNorm(x)).

    The attention is split into `project` (queries, keys and values with rotary positions) and
    `finish` (output projection, residual and feed-forward), so that the EBT can combine the
    observed stream and the predictions in either layout with the same weights.
    """

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.attention_norm = nn.RMSNorm(dim, eps=NORM_EPS)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)
        self.feed_forward_norm = nn.RMSNorm(dim, eps=NORM_EPS)
        hidden = FFN_MULTIPLIER * dim
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def project(self, x: Tensor, positions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Queries, keys and values (B, heads, T, head width) of RMSNorm(x)."""
        batch, tokens, _ = x.shape
        qkv = self.qkv(self.attention_norm(x)).view(batch, tokens, 3, self.heads, -1)
        query, key, value = qkv.permute(2, 0, 3, 1, 4)
        return rotary(query, positions), rotary(key, positions), value

    def finish(self, x: Tensor, attended: Tensor) -> Tensor:
        """Residual update of x with the attended values (B, heads, T, head width)."""
        batch, tokens, dim = x.shape
        x = x + self.output(attended.transpose(1, 2).reshape(batch, tokens, dim))
        hidden = self.feed_forward_norm(x)
        return x + self.down(F.silu(self.gate(hidden)) * self.up(hidden))

    def forward(self, x: Tensor, positions: Tensor, mask: Tensor) -> Tensor:
        return self.finish(x, attention(*self.project(x, positions), mask))


class Backbone(nn.Module):
    """Token embedding, the blocks and the final RMSNorm, shared by both text models."""

    def __init__(self, config: TextTransformerConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(config.vocabulary, config.dim)
        self.blocks = nn.ModuleList(Block(config.dim, config.heads) for _ in range(config.depth))
        self.norm = nn.RMSNorm(config.dim, eps=NORM_EPS)

    def embed(self, tokens: Tensor) -> Tensor:
        return self.embedding(tokens.long())


def initialize(module: nn.Module) -> None:
    """Xavier-normal weights for every linear layer, the fused query, key and value map as three
    separate D × D blocks; N(0, 0.02²) for every embedding of `module`; RMSNorm weights stay 1."""
    fused = {id(block.qkv) for block in module.modules() if isinstance(block, Block)}
    with torch.no_grad():
        for child in module.modules():
            if isinstance(child, nn.Linear):
                weights = child.weight.chunk(3) if id(child) in fused else (child.weight,)
                for weight in weights:
                    nn.init.xavier_normal_(weight)
            elif isinstance(child, nn.Embedding):
                nn.init.normal_(child.weight, std=EMBEDDING_STD)


class CausalTransformer(nn.Module):
    """Next-character logits (B, S, V) of tokens (B, S)."""

    def __init__(self, config: TextTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = Backbone(config)
        self.head = nn.Linear(config.dim, config.vocabulary, bias=False)
        initialize(self)

    def forward(self, tokens: Tensor) -> Tensor:
        size = tokens.shape[1]
        positions, _ = stream_positions(size, tokens.device)
        mask = causal_mask(size, tokens.device)
        x = self.backbone.embed(tokens)
        for block in self.backbone.blocks:
            x = block(x, positions, mask)
        return self.head(self.backbone.norm(x))


class TextEnergy(EnergyModel):
    """Energy per predicted position (B, S) of tokens (B, S) and candidates (B, S, V)."""

    candidate_event_dims = 1

    def __init__(self, config: TextTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = Backbone(config)
        self.vocab_to_embed = nn.Linear(config.vocabulary, config.dim, bias=False)
        self.head = nn.Linear(config.dim, 1, bias=False)
        self.norm_pred = nn.RMSNorm(config.vocabulary)
        initialize(self)
        # Created after `initialize`: it keeps PyTorch's default N(0, 1), as upstream.
        self.step_embedding = nn.Embedding(STEP_TOKENS, config.dim)

    def transform_candidate(self, candidate: Tensor) -> Tensor:
        """Softmax over the vocabulary: the thinking variable (method.md section 4.1)."""
        return candidate.softmax(dim=-1)

    def encode_candidate(self, variable: Tensor) -> Tensor:
        """Bias-free V → D map of the probabilities."""
        return self.vocab_to_embed(variable)

    def after_step(self, candidate: Tensor) -> Tensor:
        """Learnable RMSNorm over the vocabulary after every step (`norm_pred`)."""
        return self.norm_pred(candidate)

    def energy(self, context: Tensor, encoded: Tensor) -> Tensor:
        """`encoded` is the output of `encode_candidate`, (B, S, D)."""
        step = self.step_embedding.weight.expand(len(context), -1, -1)
        observed = torch.cat((step, self.backbone.embed(context)), dim=1)
        if self.config.layout == "full":
            predicted = self._full(observed, encoded)
        else:
            predicted = self._split(observed, encoded)
        return self.head(self.backbone.norm(predicted)).squeeze(-1)

    def _full(self, observed: Tensor, predicted: Tensor) -> Tensor:
        size = predicted.shape[1]
        observed_positions, predicted_positions = stream_positions(
            size, observed.device, STEP_TOKENS
        )
        positions = torch.cat((observed_positions, predicted_positions))
        mask = two_stream_mask(size, observed.device, STEP_TOKENS)
        x = torch.cat((observed, predicted), dim=1)
        for block in self.backbone.blocks:
            x = block(x, positions, mask)
        return x[:, observed.shape[1] :]

    def _split(self, observed: Tensor, predicted: Tensor) -> Tensor:
        size = predicted.shape[1]
        observed_positions, predicted_positions = stream_positions(
            size, observed.device, STEP_TOKENS
        )
        mask = causal_mask(observed.shape[1], observed.device)
        for block in self.backbone.blocks:
            observed_query, observed_key, observed_value = block.project(
                observed, observed_positions
            )
            query, key, value = block.project(predicted, predicted_positions)
            attended = prediction_attention(query, observed_key, observed_value, key, value)
            predicted = block.finish(predicted, attended)
            observed = block.finish(
                observed, attention(observed_query, observed_key, observed_value, mask)
            )
        return predicted
