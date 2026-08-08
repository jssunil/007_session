"""Input paths (SPEC.md §4). Identical interface: [B,T] int64 -> [B,T,D] float."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from kronecker_v2.codec import KroneckerCodec


class DenseEmbedding(nn.Module):
    """Baseline. Params: V * D."""

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.table = nn.Embedding(vocab_size, d_model)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.table(token_ids)


class KroneckerEmbedding(nn.Module):
    """codec (fixed buffers) -> Linear(CODE_DIM, d_model, bias=False).

    The Linear is the ONLY nn.Parameter in this module.
    Params: CODE_DIM * D, independent of V.
    """

    def __init__(self, codec: KroneckerCodec, d_model: int):
        super().__init__()
        self.codec = codec
        self.projection = nn.Linear(codec.code_dim, d_model, bias=False)

    def forward(self, token_ids: Tensor) -> Tensor:
        code = self.codec.encode(token_ids)          # [B, T, CODE_DIM], no grad
        return self.projection(code)
