"""The trunk (SPEC.md §6): small pre-norm decoder-only transformer.

Deliberately boring — it is not the object of study and is identical across
all five arms. RMSNorm, GELU MLP, learned absolute positions (held constant
across arms), causal SDPA attention.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from kronecker_v2.codec import KroneckerCodec
from kronecker_v2.config import ModelConfig
from kronecker_v2.embedding import DenseEmbedding, KroneckerEmbedding
from kronecker_v2.heads import ByteGridARHead, ByteGridHead, ByteGridTokenHead, DenseHead


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.n_heads = cfg.n_heads
        self.norm1 = RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.attn_out = nn.Linear(d, d, bias=False)
        self.norm2 = RMSNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, cfg.d_ff_mult * d, bias=False),
            nn.GELU(),
            nn.Linear(cfg.d_ff_mult * d, d, bias=False),
        )
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor) -> Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        hd = d // self.n_heads
        q, k, v = (z.view(b, t, self.n_heads, hd).transpose(1, 2) for z in (q, k, v))
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).reshape(b, t, d)
        x = x + self.dropout(self.attn_out(attn))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class LM(nn.Module):
    """Trunk + configurable input path and output head."""

    def __init__(self, cfg: ModelConfig, codec: KroneckerCodec | None = None,
                 token_freqs=None):
        super().__init__()
        self.cfg = cfg
        needs_codec = cfg.input_path == "kronecker" or cfg.output_head in (
            "bytegrid", "bytegrid_ar", "bytegrid_token")
        if needs_codec:
            assert codec is not None, "this configuration requires a codec"
            assert codec.pos_dim == cfg.pos_dim
        self.codec = codec

        if cfg.input_path == "dense":
            self.embedding = DenseEmbedding(cfg.vocab_size, cfg.d_model)
        elif cfg.input_path == "kronecker":
            self.embedding = KroneckerEmbedding(codec, cfg.d_model)
        else:
            raise ValueError(cfg.input_path)

        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.final_norm = RMSNorm(cfg.d_model)

        if cfg.output_head == "dense":
            self.head = DenseHead(cfg.d_model, cfg.vocab_size)
        elif cfg.output_head == "bytegrid":
            tied = None
            if cfg.tie_projection:
                assert cfg.input_path == "kronecker", "tying requires the Kronecker input"
                tied = self.embedding.projection
            self.head = ByteGridHead(cfg.d_model, cfg.pos_dim, codec, tied_projection=tied)
        elif cfg.output_head == "bytegrid_token":
            tied = None
            if cfg.tie_projection:
                assert cfg.input_path == "kronecker", "tying requires the Kronecker input"
                tied = self.embedding.projection
            self.head = ByteGridTokenHead(cfg.d_model, cfg.pos_dim, codec, tied_projection=tied,
                                          token_freqs=token_freqs, use_token_bias=cfg.token_bias)
        elif cfg.output_head == "bytegrid_ar":
            self.head = ByteGridARHead(
                cfg.d_model, cfg.pos_dim, codec,
                d_local=cfg.ar_d_local, n_layers=cfg.ar_n_layers, n_heads=cfg.ar_n_heads,
            )
        else:
            raise ValueError(cfg.output_head)

        self.apply(self._init)

    def _init(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def trunk(self, token_ids: Tensor) -> Tensor:
        """[B, T] -> final hidden states [B, T, D]."""
        b, t = token_ids.shape
        x = self.embedding(token_ids) + self.pos_emb.weight[:t]
        for blk in self.blocks:
            x = blk(x)
        return self.final_norm(x)

    def loss(self, token_ids: Tensor, target_ids: Tensor) -> Tensor:
        h = self.trunk(token_ids)
        return self.head.loss(h, target_ids)

    def param_counts(self) -> dict[str, int]:
        def count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters())

        head_params = count(self.head)
        if getattr(self.head, "tied", False):
            # the shared projection is counted under the input path; the
            # head's own (new) parameters are just the per-slot PAD logits
            head_params = self.head.pad_weight.numel()
        return {
            # nn.Module.parameters() de-duplicates shared parameters, so the
            # total is correct for the tied arm without adjustment
            "params_total": count(self),
            "params_input": count(self.embedding),
            "params_head": head_params,
            "params_trunk": sum(count(b) for b in self.blocks) + count(self.pos_emb) + count(self.final_norm),
        }
