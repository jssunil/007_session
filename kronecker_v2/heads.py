"""Output heads (SPEC.md §5).

Loss-scale convention (documented, CLAUDE.md risk 3): the byte-grid loss is
the MEAN over the 32 slots of the per-slot cross-entropy, so its magnitude is
comparable to the dense head's single cross-entropy term. PAD (index 256) is
an explicit target class — the loss covers all slots, teaching the model
where tokens end (see codec.py padding convention).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from kronecker_v2.codec import KroneckerCodec
from kronecker_v2.config import CHAR_DIM, HEAD_CHAR_DIM, PAD_IDX


class DenseHead(nn.Module):
    """Baseline. Linear(d_model, V), no bias. Params: D * V."""

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, h: Tensor) -> Tensor:
        return self.proj(h)                          # [B, T, V]

    def loss(self, h: Tensor, target_ids: Tensor) -> Tensor:
        logits = self.forward(h)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target_ids.reshape(-1))


class ByteGridHead(nn.Module):
    """THE CONTRIBUTION. Predict the discrete grid: 32 x 257-way classifications.

    Params: D * POS_DIM * 257 (no bias), independent of V.

    tied_projection: if given (arm E), the 256-wide sub-block of each slot's
    classifier reuses the input projection's weight (option (b) from SPEC §5):
    the input Linear weight is [D, CODE_DIM]; viewed as [D, CHAR_DIM, POS_DIM]
    (code layout: cell = byte*pos_dim + position), slot p's byte-classifier is
    its transpose slice. Only the PAD logit per slot is a new parameter.
    """

    def __init__(self, d_model: int, pos_dim: int, codec: KroneckerCodec,
                 tied_projection: nn.Linear | None = None):
        super().__init__()
        self.d_model = d_model
        self.pos_dim = pos_dim
        self.codec = codec
        self.tied = tied_projection is not None
        if self.tied:
            assert tied_projection.weight.shape == (d_model, CHAR_DIM * pos_dim)
            self.tied_projection = tied_projection      # shared module, not copied
            self.pad_weight = nn.Parameter(torch.zeros(pos_dim, d_model))
        else:
            self.proj = nn.Linear(d_model, pos_dim * HEAD_CHAR_DIM, bias=False)

    def forward(self, h: Tensor) -> Tensor:
        """[B, T, D] -> [B, T, POS_DIM, 257] logits."""
        if self.tied:
            # [D, CODE_DIM] -> [D, CHAR_DIM, POS_DIM] -> [POS_DIM, CHAR_DIM, D]
            w = self.tied_projection.weight.view(self.d_model, CHAR_DIM, self.pos_dim)
            w = w.permute(2, 1, 0)                                  # [P, 256, D]
            byte_logits = torch.einsum("btd,pcd->btpc", h, w)       # [B, T, P, 256]
            pad_logit = torch.einsum("btd,pd->btp", h, self.pad_weight).unsqueeze(-1)
            return torch.cat([byte_logits, pad_logit], dim=-1)      # [B, T, P, 257]
        logits = self.proj(h)
        return logits.view(*h.shape[:-1], self.pos_dim, HEAD_CHAR_DIM)

    def loss(self, h: Tensor, target_ids: Tensor) -> Tensor:
        logits = self.forward(h)                                    # [B, T, P, C]
        targets, _ = self.codec.target_bytes(target_ids)            # [B, T, P]
        return F.cross_entropy(
            logits.reshape(-1, HEAD_CHAR_DIM), targets.reshape(-1)
        )  # mean over B*T*P == mean over slots of per-slot CE

    @torch.no_grad()
    def greedy_bytes(self, h: Tensor) -> Tensor:
        """Unconstrained decode: argmax per slot. [B, T, POS_DIM] int64."""
        return self.forward(h).argmax(dim=-1)


class ByteGridARHead(nn.Module):
    """ByteGridHead's parallel logits + a small autoregressive corrector.

    Removes the conditional-independence assumption: slot p sees the true
    bytes of slots < p (teacher forcing) via a tiny causal transformer of
    width d_local whose output is ADDED to the parallel per-slot logits.
    Keeping the parallel projection makes arm D differ from arm C only in
    the autoregressive term, not in how much of h reaches the slots.
    Params: D * POS_DIM * 257 + small local decoder, independent of V.
    """

    def __init__(self, d_model: int, pos_dim: int, codec: KroneckerCodec,
                 d_local: int = 128, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.pos_dim = pos_dim
        self.codec = codec
        self.d_local = d_local
        self.proj = nn.Linear(d_model, pos_dim * HEAD_CHAR_DIM, bias=False)
        self.h_proj = nn.Linear(d_model, d_local, bias=False)
        self.byte_emb = nn.Embedding(HEAD_CHAR_DIM + 1, d_local)  # +1: BOS index 257
        self.slot_emb = nn.Embedding(pos_dim, d_local)
        layer = nn.TransformerEncoderLayer(
            d_model=d_local, nhead=n_heads, dim_feedforward=4 * d_local,
            dropout=0.0, batch_first=True, norm_first=True, activation="gelu",
        )
        self.decoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out = nn.Linear(d_local, HEAD_CHAR_DIM, bias=False)
        self.bos_idx = HEAD_CHAR_DIM  # 257

    def _slot_logits(self, h_flat: Tensor, prev_bytes: Tensor) -> Tensor:
        """h_flat: [N, D], prev_bytes: [N, P] (slot p holds byte of slot p-1,
        slot 0 holds BOS). Returns [N, P, 257]."""
        n, p = prev_bytes.shape
        x = self.byte_emb(prev_bytes) + self.slot_emb.weight[:p] + self.h_proj(h_flat).unsqueeze(1)
        mask = nn.Transformer.generate_square_subsequent_mask(p, device=x.device)
        return self.out(self.decoder(x, mask=mask, is_causal=True))

    def _parallel_logits(self, h: Tensor) -> Tensor:
        return self.proj(h).view(*h.shape[:-1], self.pos_dim, HEAD_CHAR_DIM)

    def forward(self, h: Tensor, target_ids: Tensor) -> Tensor:
        """Teacher-forced logits [B, T, POS_DIM, 257]."""
        targets, _ = self.codec.target_bytes(target_ids)            # [B, T, P]
        b, t, p = targets.shape
        h_flat = h.reshape(b * t, -1)
        prev = torch.full((b * t, p), self.bos_idx, dtype=torch.int64, device=h.device)
        prev[:, 1:] = targets.reshape(b * t, p)[:, :-1]
        ar = self._slot_logits(h_flat, prev).view(b, t, p, HEAD_CHAR_DIM)
        return self._parallel_logits(h) + ar

    def loss(self, h: Tensor, target_ids: Tensor) -> Tensor:
        logits = self.forward(h, target_ids)
        targets, _ = self.codec.target_bytes(target_ids)
        return F.cross_entropy(logits.reshape(-1, HEAD_CHAR_DIM), targets.reshape(-1))

    @torch.no_grad()
    def greedy_bytes(self, h: Tensor) -> Tensor:
        """Greedy left-to-right decode. [B, T, POS_DIM] int64."""
        b, t = h.shape[:2]
        h_flat = h.reshape(b * t, -1)
        n = h_flat.shape[0]
        parallel = self._parallel_logits(h).view(n, self.pos_dim, HEAD_CHAR_DIM)
        prev = torch.full((n, 1), self.bos_idx, dtype=torch.int64, device=h.device)
        out_bytes = []
        for p in range(self.pos_dim):
            logits = self._slot_logits(h_flat, prev)                # [N, p+1, 257]
            nxt = (parallel[:, p] + logits[:, -1]).argmax(dim=-1)   # [N]
            out_bytes.append(nxt)
            prev = torch.cat([prev, nxt.unsqueeze(1)], dim=1)
        return torch.stack(out_bytes, dim=1).view(b, t, self.pos_dim)
