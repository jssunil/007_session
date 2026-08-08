"""Metrics (SPEC.md §9), including the correctness-critical comparable-NLL.

Comparable token-level NLL
--------------------------
Dense-head loss (one CE over V) and byte-grid loss (mean CE over 32 slots)
are NOT directly comparable. The common basis is the log-probability the head
assigns to the correct *token*, normalised over the valid vocabulary:

    score(v)   = log P(v's byte grid | h)   (sum of per-slot log-probs,
                                             PAD slots included — they are
                                             part of the token's identity)
    NLL(y)     = -[ score(y) - logsumexp_v score(v) ]

For a dense head score(v) is just the logit of v, and the same formula
reduces exactly to standard cross-entropy — that identity is unit-tested
(tests/test_metrics.py) and is the gate for M5.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from kronecker_v2.codec import KroneckerCodec
from kronecker_v2.config import HEAD_CHAR_DIM


@torch.no_grad()
def bytegrid_token_scores(slot_logits: Tensor, byte_buffer: Tensor,
                          chunk: int = 2048) -> Tensor:
    """score(v) for every vocab token. [N, P, 257] x [V, P] -> [N, V].

    Sums per-slot log-probs of each token's byte at each slot. PAD slots are
    included: a short token's identity is its bytes *and* where it ends.
    """
    n, p, c = slot_logits.shape
    assert c == HEAD_CHAR_DIM
    logprobs = F.log_softmax(slot_logits.float(), dim=-1)           # [N, P, 257]
    v = byte_buffer.shape[0]
    out = torch.empty(n, v, device=slot_logits.device, dtype=torch.float32)
    for s in range(0, n, chunk):
        lp = logprobs[s : s + chunk]                                # [n', P, 257]
        acc = torch.zeros(lp.shape[0], v, device=lp.device)
        for pos in range(p):
            acc += lp[:, pos, :].index_select(1, byte_buffer[:, pos])
        out[s : s + chunk] = acc
    return out


@torch.no_grad()
def ar_token_scores(head, h_flat: Tensor, byte_buffer: Tensor,
                    vocab_chunk: int = 8192) -> Tensor:
    """Exact score(v) under the AR head, for every vocab token. [N, D] -> [N, V].

    The AR head's token probability does not factorise over slots
    independently — every candidate token must be teacher-forced through the
    local decoder. Cost is one [V, P] decoder pass per position, so this is
    run on a REDUCED eval subset for arm D (documented wherever reported).
    """
    n = h_flat.shape[0]
    v, p = byte_buffer.shape
    bos = torch.full((v, 1), head.bos_idx, dtype=torch.int64, device=h_flat.device)
    prev_all = torch.cat([bos, byte_buffer[:, :-1].to(h_flat.device)], dim=1)  # [V, P]
    out = torch.empty(n, v, device=h_flat.device)
    for i in range(n):
        h_row = h_flat[i : i + 1]
        scores = []
        for s in range(0, v, vocab_chunk):
            prev = prev_all[s : s + vocab_chunk]
            hb = h_row.expand(prev.shape[0], -1)
            parallel = head._parallel_logits(hb).view(prev.shape[0], p, -1)
            logits = parallel + head._slot_logits(hb, prev)          # [v', P, 257]
            lp = F.log_softmax(logits.float(), dim=-1)
            tgt = byte_buffer[s : s + vocab_chunk].to(h_flat.device)
            scores.append(lp.gather(2, tgt.unsqueeze(-1)).squeeze(-1).sum(dim=1))
        out[i] = torch.cat(scores)
    return out


@torch.no_grad()
def comparable_nll(token_scores: Tensor, target_ids: Tensor) -> Tensor:
    """-log P(target | h), vocabulary-normalised. [N, V], [N] -> [N]."""
    return -(
        token_scores.gather(1, target_ids.unsqueeze(1)).squeeze(1)
        - torch.logsumexp(token_scores, dim=1)
    )


class Evaluator:
    """Shared evaluation for all arms. Holds vocab metadata on device."""

    def __init__(self, codec: KroneckerCodec, byte_strings: list[bytes],
                 scripts: list[str], frequencies: np.ndarray, device: torch.device):
        self.codec = codec
        self.byte_strings = byte_strings
        self.byte_to_id = {bs: i for i, bs in enumerate(byte_strings)}
        self.scripts = np.asarray(scripts)
        self.device = device
        # frequency decile per token: decile 0 = rarest 10% of vocab entries
        order = np.argsort(frequencies, kind="stable")
        decile = np.empty(len(frequencies), dtype=np.int64)
        v = len(frequencies)
        for d in range(10):
            decile[order[d * v // 10 : (d + 1) * v // 10]] = d
        self.decile = decile

    @torch.no_grad()
    def token_nll_dense(self, logits: Tensor, targets: Tensor) -> Tensor:
        """[N, V], [N] -> [N]. Standard CE, per position."""
        return F.cross_entropy(logits.float(), targets, reduction="none")

    @torch.no_grad()
    def token_nll_bytegrid(self, slot_logits: Tensor, targets: Tensor) -> Tensor:
        scores = bytegrid_token_scores(slot_logits, self.codec.byte_buffer)
        return comparable_nll(scores, targets)

    @torch.no_grad()
    def exact_recovery(self, pred_bytes: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
        """Greedy byte predictions vs target token ids.

        pred_bytes: [N, POS_DIM] int64 (PAD-terminated).
        Returns (correct [N] bool, valid [N] bool).
        """
        decoded = self.codec.decode(pred_bytes)
        n = len(decoded)
        correct = torch.zeros(n, dtype=torch.bool)
        valid = torch.zeros(n, dtype=torch.bool)
        tgt = targets.cpu().tolist()
        for i, bs in enumerate(decoded):
            tid = self.byte_to_id.get(bs)
            valid[i] = tid is not None
            correct[i] = tid == tgt[i]
        return correct, valid

    def breakdown(self, correct: np.ndarray, targets: np.ndarray) -> dict:
        """Per-script and per-frequency-decile accuracy for a bool array."""
        out: dict = {"by_script": {}, "by_decile": {}}
        tgt_scripts = self.scripts[targets]
        for s in ["latin", "devanagari", "other"]:
            m = tgt_scripts == s
            if m.sum() > 0:
                out["by_script"][s] = {"n": int(m.sum()), "acc": float(correct[m].mean())}
        tgt_dec = self.decile[targets]
        for d in range(10):
            m = tgt_dec == d
            if m.sum() > 0:
                out["by_decile"][str(d)] = {"n": int(m.sum()), "acc": float(correct[m].mean())}
        return out


def train_memory_gb(params: int, bytes_per_param: int = 16) -> float:
    """AdamW mixed-precision accounting (2 bf16 weight + 2 bf16 grad +
    4 fp32 master + 4 + 4 Adam moments), matching the course's arithmetic.
    This is stated accounting, not a measurement."""
    return params * bytes_per_param / 1e9
