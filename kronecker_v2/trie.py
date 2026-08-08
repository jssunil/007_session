"""Trie over vocabulary byte strings, for constrained decoding (SPEC.md §5).

Tokens are keyed by their first pos_dim bytes (the codec's view of them).
A node is terminal if some token's (truncated) byte string ends there.
"""

from __future__ import annotations

import torch
from torch import Tensor

from kronecker_v2.config import PAD_IDX


class ByteTrie:
    def __init__(self, vocab_byte_strings: list[bytes], pos_dim: int):
        self.pos_dim = pos_dim
        # node = (children: dict[int, node], token_id: int | None)
        self.root: dict = {}
        self.terminal: dict[int, int] = {}  # id(node-dict) -> token id
        for tid, bs in enumerate(vocab_byte_strings):
            node = self.root
            for byte in bs[:pos_dim]:
                node = node.setdefault(byte, {})
            # first token wins for duplicate (colliding) prefixes
            self.terminal.setdefault(id(node), tid)

    @torch.no_grad()
    def constrained_greedy(self, slot_logprobs: Tensor) -> tuple[Tensor, Tensor]:
        """Greedy decode restricted to real vocabulary prefixes.

        slot_logprobs: [N, POS_DIM, 257] log-probs (or logits; only argmax
        order matters within the allowed set).
        Returns (byte_values [N, POS_DIM] with PAD fill, token_ids [N]).
        """
        n, p, c = slot_logprobs.shape
        lp = slot_logprobs.cpu()
        out_bytes = torch.full((n, p), PAD_IDX, dtype=torch.int64)
        out_ids = torch.zeros(n, dtype=torch.int64)
        for i in range(n):
            node = self.root
            for pos in range(p):
                allowed = list(node.keys())
                can_stop = id(node) in self.terminal
                if can_stop:
                    allowed = allowed + [PAD_IDX]
                if not allowed:
                    break
                scores = lp[i, pos, allowed]
                choice = allowed[int(scores.argmax())]
                if choice == PAD_IDX:
                    break
                out_bytes[i, pos] = choice
                node = node[choice]
            out_ids[i] = self.terminal.get(id(node), -1)
            if out_ids[i] == -1:
                # prefix ended at a non-terminal node (only possible when the
                # trie ran out of children); fall back to deepest terminal seen
                out_ids[i] = 0
        return out_bytes, out_ids
