"""The fixed byte/grid codec (SPEC.md §3). No trainable parameters.

Padding convention (documented decision):
    The INPUT grid has exactly CHAR_DIM=256 rows; padded slots contribute
    nothing to the code (they are simply not marked). The OUTPUT head predicts
    HEAD_CHAR_DIM=257 categories per slot, where index PAD_IDX=256 means
    "this slot is beyond the token's length". The model therefore learns where
    a token ends, and no separate length head is needed.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from kronecker_v2.config import CHAR_DIM, DEFAULT_POS_DIM, PAD_IDX


class KroneckerCodec(nn.Module):
    """Fixed, deterministic token <-> byte-grid codec. Never trained.

    Everything here is a registered buffer; `.parameters()` is empty.
    """

    def __init__(self, vocab_byte_strings: list[bytes],
                 char_dim: int = CHAR_DIM, pos_dim: int = DEFAULT_POS_DIM):
        super().__init__()
        self.char_dim = char_dim
        self.pos_dim = pos_dim
        self.code_dim = char_dim * pos_dim
        self.vocab_size = len(vocab_byte_strings)

        byte_buffer = torch.full((self.vocab_size, pos_dim), PAD_IDX, dtype=torch.int64)
        length_buffer = torch.zeros(self.vocab_size, dtype=torch.int64)
        for i, bs in enumerate(vocab_byte_strings):
            L = min(len(bs), pos_dim)
            length_buffer[i] = L
            for p in range(L):
                byte_buffer[i, p] = bs[p]
        self.register_buffer("byte_buffer", byte_buffer, persistent=False)
        self.register_buffer("length_buffer", length_buffer, persistent=False)

    @torch.no_grad()
    def encode(self, token_ids: Tensor) -> Tensor:
        """[B, T] int64 -> [B, T, CODE_DIM] float32, the normalised sparse code.

        Built by scatter into zeros — a dense [V, CODE_DIM] table is never
        materialised (that would defeat the entire point).
        """
        shape = token_ids.shape
        flat = token_ids.reshape(-1)                                # [N]
        bytes_ = self.byte_buffer[flat]                             # [N, P]
        lengths = self.length_buffer[flat].clamp(min=1)             # [N]

        # grid cell index = byte_value * pos_dim + position
        positions = torch.arange(self.pos_dim, device=flat.device)  # [P]
        cell = bytes_ * self.pos_dim + positions                    # [N, P]
        valid = bytes_ != PAD_IDX                                   # [N, P]
        # route padded slots to cell 0 with weight 0
        cell = torch.where(valid, cell, torch.zeros_like(cell))

        scale = lengths.to(torch.float32).rsqrt().unsqueeze(1)      # 1/sqrt(L)
        src = valid.to(torch.float32) * scale                       # [N, P]

        code = torch.zeros(flat.numel(), self.code_dim, device=flat.device)
        code.scatter_add_(1, cell, src)

        # z-normalise per code
        mean = code.mean(dim=1, keepdim=True)
        std = code.std(dim=1, keepdim=True).clamp(min=1e-6)
        code = (code - mean) / std
        return code.reshape(*shape, self.code_dim)

    @torch.no_grad()
    def target_bytes(self, token_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Supervision targets for the byte-grid head.

        byte_targets: [..., pos_dim] int64 in [0, 256]; PAD_IDX beyond length.
        byte_mask:    [..., pos_dim] bool, True where slot < L (real byte).
        With the 257-category convention the loss is computed over ALL slots
        (PAD is a legitimate target class); the mask is provided for metrics
        and for the alternative masked-loss ablation.
        """
        targets = self.byte_buffer[token_ids]                       # [..., P]
        mask = targets != PAD_IDX
        return targets, mask

    @torch.no_grad()
    def decode(self, byte_values: Tensor, lengths: Tensor | None = None) -> list[bytes]:
        """[N, pos_dim] int64 -> list of byte strings.

        A slot equal to PAD_IDX terminates the token (first PAD wins); if
        `lengths` is given it overrides PAD-based termination.
        """
        byte_values = byte_values.cpu()
        out: list[bytes] = []
        for row_i in range(byte_values.shape[0]):
            row = byte_values[row_i]
            if lengths is not None:
                L = int(lengths[row_i])
            else:
                pad_pos = (row == PAD_IDX).nonzero()
                L = int(pad_pos[0]) if pad_pos.numel() > 0 else self.pos_dim
            out.append(bytes(row[:L].tolist()))
        return out
