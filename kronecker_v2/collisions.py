"""Collision audit (SPEC.md §8). Runs before any training.

Two tokens whose first `pos_dim` bytes agree produce identical Kronecker
codes and are permanently indistinguishable to any byte-grid head. The audit
counts these groups and derives the hard ceiling on exact-recovery accuracy.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np


@dataclass
class CollisionReport:
    pos_dim: int
    vocab_size: int
    n_groups: int
    tokens_involved: int
    ceiling_unweighted: float
    ceiling_freq_weighted: float
    per_script: dict = field(default_factory=dict)
    examples: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def audit(
    vocab_byte_strings: list[bytes],
    script_labels: list[str],
    pos_dim: int,
    frequencies: np.ndarray | None = None,
    max_examples: int = 5,
) -> CollisionReport:
    V = len(vocab_byte_strings)
    groups: dict[bytes, list[int]] = defaultdict(list)
    for i, bs in enumerate(vocab_byte_strings):
        groups[bs[:pos_dim]].append(i)

    colliding = {prefix: ids for prefix, ids in groups.items() if len(ids) > 1}
    tokens_involved = sum(len(ids) for ids in colliding.values())

    # Ceiling: within a collision group only one token can ever be recovered.
    # Unweighted: a decoder can still get 1 token per group right.
    recoverable = V - tokens_involved + len(colliding)
    ceiling_unweighted = recoverable / V

    ceiling_freq = 1.0
    if frequencies is not None:
        total = float(frequencies.sum())
        lost = 0.0
        for ids in colliding.values():
            fs = frequencies[np.asarray(ids)]
            # best case: the decoder always resolves the group to its most
            # frequent member; the rest of the group's mass is lost
            lost += float(fs.sum() - fs.max())
        ceiling_freq = 1.0 - lost / max(1.0, total)

    per_script: dict[str, dict] = {}
    examples: dict[str, list] = defaultdict(list)
    script_arr = np.asarray(script_labels)
    for s in sorted(set(script_labels)):
        per_script[s] = {"vocab_entries": int((script_arr == s).sum()), "tokens_in_collisions": 0, "groups": 0}
    for prefix, ids in colliding.items():
        s = script_labels[ids[0]]
        per_script[s]["tokens_in_collisions"] += len(ids)
        per_script[s]["groups"] += 1
        if len(examples[s]) < max_examples:
            examples[s].append(
                [vocab_byte_strings[i].decode("utf-8", errors="replace") for i in ids[:4]]
            )

    return CollisionReport(
        pos_dim=pos_dim,
        vocab_size=V,
        n_groups=len(colliding),
        tokens_involved=tokens_involved,
        ceiling_unweighted=round(ceiling_unweighted, 6),
        ceiling_freq_weighted=round(ceiling_freq, 6),
        per_script=per_script,
        examples=dict(examples),
    )
