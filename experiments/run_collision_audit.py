"""M2 gate (real vocabulary) + M3 collision audit.

1. Gate: for every non-colliding token in the real vocabulary,
   decode(target_bytes(id)) == vocab bytes exactly, with zero training.
2. Audit: collision groups / ceilings for pos_dim in {32, 48, 64},
   written to results/metrics/collisions.json + a markdown table.

Run: python experiments/run_collision_audit.py [--tag ""]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kronecker_v2.codec import KroneckerCodec
from kronecker_v2.collisions import audit
from kronecker_v2.data import load_vocab_meta


def roundtrip_gate(byte_strings: list[bytes], pos_dim: int = 32) -> None:
    codec = KroneckerCodec(byte_strings, pos_dim=pos_dim)
    prefix_groups: dict[bytes, int] = defaultdict(int)
    for bs in byte_strings:
        prefix_groups[bs[:pos_dim]] += 1
    non_colliding = [i for i, bs in enumerate(byte_strings) if prefix_groups[bs[:pos_dim]] == 1]

    ids = torch.arange(len(byte_strings))
    targets, _ = codec.target_bytes(ids)
    decoded = codec.decode(targets)
    failures = [i for i in non_colliding if decoded[i] != byte_strings[i][:pos_dim]]
    # For non-colliding tokens <= pos_dim bytes the round-trip must be EXACT.
    exact_failures = [
        i for i in non_colliding if len(byte_strings[i]) <= pos_dim and decoded[i] != byte_strings[i]
    ]
    assert not failures and not exact_failures, (
        f"M2 GATE FAILED: {len(failures)} prefix mismatches, {len(exact_failures)} exact mismatches"
    )
    print(f"M2 gate PASSED: {len(non_colliding):,}/{len(byte_strings):,} non-colliding tokens "
          f"round-trip exactly at pos_dim={pos_dim}, zero training")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    byte_strings, scripts, freqs = load_vocab_meta(f"data/tokenizer{args.tag}")
    roundtrip_gate(byte_strings)

    reports = {}
    for pos_dim in (32, 48, 64):
        rep = audit(byte_strings, scripts, pos_dim, frequencies=freqs)
        reports[pos_dim] = rep.to_dict()
        print(f"pos_dim={pos_dim}: {rep.n_groups} groups, {rep.tokens_involved} tokens involved, "
              f"ceiling={rep.ceiling_unweighted:.6f} (unweighted) "
              f"{rep.ceiling_freq_weighted:.6f} (freq-weighted)")

    out = Path(f"results/metrics/collisions{args.tag}.json")
    out.write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")

    # markdown table for the README
    lines = [
        "| pos_dim | collision groups | tokens involved | ceiling (unweighted) | ceiling (freq-weighted) |",
        "|---|---|---|---|---|",
    ]
    for pd, r in reports.items():
        lines.append(
            f"| {pd} | {r['n_groups']} | {r['tokens_involved']} | "
            f"{r['ceiling_unweighted']:.4%} | {r['ceiling_freq_weighted']:.4%} |"
        )
    md = "\n".join(lines)
    Path(f"results/metrics/collisions{args.tag}.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
