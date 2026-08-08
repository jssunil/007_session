"""M4 acceptance: parameter counts per arm at V=32768 and V=131072.

Counts are taken from instantiated models (counted, not estimated) at the
experiment shape d_model=384. The byte-grid head's V-independence must be
visible: its params_head column is identical at both vocabulary sizes.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kronecker_v2.codec import KroneckerCodec
from kronecker_v2.config import ModelConfig
from kronecker_v2.model import LM

ARMS = {
    "A dense/dense": ("dense", "dense", False),
    "B kron/dense": ("kronecker", "dense", False),
    "C kron/bytegrid": ("kronecker", "bytegrid", False),
    "D kron/bytegrid_ar": ("kronecker", "bytegrid_ar", False),
    "E kron/bytegrid_tied": ("kronecker", "bytegrid", True),
}


def main() -> None:
    for v in (32768, 131072):
        vocab = [f"tok{i}".encode() for i in range(v)]
        codec = KroneckerCodec(vocab)
        print(f"\nV = {v:,} (d_model=384)")
        print(f"{'arm':<22}{'total':>14}{'input':>14}{'head':>14}{'trunk':>14}")
        for name, (inp, out, tie) in ARMS.items():
            cfg = ModelConfig(vocab_size=v, input_path=inp, output_head=out, tie_projection=tie)
            m = LM(cfg, codec)
            c = m.param_counts()
            print(f"{name:<22}{c['params_total']:>14,}{c['params_input']:>14,}"
                  f"{c['params_head']:>14,}{c['params_trunk']:>14,}")


if __name__ == "__main__":
    main()
