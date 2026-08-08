"""Vocabulary scaling figure (SPEC.md §11).

Two components, honestly separated:

1. ARITHMETIC (no training): dense head params (D x V) vs byte-grid head
   params (D x POS_DIM x 257, constant in V), and the x16-byte AdamW training
   memory for each, across V from 8K to 1M. The figure is labelled as
   arithmetic. The crossover is marked: the byte-grid head is *larger* than
   dense below V = POS_DIM x 257 = 8,224 (~ CODE_DIM = 8,192; the extra
   32 x D is the explicit PAD class). This regime is stated, not hidden.

2. EMPIRICAL ANCHORS (measured): comparable NLL of trained arms A and C at
   V = 32,768 and V = 65,536, read from results/metrics/*.json produced by
   experiments/run.py. Plotted only if those files exist — nothing is
   extrapolated or estimated.

Outputs:
  results/figures/vocab_scaling_params.png
  results/figures/vocab_scaling_anchors.png      (only when anchor runs exist)
  results/metrics/vocab_scaling_arithmetic.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kronecker_v2.config import CHAR_DIM, DEFAULT_POS_DIM, HEAD_CHAR_DIM

# Course reference width (ERA V5 / LightningLM), used for the headline
# arithmetic so the numbers match the course's own memory discussion.
D_COURSE = 8096
POS_DIM = DEFAULT_POS_DIM
BYTES_PER_PARAM = 16  # AdamW mixed precision: 2+2 bf16 + 4 fp32 master + 4+4 Adam

VOCABS = [8_192, 16_384, 32_768, 65_536, 131_072, 262_144, 524_288, 1_000_000]

# Anchor runs: (arm label, metrics file, vocab size)
ANCHORS = [
    ("A dense head", "arm_a_dense_dense.json", 32_768),
    ("A dense head", "arm_a_dense_dense_64k.json", 65_536),
    ("C byte-grid head", "arm_c_kron_bytegrid.json", 32_768),
    ("C byte-grid head", "arm_c_kron_bytegrid_64k.json", 65_536),
]


def dense_head_params(d: int, v: int) -> int:
    return d * v


def bytegrid_head_params(d: int) -> int:
    return d * POS_DIM * HEAD_CHAR_DIM  # constant in V


def crossover_v(d: int) -> float:
    """V at which dense head params equal byte-grid head params."""
    return bytegrid_head_params(d) / d  # = POS_DIM * 257 = 8,224


def arithmetic_table() -> dict:
    rows = []
    bg = bytegrid_head_params(D_COURSE)
    for v in VOCABS:
        dn = dense_head_params(D_COURSE, v)
        rows.append({
            "vocab": v,
            "dense_head_params": dn,
            "bytegrid_head_params": bg,
            "dense_train_gb_16B": dn * BYTES_PER_PARAM / 1e9,
            "bytegrid_train_gb_16B": bg * BYTES_PER_PARAM / 1e9,
            "ratio_dense_over_bytegrid": dn / bg,
        })
    return {
        "note": "ARITHMETIC, not measured. D = 8096 (course reference width). "
                "Byte-grid head = D x POS_DIM x 257; dense head = D x V. "
                "Memory = params x 16 bytes (AdamW mixed-precision accounting).",
        "pos_dim": POS_DIM,
        "head_char_dim": HEAD_CHAR_DIM,
        "crossover_vocab": crossover_v(D_COURSE),
        "crossover_note": "Below V = POS_DIM x 257 = 8,224 (~CODE_DIM 8,192) "
                          "the byte-grid head is STRICTLY WORSE than dense.",
        "rows": rows,
    }


def plot_params(table: dict, out_png: Path) -> None:
    vs = [r["vocab"] for r in table["rows"]]
    dense = [r["dense_head_params"] for r in table["rows"]]
    bg = [r["bytegrid_head_params"] for r in table["rows"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, dense_y, bg_y, ylabel in [
        (ax1, dense, bg, "output head parameters"),
        (ax2, [d * BYTES_PER_PARAM / 1e9 for d in dense],
              [b * BYTES_PER_PARAM / 1e9 for b in bg],
              "head training memory (GB, x16 B accounting)"),
    ]:
        ax.plot(vs, dense_y, "o-", label="dense head  (D x V)", color="#c0392b")
        ax.plot(vs, bg_y, "s-", label="byte-grid head  (D x 32 x 257)", color="#2471a3")
        xover = table["crossover_vocab"]
        ax.axvline(xover, color="gray", ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.annotate("crossover V=8,224\n(byte-grid worse below)",
                    xy=(xover, 1), xycoords=("data", "axes fraction"),
                    xytext=(8, -30), textcoords="offset points",
                    fontsize=8, color="gray", va="top")
        ax.set_xlabel("vocabulary size V")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle(f"Output head cost vs vocabulary (ARITHMETIC, D={D_COURSE})", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def plot_anchors(metrics_dir: Path, out_png: Path) -> bool:
    """Measured comparable-NLL anchors. Returns False if no anchor runs exist."""
    series: dict[str, list[tuple[int, float]]] = {}
    for label, fname, v in ANCHORS:
        f = metrics_dir / fname
        if not f.exists():
            continue
        blob = json.loads(f.read_text())
        series.setdefault(label, []).append((v, blob["comparable_nll"]))
    if not any(len(pts) > 0 for pts in series.values()):
        return False

    fig, ax = plt.subplots(figsize=(6, 4.2))
    colors = {"A dense head": "#c0392b", "C byte-grid head": "#2471a3"}
    for label, pts in series.items():
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-",
                label=label, color=colors.get(label))
        for v, nll in pts:
            ax.annotate(f"{nll:.3f}", (v, nll), textcoords="offset points",
                        xytext=(0, 7), fontsize=8, ha="center")
    ax.set_xscale("log")
    ax.set_xlabel("vocabulary size V")
    ax.set_ylabel("val comparable NLL (nats/token)")
    ax.set_title("MEASURED anchors: quality vs vocabulary", fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    return True


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    figures = root / "results" / "figures"
    metrics = root / "results" / "metrics"
    figures.mkdir(parents=True, exist_ok=True)

    table = arithmetic_table()
    (metrics / "vocab_scaling_arithmetic.json").write_text(json.dumps(table, indent=2))
    plot_params(table, figures / "vocab_scaling_params.png")
    print(f"wrote {figures / 'vocab_scaling_params.png'}")
    print(f"crossover at V = {table['crossover_vocab']:.0f}")

    if plot_anchors(metrics, figures / "vocab_scaling_anchors.png"):
        print(f"wrote {figures / 'vocab_scaling_anchors.png'}")
    else:
        print("no trained anchor runs found yet - anchors figure skipped "
              "(nothing is plotted that was not measured)")


if __name__ == "__main__":
    main()
