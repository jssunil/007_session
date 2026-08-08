# SPEC.md — Technical Specification

Read `CLAUDE.md` first for background and thesis. This file specifies *what to build*.

---

## 1. Repository layout

```
kronecker-v2/
├── CLAUDE.md
├── SPEC.md
├── TASKS.md
├── README.md                    # the submission (written last, from real results)
├── requirements.txt
├── kronecker_v2/
│   ├── __init__.py
│   ├── codec.py                 # byte <-> grid codec (fixed, untrained)
│   ├── embedding.py             # input paths: dense + Kronecker
│   ├── heads.py                 # output heads: dense + byte-grid (parallel & AR)
│   ├── model.py                 # small transformer trunk
│   ├── data.py                  # corpus loading, tokenizer, batching
│   ├── collisions.py            # collision audit
│   ├── metrics.py               # perplexity, exact-recovery, per-script breakdown
│   └── config.py                # dataclass configs
├── experiments/
│   ├── run.py                   # single-arm runner
│   ├── sweep_vocab.py           # vocabulary scaling (arithmetic + 2 trained points)
│   └── configs/
│       ├── arm_a_dense_dense.yaml
│       ├── arm_b_kron_dense.yaml
│       ├── arm_c_kron_bytegrid.yaml
│       ├── arm_d_kron_bytegrid_ar.yaml
│       └── arm_e_kron_bytegrid_tied.yaml
├── tests/
│   ├── test_codec.py            # round-trip correctness — the gate
│   ├── test_heads.py
│   └── test_shapes.py
└── results/
    ├── logs/                    # one JSONL per run
    ├── metrics/                 # final metrics per arm
    └── figures/
```

---

## 2. Core constants

```python
CHAR_DIM = 256      # UTF-8 byte values
POS_DIM  = 32       # byte positions (the window)
CODE_DIM = CHAR_DIM * POS_DIM   # 8192
```

`POS_DIM` must be configurable — the collision audit needs to sweep it (32, 48, 64).

---

## 3. `codec.py` — the fixed byte/grid codec

This module holds **no trainable parameters**. Everything is a buffer.

```python
class KroneckerCodec:
    """Fixed, deterministic token <-> byte-grid codec. Never trained."""

    def __init__(self, vocab_byte_strings: list[bytes],
                 char_dim: int = 256, pos_dim: int = 32):
        """
        vocab_byte_strings[i] is the UTF-8 byte string of token id i.
        Precompute and store as buffers:
          - byte_buffer:  [V, pos_dim] int64, byte values, PAD_IDX where beyond length
          - length_buffer:[V] int64, L = min(len(bytes), pos_dim)
        """

    def encode(self, token_ids: Tensor) -> Tensor:
        """
        token_ids: [B, T] int64
        returns:   [B, T, CODE_DIM] float, the normalised sparse code

        Build by scatter into a zeros tensor (do NOT materialise a dense
        [V, CODE_DIM] table -- that defeats the entire point).
        Apply 1/sqrt(L) scaling, then z-normalise per code.
        """

    def target_bytes(self, token_ids: Tensor) -> tuple[Tensor, Tensor]:
        """
        Supervision targets for the byte-grid head.
        returns:
          byte_targets: [B, T, pos_dim] int64, byte value per slot
          byte_mask:    [B, T, pos_dim] bool, True where slot < L (real byte)
        """

    def decode(self, byte_values: Tensor, lengths: Tensor | None = None) -> list[bytes]:
        """[N, pos_dim] int64 -> list of byte strings, padding stripped."""
```

**Padding convention:** reserve byte index `256` as `PAD_IDX`, making the categorical dimension `257` for the *output head* while the *input grid* stays 256 rows (padding contributes nothing to the input code). Alternatively predict an explicit length. Pick one, document it, be consistent. The `257` approach is simpler and is the default recommendation — it lets the model learn where the token ends rather than needing a separate length head.

**Acceptance gate:** for every token in the vocabulary that is not part of a collision group, `decode(target_bytes(id)) == vocab_byte_strings[id]` exactly, with zero training. If this fails, nothing downstream is meaningful.

---

## 4. `embedding.py` — input paths

```python
class DenseEmbedding(nn.Module):
    """Baseline. nn.Embedding(vocab_size, d_model). Params: V * D."""

class KroneckerEmbedding(nn.Module):
    """
    codec (fixed buffers) -> Linear(CODE_DIM, d_model, bias=False)
    The Linear is the ONLY nn.Parameter in this module.
    Params: CODE_DIM * D, independent of V.

    forward: [B, T] int64 -> [B, T, D] float
    Expose `.projection` so the tied arm can share it.
    """
```

Both must expose an identical interface: `[B, T]` ints in, `[B, T, D]` floats out. Nothing outside these modules may know which one is in use.

---

## 5. `heads.py` — output heads

```python
class DenseHead(nn.Module):
    """
    Baseline. Linear(d_model, vocab_size).
    loss: cross_entropy over V.
    Params: D * V.
    """

class ByteGridHead(nn.Module):
    """
    THE CONTRIBUTION.
    Linear(d_model, POS_DIM * 257) -> reshape [B, T, POS_DIM, 257]
    loss: mean over slots of cross_entropy(logits[..., p, :], byte_target[..., p])
          masked so padding slots do not contribute (or predict PAD explicitly --
          see codec padding convention).
    decode: argmax over last dim -> [B, T, POS_DIM] byte values -> codec.decode
    Params: D * POS_DIM * 257, independent of V.

    Args:
        tied_projection: nn.Linear | None
            If provided, use its weight (transposed) instead of an own weight.
            Requires shape compatibility -- see the tied-arm note below.
    """

class ByteGridARHead(nn.Module):
    """
    ByteGridHead + a small autoregressive decoder over the 32 byte slots,
    to remove the conditional-independence assumption.

    Suggested: 1-2 transformer layers, d_local = 128, causal mask over the
    POS_DIM axis, conditioned on h (project h to d_local and prepend as a
    prefix token, or add it to every slot embedding).
    Train with teacher forcing. Decode greedily left to right.
    Params: D * POS_DIM * 257 + small local decoder. Still independent of V.
    """
```

**Tied-arm note.** The input projection is `Linear(CODE_DIM=8192, D)`; its transpose is `D -> 8192`, which reshapes to `[POS_DIM=32, CHAR_DIM=256]` — that is 256 categories per slot, *without* a PAD slot. So for arm E either (a) drop the PAD category and predict length with a tiny separate head, or (b) keep PAD and tie only the 256-wide sub-block, leaving the PAD logit as a small untied extra. Option (b) is less clean but keeps arms C/D/E comparable. Implement (b), document the choice, and note it as a limitation.

**Constrained decoding.** Build a trie over the vocabulary's byte strings. At decode time, mask logits to bytes that keep the current prefix valid. Report accuracy **both ways**:
- `exact_acc_unconstrained` — does the model emit a valid, correct token on its own?
- `exact_acc_constrained` — does it pick the right token when restricted to real ones?

The gap between the two is itself a reportable result (it quantifies how much of the error is "invalid byte sequence" versus "wrong token").

---

## 6. `model.py` — the trunk

A small, conventional decoder-only transformer. Keep it boring; it is not the object of study.

```python
@dataclass
class ModelConfig:
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    d_ff_mult: int = 4
    seq_len: int = 512
    dropout: float = 0.0
    vocab_size: int = 32768
    input_path: str = "dense"    # "dense" | "kronecker"
    output_head: str = "dense"   # "dense" | "bytegrid" | "bytegrid_ar"
    tie_projection: bool = False
    pos_dim: int = 32
```

Pre-norm blocks, RMSNorm, SwiGLU or plain GELU MLP, RoPE or learned absolute position (either is fine — it is held constant across arms). The trunk is **identical across all five arms**; only `input_path`, `output_head`, and `tie_projection` change.

---

## 7. `data.py` — corpus and tokenizer

- Corpus: **mixed English + at least one Indic script** (Hindi or Telugu). The Indic component is not optional — it is where the collision and per-script results come from. A Wikipedia subset or any openly available mixed corpus is fine. Target ~50–100M training tokens; less is acceptable if runs are stable.
- Tokenizer: byte-level BPE (`tokenizers` library or `sentencepiece`), `vocab_size ≈ 32,768`, trained on the mixed corpus. Save it to disk; all arms must use the identical tokenizer.
- Tag every token in the vocabulary with a **script label** (`latin` / `devanagari` / `telugu` / `other`) based on the Unicode block of its first non-ASCII codepoint. Needed for the per-script breakdown.
- Standard causal LM batching: contiguous chunks of `seq_len`, next-token prediction.
- Fixed held-out validation split. Same split for every arm.

---

## 8. `collisions.py` — the collision audit

Runs **before** any training. Independently useful.

```python
def audit(vocab_byte_strings, script_labels, pos_dim: int) -> CollisionReport:
    """
    Group tokens by their truncated byte prefix (first pos_dim bytes).
    Any group of size > 1 is a collision group.

    Report:
      - total collision groups, total tokens involved
      - breakdown per script
      - the theoretical ceiling on exact-recovery accuracy:
            ceiling = 1 - (tokens_in_collision_groups / V)
        and a token-frequency-weighted version using corpus counts
      - example colliding pairs, per script, for the README
    """
```

Sweep `pos_dim ∈ {32, 48, 64}` and produce a table. This bounds every accuracy number reported later — no arm can exceed the ceiling, and the README must say so.

---

## 9. `metrics.py`

| Metric | Definition |
|---|---|
| `val_loss`, `perplexity` | Standard next-token, on the fixed held-out split. **Note:** dense-head loss (over `V`) and byte-grid loss (over 32 slots) are *not directly comparable*. Always additionally compute a **common-basis token-level NLL** for byte-grid arms — the log-probability the head assigns to the correct token, obtained by summing the per-slot log-probs of that token's byte sequence (trie-normalised over the valid vocabulary). Report this as the comparable number. |
| `exact_acc_unconstrained` | Fraction of positions where greedy byte decode yields exactly the target token. |
| `exact_acc_constrained` | Same, with trie-constrained decoding. |
| `invalid_rate` | Fraction of unconstrained decodes that are not any real vocabulary token. |
| `exact_acc_by_script` | The above, split by target token's script label. |
| `exact_acc_by_frequency` | Split by corpus frequency decile of the target token. Tests whether the byte-grid head helps rare tokens (it should — they share byte statistics with common ones). |
| `params_total`, `params_head`, `params_input` | Counted, not estimated. |
| `train_memory_gb` | `params × 16 bytes` (AdamW mixed precision), matching the course's accounting. State the assumption. |
| `tokens_per_sec` | Measured on identical hardware, same batch size, same seed. |

The comparable-NLL point is important. Getting it wrong invalidates the whole comparison — do not simply put the two raw losses side by side.

---

## 10. Experiment arms

Trunk, data, tokenizer, seed, and schedule are **identical** across arms. One variable changes at a time.

| Arm | Input | Output head | Tests |
|---|---|---|---|
| **A** | Dense | Dense | Baseline |
| **B** | Kronecker | Dense | Current V4/V5 design — isolates input-side effect |
| **C** | Kronecker | Byte-grid (parallel) | **The core claim** |
| **D** | Kronecker | Byte-grid + local AR decoder | Does removing conditional independence close the gap to A/B? |
| **E** | Kronecker | Byte-grid, tied to input projection | True reversibility: one matrix, both directions |

**Go/no-go:** run **A and C first**. If C's comparable NLL is catastrophic relative to A, stop and reassess before spending time on B, D, E. Document that decision either way.

---

## 11. Vocabulary scaling result (the headline figure)

`experiments/sweep_vocab.py`:

- **Arithmetic component** (no training): for `V ∈ {8K, 16K, 32K, 64K, 131072, 262144, 524288, 1000000}`, plot dense head params vs. byte-grid head params, and training memory (×16 bytes). Mark the **crossover at `V = CHAR_DIM × POS_DIM = 8,192`**, below which byte-grid is *worse*. Label the figure as arithmetic.
- **Empirical anchors** (trained): retrain arms A and C at **two** vocabulary sizes (e.g. 32K and 64K) and plot the measured comparable-NLL alongside. This prevents the scaling claim from being purely theoretical — it shows quality does not collapse as `V` grows while the head stays flat.

---

## 12. README requirements

The README is the submission. It must contain, in order:

1. **Which problem** is being solved (Problem 5) and the one-paragraph thesis (predict the discrete grid, don't invert the float vector).
2. **The reframe**, explained clearly enough that someone who has not read the transcript understands why it makes an unsolved problem tractable.
3. **How to reproduce**: exact commands, config files, seed, hardware, wall-clock.
4. **Results**: the arm table, the scaling figure, the per-script and per-frequency breakdowns. Every number labelled as *measured* or *arithmetic*.
5. **The collision ceiling**, stated before the accuracy numbers, so accuracy is read against its true upper bound.
6. **An honest failure table** — what broke, what did not transfer, what the conditional-independence gap actually was, where Indic degraded. This section is not optional and should not be softened.
7. **Prior work**, positioning the contribution narrowly (see `CLAUDE.md`).
8. **Limitations and what would be needed to trust this at scale** — the course's own paper devotes a chapter to evidence boundaries; match that standard.

Figures may be static PNGs or a small self-contained HTML page. A plain README with good figures is explicitly acceptable per the assignment.
