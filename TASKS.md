# TASKS.md — Ordered Implementation Plan

Read `CLAUDE.md` (background, thesis, constraints) and `SPEC.md` (technical spec) first.

Work through milestones in order. Each has an explicit acceptance criterion. **Do not proceed past a failing gate** — report the failure and stop for a decision.

---

## M0 — Scaffold

- [ ] Create the repository layout from `SPEC.md` §1.
- [ ] `requirements.txt`: torch, tokenizers (or sentencepiece), numpy, matplotlib, pyyaml, tqdm. Nothing else unless justified.
- [ ] `config.py` with the dataclasses from `SPEC.md` §6, loadable from YAML.
- [ ] Seeding utility that seeds python / numpy / torch and sets deterministic flags. Every run records its seed in its log.
- [ ] JSONL run logger: one line per eval step, plus a final metrics blob.

**Acceptance:** `python -c "import kronecker_v2"` succeeds; a config round-trips YAML → dataclass → YAML unchanged.

---

## M1 — Data and tokenizer

- [ ] Acquire a mixed **English + Indic** corpus (Hindi or Telugu minimum). Document the source and the exact subset used.
- [ ] Train a byte-level BPE tokenizer, `vocab_size = 32768`, on the mixed corpus. Save to `data/tokenizer/`.
- [ ] Extract `vocab_byte_strings: list[bytes]` for all token ids.
- [ ] Compute `script_labels` per token from the Unicode block of the first non-ASCII codepoint (`latin` / `devanagari` / `telugu` / `other`).
- [ ] Compute per-token corpus frequency counts (needed for frequency-decile metrics and the frequency-weighted collision ceiling).
- [ ] Tokenize corpus into a memory-mapped `uint16`/`uint32` array. Fixed train/val split, saved and reused by every arm.

**Acceptance:** round-trip a held-out paragraph of each script through encode → decode losslessly. Report token counts, per-script token share, and fertility (tokens per word) per script.

---

## M2 — Codec and the correctness gate

- [ ] Implement `KroneckerCodec` per `SPEC.md` §3.
- [ ] `encode` must build codes by scatter into zeros — **never** materialise a `[V, CODE_DIM]` dense table. Assert this in a test by checking peak memory or by construction review.
- [ ] Implement `target_bytes` and `decode`.
- [ ] Settle the padding convention (recommendation: `PAD_IDX = 256`, output categorical dim `257`). Document it in a docstring.
- [ ] `tests/test_codec.py`:
  - round-trip every non-colliding vocabulary token exactly, zero training
  - `1/√L` scaling and z-normalisation applied as specified
  - codes for tokens sharing a prefix are near each other in cosine distance; unrelated tokens are not
  - batching: `encode` on `[B, T]` matches per-token `encode` looped

**Acceptance gate — hard stop if this fails.** 100% exact round-trip on all non-colliding tokens. If this does not hold, nothing downstream is meaningful.

---

## M3 — Collision audit

- [ ] Implement `collisions.audit` per `SPEC.md` §8.
- [ ] Run for `pos_dim ∈ {32, 48, 64}`.
- [ ] Emit `results/metrics/collisions.json` and a markdown table.
- [ ] Record: collision groups, tokens involved, per-script breakdown, unweighted ceiling, frequency-weighted ceiling, and a handful of real colliding example pairs per script for the README.

**Acceptance:** the ceiling for the chosen `pos_dim` is a concrete number, committed to the repo, and every later accuracy claim is read against it. Expect Indic collisions to dominate — if they do not, double-check the script labelling before believing it.

---

## M4 — Model components

- [ ] `DenseEmbedding`, `KroneckerEmbedding` (`SPEC.md` §4). Identical interfaces.
- [ ] `DenseHead`, `ByteGridHead` (`SPEC.md` §5).
- [ ] The trunk in `model.py` (`SPEC.md` §6). Identical across arms.
- [ ] Trie over vocabulary byte strings for constrained decoding.
- [ ] `tests/test_shapes.py`: every input/head combination produces correct shapes on a toy batch; parameter counts match the formulas in `CLAUDE.md` exactly (`D×V` for dense, `D×POS_DIM×257` for byte-grid, `CODE_DIM×D` for the Kronecker input projection).
- [ ] `tests/test_heads.py`: with the trunk frozen and the byte-grid head overfit on a tiny batch, exact recovery reaches ~100% on that batch. Confirms the head *can* learn the mapping at all.

**Acceptance:** parameter counts printed for each arm at `V = 32,768` and at `V = 131,072`, matching the spec arithmetic. The `V`-independence of the byte-grid head must be visible in these printouts.

---

## M5 — Metrics, including the comparable-NLL

- [ ] Implement everything in `SPEC.md` §9.
- [ ] **Comparable token-level NLL for byte-grid arms** — the correctness-critical piece. Sum the per-slot log-probabilities of the correct token's byte sequence, normalised over the valid vocabulary via the trie, so it is on the same basis as the dense head's token NLL.
- [ ] Sanity check: for a dense-head model, the comparable-NLL path must reproduce the standard token NLL exactly. Test this.
- [ ] Per-script and per-frequency-decile breakdowns.

**Acceptance:** the comparable-NLL implementation is unit-tested against the dense head. Without this, the cross-arm comparison is invalid — treat it as a gate, not a nicety.

---

## M6 — Training loop and the go/no-go

- [ ] `experiments/run.py`: single-arm runner taking a config path. AdamW, cosine schedule with warmup, gradient clipping, periodic eval, checkpointing, JSONL logging.
- [ ] Mask padded byte slots out of the byte-grid loss (or predict PAD explicitly, per the chosen convention).
- [ ] Write configs for arms A–E.
- [ ] **Train arm A (dense/dense) and arm C (Kronecker/byte-grid) first.**
- [ ] Compare on comparable-NLL, exact-recovery (both decode modes), invalid rate, params, memory, throughput.

**Go/no-go decision point.** If arm C's comparable-NLL is catastrophically worse than A — not "somewhat worse", but so far off the claim is dead — stop and report before spending compute on B, D, E. Record the decision and the numbers behind it either way. A documented negative result here is a legitimate submission; a fabricated positive one is not.

**Before concluding failure, check the usual suspects:** loss-scale mismatch between 32 CE terms and one (M7 covers this), padding leaking into the loss, and the comparable-NLL implementation itself.

---

## M7 — Remaining arms

- [ ] Arm B (Kronecker input, dense head) — isolates how much of any difference comes from the input side rather than the head.
- [ ] `ByteGridARHead` (local autoregressive decoder over the 32 slots), then arm D. The key question: **how much of the C-vs-A gap does removing conditional independence recover?** Report that number explicitly — it is the most scientifically interesting single result in the project.
- [ ] Arm E (tied projection). Implement the tying compromise from `SPEC.md` §5 and document the PAD-category caveat.
- [ ] Learning-rate fairness check: confirm arms are not being compared at a schedule that favours one. If per-arm tuning was needed, say so and show both.

**Acceptance:** five arms, one results table, every cell measured.

---

## M8 — Vocabulary scaling

- [ ] `experiments/sweep_vocab.py` arithmetic component: head parameters and training memory vs. `V ∈ {8K … 1M}`, with the **crossover at V = 8,192 marked** and the figure labelled *arithmetic*.
- [ ] Empirical anchors: retrain A and C at a second vocabulary size (e.g. 64K), plot measured comparable-NLL alongside.
- [ ] Produce the headline figure: dense head parameters climbing, byte-grid flat.

**Acceptance:** the figure exists, is generated by committed code from committed logs, and its arithmetic-versus-measured components are visually distinguishable.

---

## M9 — README

- [ ] Write `README.md` to the structure in `SPEC.md` §12.
- [ ] State the collision ceiling **before** presenting accuracy numbers.
- [ ] Label every number *measured* or *arithmetic*.
- [ ] Write the honest failure table. Include the conditional-independence gap, per-script degradation, the `V < 8,192` regime where this approach loses, and anything that broke along the way.
- [ ] Prior-work section positioning the contribution narrowly (MegaByte, Byte Latent Transformer, adaptive/hierarchical softmax, non-autoregressive decoding).
- [ ] Limitations: what a proxy run at 1B/3B scale would need to show before this is trustworthy at V5 scale.
- [ ] Reproduction instructions: exact commands, seed, hardware, wall-clock per arm.

**Acceptance:** a reader who has not seen the course material can understand the problem, the reframe, the evidence, and the limits — and can rerun it.

---

## Optional, only if time remains

- [ ] `pos_dim` sensitivity: rerun arm C at `pos_dim = 48` and check whether reduced collisions translate into higher exact recovery, particularly for Indic. This connects the collision ceiling to a measured outcome and would strengthen the submission considerably.
- [ ] Open-vocabulary demo: feed the trained arm-C model a token that was never in the training vocabulary and show it still produces a sensible byte code and can be scored — the practical payoff of vocabulary-independence.
- [ ] Small self-contained HTML page for the figures.

---

## Standing reminders

- Every README number comes from a run that happened. No estimates presented as measurements.
- Negative results are kept and written up, not buried.
- One variable changes per arm.
- Do not drift into the other four assignment problems. The collision audit is a *diagnostic bound on our result*, not a proposed solution to Problem 3.
