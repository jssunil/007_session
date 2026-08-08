# CLAUDE.md — Kronecker Embedding V2: Reversible Byte-Grid Output Head

This file is the standing context for this repository. Read it fully before starting work.

---

## What this project is

A research submission for **ERA V5, Session 7** (The School of AI). The course is building a 120B sparse-MoE language model ("LightningLM" / V5). Session 7 covered the embedding layer, and the instructor set five open research problems as the assignment, framed as candidate directions for a paper he intends to write called *Kronecker Embedding V2*.

**We are solving Problem 5: reversibility.**

> "Kronecker is forward deterministic (same word will always give same embedding). How do I make a reverse of this (same embedding gives the same Kronecker)? If we can do this, then we can get rid of the final head as well! Then we can have a vocab of 1M as well without any issues!"

The deliverable is: working code, reproducible experiments, and a README with real measured results.

---

## Background you need (self-contained — do not go looking for the course material)

### The problem the course is trying to solve

A standard language model has two vocabulary-facing matrices:

- **Input embedding**: `V × D` — one row per vocabulary token, looked up by token id.
- **Output projection ("head")**: `D × V` — scores every vocabulary token from the final hidden state.

At the course's reference shape (`V = 131,072`, `D = 8,096`), **each of these is 1,061,158,912 parameters** (~1.06B). Untied, that is 2.12B parameters for the two layers that do the least interesting work in the model.

Memory is worse than parameter count suggests. Under AdamW mixed precision a trained parameter costs **16 bytes** (2 bf16 weight + 2 bf16 gradient + 4 fp32 master copy + 4 + 4 for the two Adam moments). So the input table alone occupies **16.98 GB** of training state, and an untied pair occupies **~34 GB** — roughly two-fifths of an 80 GB accelerator, before a single attention head exists.

### Kronecker embeddings (the existing, published solution — input side only)

Instead of storing one row per token, the token's vector is **computed from its own UTF-8 bytes**:

1. **Read the bytes.** A token is a byte string. `L = min(len(byte_seq), pos_dim)`.
2. **Mark a grid.** A grid of `char_dim = 256` rows (one per possible byte value) × `pos_dim = 32` columns (one per byte position). For each byte, mark the cell at `(byte_value, position)`. Flatten → an 8,192-length sparse vector. Marking one cell is a Kronecker product of two one-hot vectors, hence the name:

   ```
   κ(b) = (1/√L) · vec( Σ over positions p of  c[byte_p] ⊗ p[position_p] )
   ```

   then z-normalised. **This code is fixed. It is never trained. It is identical every time.**
3. **Project.** One shared `Linear(8192, d_model)` turns the fixed code into the vector the model uses. **This projection is the only trained thing in the entire input path.**

Result: `8,192 × 8,096 = 66,322,432` parameters versus a dense table's `1,061,158,912` — a **93.75% reduction**. Critically, **`V` does not appear in that multiplication at all.** The input cost is identical whether the vocabulary holds 30K tokens or 500K.

### What is still unsolved

The **output head is still dense** (`D × V`), so vocabulary size still dominates the back half of the model. The instructor has been stuck on inverting the Kronecker construction for roughly two years. From the class transcript, his attempted approach was to treat it as inverting a *float vector* — predict the exact 8,096-float code and map it back to a token — and to reach for VAE / KL-divergence machinery so that approximate predictions could be tolerated ("not a point, a point cloud"). He explicitly left this unsolved. He also correctly noted that cosine-similarity nearest-neighbour lookup does not rescue it, because at initialisation, in high dimensions, all vectors are roughly equidistant.

---

## Our thesis (the contribution)

**Do not invert the float vector. Predict the discrete grid.**

The Kronecker code is not really a float vector — it is a `256 × 32` grid of one-hot marks, i.e. **32 independent 256-way categorical choices** (one byte value per byte position). So the output head should not regress a continuous target. It should:

1. Project `h: [B, T, D]` → `[B, T, 32, 256]` logits,
2. Apply softmax over the 256 byte values at each of the 32 positions,
3. Argmax → 32 bytes → strip padding → the exact token.

Decoding becomes **exact and discrete**, trained with ordinary cross-entropy. No VAE, no KL term, no distance metric, no nearest-neighbour search.

This is the same structural move that made the input side work, applied to the output side:

| | Dense | Kronecker |
|---|---|---|
| Input | look up a row in a `V × D` table | compute the row from bytes |
| Output | score against all `V` rows in a `D × V` table | **predict the bytes** |

**Headline arithmetic** (at `D = 8,096`, `char_dim × pos_dim = 8,192`):

| Vocabulary | Dense head params | Byte-grid head params |
|---|---|---|
| 32,768 | 265.3M | **66.3M** |
| 131,072 | 1.061B | **66.3M** |
| 262,144 | 2.12B | **66.3M** |
| 1,000,000 | 8.10B | **66.3M** |

The byte-grid head is **constant in `V`**. That flat line against an exploding one is the result the instructor asked for ("then we can have a vocab of 1M as well without any issues").

**Important honesty constraint:** the byte-grid head has a *fixed* cost of `D × (char_dim · pos_dim)` = `D × 8192`. It is therefore only a win when `V > 8,192`. Below that it is strictly worse than dense. State this plainly in the README — do not hide it.

**The tied variant.** If the same `8,192 × D` matrix serves as both the input projection and (transposed) the output head, the model's entire vocabulary interface becomes one 66.3M-parameter tensor replacing 2.12B. This is the literal, strongest form of "get rid of the final head." Treat whether tying helps as an **open experimental question**, not an assumption — the course chose *untied* for the dense case, but that argument was about a 1B saving being a rounding error against a geometric constraint, and here the shared tensor is only 66M while the geometric argument (input and output share the same byte basis) is much stronger.

---

## Known risks — these must be measured, not assumed

1. **Collisions are a hard ceiling.** Two tokens sharing their first `pos_dim` bytes produce *identical* codes and are therefore permanently indistinguishable — no output head can separate them, ever. This is worse for Indic scripts: Devanagari/Telugu/Tamil/Bengali cost **3 UTF-8 bytes per character**, so a 32-byte window holds ~10 characters, not 32. A single conjunct such as क्ष is three code points (क + halant + ष) = **9 of the 32 bytes**. Real colliding pair from the course material at `pos_dim=32`: अंतर्राष्ट्रीयकरण and अंतर्राष्ट्रीयता. **Measure the collision count on the actual vocabulary before training and report it as the ceiling on achievable accuracy.**
2. **Conditional independence.** Predicting all 32 byte positions independently from `h` assumes they are conditionally independent given `h`. They are not. This is the known failure mode of non-autoregressive decoding. Mitigation to test: a tiny local autoregressive decoder over the 32 byte slots (the MegaByte / Byte Latent Transformer pattern).
3. **Loss scale mismatch.** 32 cross-entropy terms versus one changes the gradient magnitude reaching the trunk. Verify the arms are compared fairly (same LR schedule, or explicitly tuned per arm and reported as such).
4. **Padding.** Short tokens use few of the 32 slots. Padded positions must be masked out of the loss. Decide, implement, and document.
5. **Per-script asymmetry.** Indic accuracy will likely be worse than English. If so that is a *result*, not a bug to hide.

---

## Prior work — position the contribution narrowly

Byte-level and character-level output heads are **not new**. Relevant prior art: adaptive softmax, hierarchical softmax, MegaByte (Yu et al., 2023), Meta's Byte Latent Transformer, and the broader non-autoregressive decoding literature.

**Our contribution is narrower and must be stated as such:** making the output head the *exact structural inverse of the Kronecker input codec*, so that a single byte/position basis serves both directions and vocabulary size disappears from the entire token interface — plus the empirical question of whether tying that shared projection helps or hurts.

Do not claim novelty we do not have. The course's own reference paper is written this way ("None of the underlying primitives is new... What is new at the systems level is the integration").

---

## Working conventions

- **PyTorch**, single GPU, CPU fallback must work for tests.
- **Every number in the README must come from a run that actually happened.** Do not fabricate, extrapolate, or estimate results and present them as measured. If a number is arithmetic rather than measured (e.g. the vocabulary scaling table), label it as arithmetic.
- **Seed everything.** Report the seed. Runs must be reproducible from the committed config.
- Log to a plain CSV or JSONL per run — no dependency on external experiment trackers.
- Keep the model small enough that a full arm trains in a few hours on one GPU. This is a demonstration, not a scale record.
- If an arm fails, **keep the result and write it up**. Negative results about conditional independence or Indic collisions are among the most valuable things this submission can contain.
- Prefer clear code over clever code. This is being read by a human reviewer as evidence.

## Non-goals

- Do not attempt the other four assignment problems (mathematical structure, multimodal, dynamic window, Fourier). The instructor explicitly said not to mix them. The **only** exception is the collision audit, which enters as a *diagnostic bound on our own result*, not as a proposed solution to Problem 3.
- Do not try to beat a production model. Do not scale up.
- Do not build training infrastructure beyond what the five experiment arms need.

## Deliverables

1. `kronecker_v2/` — the library (codec, heads, model, training loop).
2. `experiments/` — configs and runner for the five arms.
3. `results/` — raw logs, metrics, and generated figures.
4. `README.md` — the submission. States which problem is being solved, the thesis, the evidence, the honest failure table, and the prior-work positioning.

See `SPEC.md` for the technical specification and `TASKS.md` for the ordered implementation plan.
