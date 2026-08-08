# How to read this repository

This is a companion guide to `README.md`. The README states the thesis and
the results; this file explains **how to navigate the code, rerun things,
and interpret every number and file you will encounter** — including how to
check that nothing was fabricated.

---

## 1. The problem in one paragraph

A language model normally ends with a dense output head: a `D × V` matrix
that scores every vocabulary token. At course scale (`D=8,096`,
`V=131,072`) that single matrix is ~1.06B parameters and ~17 GB of AdamW
training state, and it grows linearly with vocabulary size. The course's
Kronecker embedding already removed the `V × D` table on the *input* side by
computing each token's vector from its own UTF-8 bytes. The instructor's
open Problem 5 asks for the reverse: recover the token from a vector, so the
output head can be removed too. His attempts treated this as inverting a
*float vector* (VAE machinery, nearest-neighbour search) and stalled for two
years. **Our move: don't invert the float vector — predict the discrete
grid it was built from.** The Kronecker code is really 32 independent
256-way choices (one byte value per position), so the head becomes 32 small
classifiers trained with ordinary cross-entropy. Decoding is exact argmax,
and vocabulary size disappears from the head entirely.

## 2. Reading order

1. `README.md` — thesis, results, honest failure table. Start here.
2. This file — how to interpret what you see.
3. `CLAUDE.md` / `SPEC.md` / `TASKS.md` — standing context, technical spec,
   and the milestone plan the work followed (with correctness gates).
4. Code, in dependency order:
   `kronecker_v2/config.py` → `codec.py` → `embedding.py` → `heads.py` →
   `model.py` → `metrics.py` → `trie.py` → `collisions.py` →
   `experiments/run.py`.

## 3. Map of the repository

```
kronecker_v2/          the library
  config.py            constants + dataclass configs (seeds live here)
  codec.py             the FIXED byte-grid codec: token bytes -> 256x32 grid
                       code (never trained), and bytes -> token decode
  embedding.py         input paths: dense table, or codec + one Linear
  heads.py             output heads: dense, byte-grid, byte-grid + AR corrector
  model.py             small decoder-only transformer trunk, assembles arms
  metrics.py           comparable NLL (the cross-head metric), evaluator
  trie.py              vocabulary byte-trie for constrained decoding
  collisions.py        collision audit (the accuracy ceiling)
  data.py              corpus prep, BPE tokenizer, exact vocab byte strings
experiments/
  run.py               single-arm trainer: YAML in, logs + metrics out
  sweep_vocab.py       vocabulary-scaling figure (arithmetic + trained anchors)
  configs/arm_*.yaml   the five arms (see §4); _smoke_* are 60-step shakedowns
  download_corpus.py, prepare_data.py   data pipeline
tests/                 21 unit tests, all runnable on CPU
results/
  logs/<run>.jsonl     one line per eval during training (live-tailable)
  metrics/<run>.json   final report per run  <- the numbers in the README
  figures/*.png        generated figures
  checkpoints/<run>.pt model + optimizer state
data/                  corpus, tokenizer, tokenized binaries (gitignored)
```

## 4. The five experiment arms

| Arm | Input path | Output head | Question it answers |
|---|---|---|---|
| A | dense table | dense `D×V` | baseline: what normal looks like |
| B | Kronecker | dense `D×V` | does the Kronecker *input* cost quality? (isolates input effect) |
| C | Kronecker | **byte-grid** | the thesis: can the head predict bytes instead of scoring V tokens? |
| D | Kronecker | byte-grid + tiny AR corrector | does fixing conditional independence (known non-AR failure mode) help? |
| E | Kronecker | byte-grid, **tied** to input projection | the strongest form: one shared tensor is the whole vocab interface |

All arms share the same trunk (6 layers, d_model 384), data, seed (1337),
LR schedule, and evaluation batches. D differs from C *only* by the added
autoregressive term, so any gap between them is attributable to it.

## 5. How to reproduce

```powershell
pip install -r requirements.txt
python -m pytest tests/ -v                       # 21 tests, CPU-safe
python experiments\download_corpus.py            # wikipedia en + hi shards
python experiments\prepare_data.py               # tokenizer + train/val bins
python -m kronecker_v2.collisions                # the ceiling, before training
python experiments\run.py experiments\configs\arm_a_dense_dense.yaml
python experiments\run.py experiments\configs\arm_c_kron_bytegrid.yaml
# ... arms b, d, e likewise
python experiments\sweep_vocab.py                # figures
```

Hardware used: one RTX 5070 Ti (16 GB), Windows 11, PyTorch 2.7 + cu128.
Everything is seeded (`seed: 1337` in every config); the exact config used
for a run is embedded verbatim in the first line of its JSONL log.

## 6. How to read a training log (`results/logs/<run>.jsonl`)

Line 1 is a `start` event: the full config, parameter counts, device, torch
version. Every subsequent line is one evaluation. Fields:

- `own_loss` — the arm's *own* training objective on validation data.
  **Never compare this across head types**: a dense head's single CE over
  32,768 classes and a byte-grid head's mean CE over 32 slots of 257 classes
  live on different scales. It is logged only for curve shape.
- `comparable_nll` / `ppl` — the number that IS comparable across all arms.
  See §7. Lower is better; `ppl = exp(comparable_nll)`.
- `exact_acc_unconstrained` — greedy argmax at every byte slot, decode,
  check the result is exactly the right token. The model gets no help: if
  it emits a byte string that isn't in the vocabulary, that's wrong.
- `exact_acc_constrained` — same, but decoding walks the vocabulary trie so
  only real tokens can be produced. The gap between constrained and
  unconstrained measures how often the model's byte marginals are
  individually reasonable but jointly invalid — the conditional-independence
  failure mode in its purest form.
- `invalid_rate` — fraction of unconstrained decodes that were not a
  vocabulary token at all.
- `nll_positions` — how many validation positions the NLL was computed on
  (reduced for arm D, whose exact scoring is expensive; see §7).

For dense arms the "decode" is just argmax over V, so unconstrained and
constrained are identical and `invalid_rate` is 0 by construction.

## 7. The comparable-NLL metric (read this before trusting any table)

Dense loss and byte-grid loss are not comparable, so all cross-arm quality
claims use one common basis: the probability the head assigns to the correct
**token**, normalised over the vocabulary.

- For any head, define `score(v)` = log-probability of token `v`'s complete
  byte grid (byte-grid heads: sum of per-slot log-probs, PAD slots included
  — where a token *ends* is part of its identity; dense heads: the logit).
- Then `NLL = -(score(target) - logsumexp over all 32,768 tokens)`.

For a dense head this formula collapses to ordinary cross-entropy — that
identity is unit-tested (`tests/test_metrics.py`), which is what licenses
putting arm A and arm C in the same column. For the AR head (arm D) the
score does not factorise, so every candidate token must be teacher-forced
through the local decoder; that is exact but expensive, so arm D's NLL is
computed on a reduced, fixed subset of positions and the count is recorded
in `nll_positions` wherever it is reported.

## 8. The collision ceiling (read before any accuracy number)

Two tokens whose first 32 bytes are identical produce *identical* codes —
no output head can ever tell them apart. The audit
(`results/metrics/collisions.json`) runs before training and bounds every
accuracy number. At our V=32,768 vocabulary the longest token is 22 bytes,
so there are **zero collisions and the ceiling is 100%** — checked, not
assumed. This is a property of *this* vocabulary size: bigger vocabularies
have longer tokens (the course's own Devanagari example collides at 33+
bytes), which is why the audit is a required first step at any new scale,
and why per-script results matter (Devanagari costs 3 bytes per character,
so its effective window is ~10 characters, not 32).

## 9. How to read `results/metrics/<run>.json`

The final report per run. Key fields beyond §6:

- `params_input` / `params_head` / `params_trunk` — measured by counting
  tensors, not arithmetic. The V-independence claim is visible here:
  compare arm A's head against arm C's at different vocab sizes.
- `train_memory_gb_16B_accounting` — params × 16 bytes (AdamW mixed
  precision: bf16 weight+grad, fp32 master, two Adam moments). This is
  **stated accounting matching the course's arithmetic, not a measurement**.
- `breakdown_unconstrained` / `breakdown_constrained` — accuracy by script
  (latin / devanagari / other) and by vocabulary-frequency decile
  (decile 0 = rarest 10% of vocab entries). This is where per-script
  asymmetry shows up — it is reported, not hidden.
- `eval_protocol` — the eval sizes used, so reduced subsets are explicit.

## 10. Measured vs arithmetic — the honesty rule

Every number in the README is labelled **measured** (came out of a run whose
log is in `results/`) or **arithmetic** (calculated from a formula, e.g. the
course-scale head sizes in the scaling figure — nobody trained a 1M-vocab
model here). If you want to audit a measured number, its raw source is the
JSONL/JSON file named after the run; the config that produced it is embedded
in the log's first line, and rerunning the config with the same seed
reproduces it.

## 11. What "success" and "failure" look like here

- Arm C matching arm A's comparable NLL while its head stays 3.2M params
  (vs A's 12.6M at V=32K, and constant as V grows) = the thesis holds.
- A large unconstrained↔constrained gap, or D clearly beating C = the
  conditional-independence problem is real and the AR corrector is the
  price of fixing it.
- Devanagari accuracy below Latin = the per-script asymmetry the byte
  window predicts. A result, not a bug.
- Arm E (tied) worse than C = tying costs quality and "one tensor for the
  whole interface" is not free.

Any of these outcomes is reported as-is in the README's failure table.

---

## 12. Detailed Code Walkthrough

This section provides a deep-dive walkthrough of each Python source file in the repository, explaining its purpose, how to read it, and what key implementations to look for.

### [config.py](./kronecker_v2/config.py) — Dataclass Configurations
*   **Purpose**: Centralizes the model, training, and data configs using Python `@dataclass`. This ensures configuration parameters are fully typed and easily saved/loaded to/from YAML.
*   **Key Classes**:
    *   [ModelConfig](./kronecker_v2/config.py#L22-L37): Holds hyperparameters for the model architecture (layer counts, hidden size $D$, paths like `dense`/`kronecker`). It also sets the `pos_dim` (default 32) and the local AR corrector dimensions.
    *   [TrainConfig](./kronecker_v2/config.py#L41-L56): Holds optimizer parameters (learning rate, betas, weight decay), scheduler parameters, seeds, and device pref.
    *   [RunConfig](./kronecker_v2/config.py#L69-L75): Bundles `ModelConfig`, `TrainConfig`, and `DataConfig`.
*   **How to read it**:
    *   Start at the top with the core constants: `CHAR_DIM = 256`, `DEFAULT_POS_DIM = 32`, and `HEAD_CHAR_DIM = 257` (256 bytes + 1 padding category).
    *   Look at `save_yaml` and `load_yaml` at the end to understand how configuration files are parsed and serialized.

### [codec.py](./kronecker_v2/codec.py) — The Kronecker Codec
*   **Purpose**: Handles encoding tokens into sparse grid vectors (inputs) and decoding byte sequences back into text tokens (outputs). **Critically, this class contains 0 trainable parameters.**
*   **Key Classes**:
    *   [KroneckerCodec](./kronecker_v2/codec.py#L19): Extends `nn.Module` but registers all variables as non-trainable buffers (`self.register_buffer`).
*   **How to read it**:
    *   **Buffer initialization (`__init__`)**: Note how `vocab_byte_strings` is parsed to build `self.byte_buffer` (shape `[V, pos_dim]`) and `self.length_buffer` (shape `[V]`).
    *   **Scatter-based Encoding (`encode`)**: **This is the core algorithm.** Instead of storing a large `[V, CODE_DIM]` lookup matrix, it extracts the byte sequence for the batch indices and uses PyTorch's `scatter_add_` to populate the `[B, T, CODE_DIM]` tensor dynamically. Look at lines 55-66 to see how byte values and positions map to cells, and how `1/sqrt(L)` scaling is applied before z-score normalization.
    *   **Decoding (`decode`)**: Notice how it scans the byte sequence, stops at the first `PAD_IDX` (256) unless overridden by a fixed length, and converts the integer indices back to a python `bytes` object.

### [embedding.py](./kronecker_v2/embedding.py) — Input Path Embeddings
*   **Purpose**: Exposes a standard interface `[B, T] -> [B, T, D]` for the transformer trunk, hiding whether the embedding is a dense table or computed via Kronecker.
*   **Key Classes**:
    *   [DenseEmbedding](./kronecker_v2/embedding.py#L11): A wrapper around PyTorch's standard `nn.Embedding`. Params: $V \times D$.
    *   [KroneckerEmbedding](./kronecker_v2/embedding.py#L22): Computes sparse codes using the fixed `KroneckerCodec` and projects them to the model's hidden dimension using a single linear layer `nn.Linear(CODE_DIM, d_model)`. **Params: $8,192 \times D$ (independent of vocabulary size $V$).**

### [heads.py](./kronecker_v2/heads.py) — Output Heads
*   **Purpose**: Predicts the next token given final hidden states `[B, T, D]`. It contains the baseline dense head, the proposed parallel byte-grid head, and the autoregressive byte-grid head.
*   **Key Classes**:
    *   [DenseHead](./kronecker_v2/heads.py#L20): Predicts the probability of all $V$ tokens directly.
    *   [ByteGridHead](./kronecker_v2/heads.py#L35): Projects the hidden state to a flat vector of size `pos_dim * 257` logits and reshapes it to `[B, T, pos_dim, 257]`. Look at `forward` to see how it handles the **tied projection (Arm E)**: it views the input projection's weights as `[d_model, CHAR_DIM, pos_dim]`, permutes them, and does an `einsum` to share weights for the first 256 categories, while keeping the `PAD` category untied.
    *   [ByteGridARHead](./kronecker_v2/heads.py#L86): Implements the autoregressive decoder to resolve conditional independence.
*   **How to read it**:
    *   In `ByteGridARHead`, look at `forward` and `_slot_logits`. It adds a small causal Transformer over the 32 slots. During training, it uses teacher forcing (prev byte targets are passed causality-masked).
    *   Look at `greedy_bytes` in `ByteGridARHead` to see how it generates the tokens sequentially, slot-by-slot, from left to right, during evaluation.

### [model.py](./kronecker_v2/model.py) — The Transformer Trunk
*   **Purpose**: Defines the backbone language model that processes the sequence. The trunk is kept identical across all experiments to isolate changes to the embedding/head interfaces.
*   **Key Classes**:
    *   [LM](./kronecker_v2/model.py#L61): The master module that instantiates the embedding (from `embedding.py`), the transformer blocks (`Block`), and the head (from `heads.py`).
    *   [Block](./kronecker_v2/model.py#L33) and [RMSNorm](./kronecker_v2/model.py#L22): Custom implementations of a pre-norm transformer layer and root mean square normalization.
*   **How to read it**:
    *   Look at `param_counts()` to see how parameters are tracked. Note the special check for `tied` heads to avoid double-counting the shared projection matrix.

### [metrics.py](./kronecker_v2/metrics.py) — Comparable Perplexity & Evaluation
*   **Purpose**: Implements the math required to compare the dense head (a single 32,768-class prediction) with the byte-grid head (32 independent 257-class predictions).
*   **Key Functions**:
    *   [bytegrid_token_scores](./kronecker_v2/metrics.py#L30): Calculates `score(v)` for each token in the vocabulary under the parallel byte-grid head by summing the log-probabilities of each token's bytes at each slot.
    *   [ar_token_scores](./kronecker_v2/metrics.py#L52): Calculates exact scores under the autoregressive head. Since it does not factorize, this requires teacher-forcing every token in the vocabulary through the AR head for every sequence position, which is extremely expensive. Hence, this is computed on a smaller subset of evaluation points.
    *   [comparable_nll](./kronecker_v2/metrics.py#L82): Combines token scores and target IDs to produce the normalized token NLL: `-(score(target) - logsumexp(scores))`.
    *   [Evaluator](./kronecker_v2/metrics.py#L91): Performs the evaluation loop, computing accuracy splits per script and per-frequency decile.

### [trie.py](./kronecker_v2/trie.py) — Constrained Prefix Decoding
*   **Purpose**: Restricts the byte-grid head's predictions to only valid vocabulary byte strings, preventing the model from predicting invalid character sequences.
*   **Key Classes**:
    *   [ByteTrie](./kronecker_v2/trie.py#L15): Builds a prefix-trie tree from the vocabulary.
*   **How to read it**:
    *   Look at `constrained_greedy`: It takes the logits at each position, restricts the argmax choices to only the branches that exist under the current prefix in the trie, and chooses the highest scoring valid character or `PAD_IDX` if it reaches a terminal node.

### [collisions.py](./kronecker_v2/collisions.py) — Prefix Collision Audit
*   **Purpose**: Analyzes the vocabulary to find tokens that share their prefix up to `pos_dim` bytes. If two tokens collide, their Kronecker code is identical, making them indistinguishable.
*   **Key Functions**:
    *   [audit](./kronecker_v2/collisions.py#L31): Groups tokens by their first `pos_dim` bytes, flags collision groups, and calculates the unweighted and frequency-weighted accuracy ceiling.

### [data.py](./kronecker_v2/data.py) — Data Pipeline and Tokenization
*   **Purpose**: Manages downloading and loading Wikipedia text, training a byte-level BPE tokenizer, labelling token scripts, and memory-mapping tokenized binaries.
*   **Key Functions**:
    *   [vocab_byte_strings](./kronecker_v2/data.py#L61): Reconstructs the exact raw bytes for each token ID, ensuring mid-character UTF-8 splits do not get replaced by the fallback U+FFFD unicode symbol.
    *   [script_label](./kronecker_v2/data.py#L96): Determines if a token belongs to `latin`, `devanagari`, or `telugu` based on its first non-ASCII codepoint.
    *   [TokenBatcher](./kronecker_v2/data.py#L174): Feeds contiguous batches of token sequences using a reproducible numpy random generator.

---

## 13. Running Experiments and Helper Scripts

Beyond the library code, the repository contains runner and plotting scripts:

*   **`experiments/run.py`**: The training runner. It instantiates the model (`LM`), loader (`TokenBatcher`), optimization loop (AdamW with Cosine Warmup), and evaluators. Outputs are written in JSONL formatting for evaluation steps and final metrics.
*   **`experiments/sweep_vocab.py`**: Computes head parameter counts and training memory demands across vocabulary sizes from 8k to 1M, and maps the crossover threshold.
*   **`experiments/download_corpus.py`**: Utility downloading parquet shards from HF datasets using standard system `curl`.
*   **`experiments/prepare_data.py`**: Extracts text from raw shards, trains the tokenizer, maps the vocab meta-data, and builds the training/val memmap files.

