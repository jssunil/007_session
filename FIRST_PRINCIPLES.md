# First Principles: Why the Byte-Grid Output Head Works (and Where It Doesn't)

This note builds the project up from the ground. Each step says what the code does, why it is there, and what the math is. Numbers marked **smoke** come from 983k-token, single-seed runs and are not benchmarks (see README Section 6).

---

## 0. The problem

A language model must choose the next token from V candidates (V = 32,768 here). The standard **dense head** multiplies the hidden state `h` (size D) by a matrix `W` of shape `[V, D]`:

```
logit(v) = W[v] · h            P(v | h) = softmax_v( logit(v) )
```

`W` has V × D parameters, so it grows with the vocabulary. The question: can the head be built from the structure of the token itself, so it does not grow with V?

---

## 1. A token is a short byte string (`codec.py`)

Each token is a sequence of bytes `(b_0, b_1, ..., b_31)`. Each `b_p` takes one of 257 values: 256 byte values plus **PAD** (256), meaning "the token already ended".

- Code: `KroneckerCodec.byte_buffer`, a table of shape `[V, 32]`.
- Use: a token's identity becomes 32 small categorical choices, not one choice among 32,768.
- Limit: tokens longer than 32 bytes are truncated, so two tokens sharing a 32-byte prefix are indistinguishable. This is why the collision audit exists. A zero-collision result is an accuracy **ceiling**, not an accuracy.

---

## 2. Arm C: the parallel byte-grid head (`ByteGridHead`)

The head maps `h` to logits `S` of shape `[32, 257]`. Row `p` scores "which byte sits at position p". Each row is trained with ordinary cross-entropy.

```
P(token | h) = Π_p P(b_p | h)        (assumes the 32 slots are independent given h)
```

- Parameters: `D × 32 × 257`, independent of V. It beats a dense head only when V is above roughly 8,224.
- **The flaw.** The independence assumption is false. After "c", the next byte should be "a", not "x". Each slot can be individually plausible while the combination is not a real token, so greedy argmax per slot produces invalid strings.
- Smoke evidence: 98.9% of unconstrained outputs were not real tokens.

---

## 3. Arm E: tying input and output (`tied_projection`)

The input side already has a matrix `Linear(8192, D)`. Arm E reuses its transpose as the output classifier, so one tensor serves both directions.

- Math: the input maps a grid cell (byte, position) to a column `W[:, cell]`; the output score of that cell is `h · W[:, cell]`, the same vector used backwards.
- Smoke evidence: the tied head still produced 100% invalid tokens. Tying alone does not fix the independence flaw, so it is a hypothesis, not a result.

---

## 4. Comparing heads fairly (`metrics.py: comparable_nll`)

Dense loss and byte-grid loss are different quantities. The common basis is the probability the head gives the correct **whole token**:

```
score(v) = Σ_p log P(b_p(v) | h)                 (v's own bytes at each slot, PAD included)
NLL(y)   = -[ score(y) - logsumexp_v score(v) ]
```

- For a dense head `score(v)` is the logit, and the formula reduces exactly to cross-entropy. A unit test checks this to 1e-5.
- Code: `bytegrid_token_scores` picks `logprob[p, byte_buffer[v, p]]` for every token and sums over `p`.

---

## 5. Constrained decoding: the vocab-MAP (`Evaluator.vocab_map_hits`)

Instead of an argmax per slot, choose the single real token with the best score:

```
v* = argmax_{v in vocabulary} score(v)
```

- It can only return a real token.
- Any trie search or beam, of any width, searches the same candidates with the same scoring, so the best it can reach is `v*`. For the parallel heads this is the upper bound, so a separate beam search adds nothing. (A beam only matters for the autoregressive head, arm D.)
- Smoke evidence (arm C): vocab-MAP 1.65% top-1 vs greedy trie decoding 1.25%.

---

## 6. Arm F: train on the whole token (`ByteGridTokenHead`)

Arm F keeps arm C's network but changes the loss:

```
score(v) = Σ_p S[p, b_p(v)]  +  β_v
loss     = cross_entropy over all V tokens of score(v)
```

`S[p, c]` is the logit for byte `c` at slot `p`; `β_v` is one learned scalar per token.

**Why this removes the independence flaw.** Arm C trains each slot to be right alone. Arm F trains the *sum over a token's slots* to beat the sum for every other real token. The model is penalised for combinations that are not real tokens, and independence is never assumed.

**Why raw logits are enough.** Log-softmax at slot `p` subtracts `logsumexp_c S[p, c]`, which is the same for every token `v`. That shifts every `score(v)` by one common constant, and softmax over V ignores a common shift. So training can use raw logits (cheaper) and evaluation can use log-probabilities, with identical results. A test checks this.

**Why the bias `β_v`.** All byte codes have about the same size, so the sum cannot express that "the" is far more common than a rare token. A dense head stores that in the norm of `W[v]`. We initialise

```
β_v = log( freq(v) / Σ_u freq(u) )         (log unigram probability)
```

so at step 0 the model already knows token frequencies, and training only learns the context-dependent part. Cost: V extra parameters (about 0.26% of a dense head). The head is therefore **no longer strictly independent of V**, and the README must say so.

**Chunking.** Scoring 16,384 positions against 32,768 tokens at once builds a 2 GB table. The loss runs 2,048 rows at a time and recomputes each chunk in the backward pass (`torch.utils.checkpoint`).

---

## 7. What the tests prove (`tests/`)

| Test | Question it answers | Math it checks |
|---|---|---|
| `test_free_decoding_validity_vs_vocab_map` | Does argmax emit non-tokens, and does vocab-MAP fix it? | argmax over slots vs argmax over real tokens |
| `test_code_noise_robustness_argmax_vs_nearest_vocab` | Does the vocabulary act as an error-correcting prior? | per-position argmax vs nearest vocabulary code under Gaussian noise |
| `test_active_slot_accuracy_ignores_padding` | Is accuracy inflated by free PAD hits? | accuracy over slots where target ≠ PAD |
| `test_gather_scores_equal_explicit_expanded_matrix` | Is the fast gather the same linear map as a one-hot product? | `Σ_p lp[p, b_p] = ⟨lp, onehot⟩` |
| `test_loss_matches_explicit_dense_ce_and_chunking` | Does chunked arm F loss equal plain cross-entropy? | `CE(score)` with a dense one-hot matrix |
| `test_eval_scores_consistent_with_training_scores` | Do raw-logit training and log-prob evaluation agree? | constant shifts cancel in softmax |
| `test_bias_init_is_log_unigram_probability` | Is the prior correct at step 0? | `softmax(β) = freq / Σ freq` |

All are CPU-only and use no training, so they check mechanisms, not model quality.

---

## 8. Smoke results and the speed problem

Smoke (983k tokens, 1 seed, 60 steps):

| Arm | Comparable NLL | Vocab-MAP top-1 | Unconstrained invalid rate | Tokens/s |
|---|---|---|---|---|
| A dense | 8.99 | n/a | 0% | 228k |
| C parallel | 8.45 | 1.65% | 98.9% | 222k |
| F whole-token + bias | 5.20 | 12.97% | 99.6% | 36k (after `embedding_bag`; 5.3k before; 75% GPU cap included) |
| F whole-token, no bias | 6.75 | 3.4% | 87.8% | 37k |

Caveats:
- Arm F with the bias starts with a frequency prior and the others do not. The `token_bias: false` ablation separates the two effects: the whole-token loss alone gives 6.75 (better than A 8.99 and C 8.45), and the prior takes it to 5.20. Both runs are 60 steps and one seed, so the split is a hint only.
- Unconstrained greedy output stays invalid for arm F, because it is never trained per slot. Its value is the whole-token ranking.
- Arm F was about 42× slower than arm C with 32 gathers; with `embedding_bag` it is about 6× slower (36k vs 222k tokens/s).

**The speed problem as math.** `score = Σ_p S[p, b_p(v)]` is a sparse linear map: the flattened logits (length 32 × 257 = 8,224) times a fixed 0/1 matrix `M` of shape `[8224, V]`, where column `v` has exactly 32 ones at `v`'s own (slot, byte) cells. Today it runs as 32 separate gathers, whose backward pass is a slow scatter-add. A sparse-COO matmul computes the same numbers but was about 12× slower on CPU, so the code uses `F.embedding_bag(mode="sum")` over a fixed `[V, 32]` table of cell ids instead: the same sum, fused. `M` is fixed, so it adds no parameters. A test checks that values and gradients equal the 32-gather reference.

---

## 9. What is proven and what is not

- **Proven by tests:** the scoring identities, padding handling, vocab-MAP validity, and the equivalence of the chunked arm F loss with plain cross-entropy.
- **Seen at smoke scale only:** arm F's lower NLL and higher vocab-MAP accuracy than arm C.
- **Not yet shown:** whether arm F matches the dense head at full length, across seeds, with and without the bias. Only the full matched runs can answer that.
