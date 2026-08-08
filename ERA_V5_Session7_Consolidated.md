# ERA V5 — Session 7: Embeddings and Model Internals
### Consolidated reference: official lesson text + live classroom transcript + Q&A

*This document merges three sources into one reference: (1) the official Session 7 lesson page (theschoolofai.in Axiom platform) — the authoritative text, numbers, and formulas; (2) the 162-minute live class transcript — the instructor's analogies, worked examples, and student Q&A that motivate and clarify the material; (3) the interactive widgets described in the lesson, summarized as text. Where the transcript adds color, corrects, or extends the official text, it is marked as **Live classroom notes**. Numbers are quoted exactly as given in the source material.*

---

## 1. What this session is

Four sessions ago this course turned away from the model and went to work on the data, and it stayed there. Session 3 decided what to collect, Session 4 built the pipeline that turns collected bytes into admissible text, Session 5 arranged the clean pools into a mixture and a curriculum, and Session 6 compiled that recipe into an executable stream with a ledger behind it. That arc is finished. What comes out of Session 6 is a packed batch of integer token identifiers, and this session is where those integers finally meet the model.

The place we are standing was named precisely once before. Session 2 ended on the sentence that **the tokenizer's job is complete the moment text has become a sequence of integers**, and that the question of what vector each integer turns into belongs to a different component on the far side of a **clean seam**. This is that component, and this is that class. Session 2 also left an unpaid debt here in writing: the full treatment of **Kronecker factorization**, deferred at the time with the note that it is an embedding-side trick belonging to the class on embeddings and model internals. That debt is the centre of today.

Session 6 closed by pointing forward at the distributed training loop, gradient accumulation, and parallelism — but that pointer was wrong against the calendar. Those belong to Sessions 10–13. There is a reason the model's front and back doors come first: **the two matrices that face the token vocabulary are the largest single tensors in the model**. They are decided before any training system is designed around them, and at the vocabulary and width V5 is heading toward, they are large enough that the decision changes what the rest of the architecture can afford.

**The number that makes the session necessary.** The V4 BrahmicTokenizer carries a vocabulary of **131,072 tokens**, paired with a model width of **8,096**. A dense input embedding table at that shape holds `131,072 × 8,096 = 1,061,158,912 parameters` — a little over a billion parameters, occupying **2.12 GB in bf16**, before a single attention head or feedforward block exists. If the output projection is untied, it is a *second* matrix of the same size. This is the storage problem Session 2 set up and handed forward; the whole of today is spent first making it visible, then designing a way out of it that can be defended.

**The Seam.** Session 6's packed batch hands over five fields per token position: **token ids**, **position ids**, a **loss mask**, **mixture lane/stage metadata**, and **ledger tags** (run, step, shard id). Each field is traced to the model object that consumes it, and to what breaks when it is wrong. The core rule: a token id is *an address, and nothing more* — it says which row to read, not what that row contains. If the tokenizer version changes silently, ids get silently reassigned to different tokens, which is why the tokenizer's hash/version must travel with the data in the shard manifest and the embedding policy record. The tokenizer's contract ends at an integer; everything in this session lives on the far side of that boundary.

**Live classroom notes — course arc and framing.** The instructor opened by noting the last six sessions focused on *what* data to collect, how to clean it, and how to deduplicate it, and that this session marks a much tighter integration between data decisions and model decisions — "both are not isolated, and the direct effect is going to be on the embedding side." The next five to six sessions move from data into understanding the transformer itself, "from being a small model to a big [one]." The instructor also flagged this session as one where two prior students' work (one in tokenizer design, one in embedding layer research) became publishable results, and invited the class to think of today's open problems (Section 14) as similarly paper-worthy.

**Why embedding dimensions aren't individually assignable (pre-2012 feature engineering).** Before 2012, "feature engineering" meant humans manually deciding what each dimension of a representation should mean (e.g., "dimension 1 = gender"). This broke down for genuinely hard problems. Worked example given live: detecting a circle in an image. A hand-designed kernel can't handle every radius, every occlusion, every non-circular ellipse — you'd need infinite variations. When the class was asked what kind of kernel a trained neural network actually learns to do this job, a student (lokesh) correctly guessed: **a spiral** — a single learned kernel that implicitly contains circles, ellipses, and edges at every scale and rotation, something no human would design by hand. This is the direct analogy for why none of the 8,096 embedding dimensions is hand-assigned a meaning like "gender" — the network finds representations no human would think to specify, by looking at millions of examples.

**The "bandwidth" analogy.** Embedding width is compared directly to internet bandwidth: a word represented in 8,096 numbers can encode far more shades of meaning than the same word in 1,024 numbers, the same way an 8 Mbps connection carries more than a 0.5 Mbps one. Illustrated with "how many colors can you name" as a stand-in for representational resolution — more dimensions, more distinguishable shades of meaning. Every sense a word carries across every context it's ever appeared in — *apple* the fruit, *Apple* the company, "apple of my eye"; *bank* the institution, the riverbank, the banking angle of a road — must be compressible into that one fixed-width vector.

---

## 2. The lookup is a gather, not a matrix multiply

Almost every explanation of an embedding layer describes it as a matrix multiplication against a one-hot vector, and that description is **mathematically true and operationally misleading**. Nobody builds a 131,072-element one-hot vector and multiplies it by a billion-parameter matrix to retrieve one row. The operation that actually runs is a **gather**: the integer is an offset, the hardware reads the row at that offset, and the cost is **a memory read rather than an arithmetic one**. Holding the gather picture rather than the matrix-multiply picture is what makes the rest of this session's behaviour predictable instead of surprising.

**The backward pass.** When gradient flows back into an embedding layer, **it does not update the table — it updates the rows that appeared in this batch**, leaving every other row untouched. The operation is a **scatter-add**: it accumulates one gradient contribution per *occurrence* of a token into that token's row, and writes nothing anywhere else. A row that did not appear in the batch receives **exactly zero** gradient this step, and under a plain optimizer does not move at all.

**Coverage vs. magnitude.** A global batch of 256 sequences at 8,192 tokens is **2,097,152 token positions**. It's tempting to think a batch this large leaves most of the table untouched — **it does not**. Nearly every row is gathered on nearly every step, because two million draws against 131,000 rows will find even quite rare tokens. **Coverage is not the problem.** The asymmetry is in *how much* gradient each row receives: natural language is Zipfian, so the most frequent token contributes on the order of **a hundred thousand gradient terms** to its own row in a single step, while a tail token contributes **one, or less than one on average**. The scatter-add sums those contributions, so the effective step size applied to a row scales with how often its token appeared. Across a realistic vocabulary that spread runs to **five or six orders of magnitude**.

**The embedding table is not one object that trains at one rate.** It is a hundred and thirty-one thousand small objects whose effective learning rates are set by the corpus rather than by the optimizer, and the slow end of that range is where the low-resource languages live. At the *microbatch* a single GPU actually processes, coverage genuinely is sparse and tail rows go many steps between visits — which is why gradient accumulation across a large global batch does more work for the embedding layer than for any other tensor in the model.

**Connection to the data side (Session 3/5).** When an English-tuned filter thins the Indic pools, the effect isn't just that the model sees less Indic text — **the Indic rows of the embedding table receive gradient less often**, so they stay closer to random initialization for longer, and a row near initialization injects noise into every sequence it appears in. Protection at the mixture level (Session 5's "always-on lane") is *protection at the gradient level* for these specific rows.

**Interactive widget: gather and scatter-add lab.** Walks one optimizer step through three stages — batch of ids arriving, forward gather (one row read per token), backward scatter-add (writing only into rows that were read; a token appearing four times contributes four terms summed into one update, not four separate updates). A second panel runs this sampling for hundreds/thousands of steps and plots cumulative gradient per row — demonstrating rows end up trained to wildly different degrees not because the optimizer treated them differently (it treated them identically) but because **the corpus did not**.

**Live classroom notes.** The instructor built this with a concrete worked example: if the token "the" appears 67 times in a batch, you do **not** apply 67 separate edits — you accumulate all 67 gradient contributions and apply **one** update per step. This is exactly why updating an embedding table is computationally cheap despite Zipfian skew. When a student (Dattatreya) proposed manually normalizing/scaling up rare words to fix the imbalance, the instructor generalized into a broader design principle: *"If you think there's a solution to a problem and you implement that in a neural network, you'll see it actually becomes worse... the ideal solution is to give a feature to a neural network [and] not implement that solution if it is not required."* Cited real examples of this pattern: batch normalization, learned loss weighting, and KL-divergence-based regularization — build a lever, don't hardcode the fix. This is exactly the philosophy behind adaptive optimizers (Adam, RMSProp): they track how much a parameter has historically been updated and automatically scale future updates down for over-updated parameters and up for under-updated ones, rather than a human hand-tuning per-token learning rates. A student (Yasir) asked a sharper follow-up — not just *how many* updates a parameter gets, but *how large* each individual update is — which the instructor credited as "exactly the right question" that historically motivated RMSProp/Adam's designers.

The instructor also used a **heat-seeking missile** analogy to build intuition for why training uses momentum/smoothed updates rather than reacting instantly to each new gradient: a missile tracking a moving target doesn't just aim at the target's last known position — it accounts for velocity, acceleration, and smooths its corrections, because reacting too sharply to noisy, momentary signals (like a single data example, or a single stock price dip) produces unstable tracking. Adam and its relatives (AdaGrad, RMSProp, AdamW) apply this same smoothing (an exponential moving average) to gradient updates for exactly this reason.

---

## 3. What the table costs, and closing Session 2's loop

Session 2 ended its treatment of vocabulary size by reporting that the embedding table at the 131K setting is already **larger than many complete models**, then handed the question forward. The answer is arithmetic, and **the number that hurts is not the one most people quote**.

**Parameter count (the easy part).** Input table is `V × D`. Output projection (turns the final hidden state into a score for every vocabulary token) is another `D × V`. At V5's reference shape, each is **1.06 billion parameters**, and untied they are **2.12 billion** together. Against a 120B mixture-of-experts model that's a small share of the total; against a 7B dense model it would be **a third of everything**. This is exactly why the same tokenizer decision reads as trivial at one scale and catastrophic at another.

**Memory during training (the number that actually constrains the run).** A parameter trained under AdamW in mixed precision is **not two bytes** — it is two bytes bf16 weight + two bytes bf16 gradient + four bytes fp32 master copy + four bytes each for the two Adam optimizer moments: **sixteen bytes per parameter**. The input table alone occupies `1,061,158,912 × 16 bytes = 16.98 GB` of training state; an untied pair occupies just under **34 GB**. On an 80 GB accelerator that's **two-fifths of the card** consumed by the two matrices that do "the least interesting work in the model" (a lookup and a dot product), before a single attention head has been placed.

**This is why the embedding decision is an architecture decision, not a detail** — it's the pressure that makes every technique in the rest of the session worth its added complexity.

**Interactive widget: Parameter and Memory Budget.** Lets you set vocabulary, width, depth, feedforward multiplier, tying, and precision; computes the input table, output head, and transformer stack as competing shares of total parameters, and converts each into training memory with full optimizer accounting. Sweeping width from 2,048 to 8,096 shows the embedding *share* of parameters collapsing while its *absolute* memory climbs — the central tension the rest of the session resolves. (Live-tested at a larger configuration during the session: 293B total parameters, 96 layers, 12,288 width — embedding share fell to just 1.0% of parameters, yet its share of one accelerator's memory *rose* to 67.1%, because embedding memory scales with width, independent of the parameter-share trend.)

**Live classroom notes.** In-class arithmetic matched the lesson text closely (~17GB stated live vs. 16.98GB in the official text). The instructor stressed this cost is per-batch, and scales further with batch size and number of GPUs — "it's not just about the model weights... in RAM it's going to be a lot." Also flagged: BF16 (brain float 16) was invented at DeepMind; the instructor noted one of its authors, Demis Hassabis, later won a Nobel Prize (in Chemistry, for AlphaFold) — offered as a "why does this format exist" aside, with the promise of a dedicated paper/discussion on precision formats (fp32 vs fp16 vs bf16) in a future session.

---

## 4. The tokenizer and the embedding are one design surface

Vocabulary size isn't handed down from outside — it's a decision made jointly with the embedding system, or the two will fight each other. The dial pulls in **both directions at once**. A **smaller** vocabulary means fewer rows and a smaller table, but raises **fertility** (the same document becomes more tokens), and every extra token costs attention compute at every layer, for the whole of training *and* inference. A **larger** vocabulary lowers fertility and shortens sequences, but grows the table and output head linearly. **Neither end is free; there is no neutral setting.**

For an English-only model this is a mild optimization. For an **India-first** model it is the whole problem, because fertility is **not uniform** across languages. Session 3 measured that a tokenizer splitting Telugu into roughly **three times** as many tokens as English for the same meaning hands Telugu **a third of the effective context and three times the inference bill** — and no amount of extra vocabulary fixes that unless the extra slots are spent on the scripts that need them. Vocabulary is a budget; its allocation across scripts is a **sovereign decision** — precisely what the BrahmicTokenizer retrofit addressed.

**Real numbers from the fertility/cost lab.** The combined cost (attention-side, falling as vocabulary grows; parameter-side, rising linearly) has a minimum that **moves with the corpus mix**: English-only puts the minimum near **53K**; the V5 mix moves it to about **101K**; an Indic-heavy mix pushes it past **113K**. This happens because fertility decays toward a floor of one token per word — a language's *reducible* headroom is what extra vocabulary can actually buy, and English has almost none while Tamil has a great deal.

**Live classroom notes.** The instructor reinforced this with a **GPU-as-truck analogy**: "a GPU is like a truck — it can carry 4,000 toothbrushes, but the whole truck still has to go." Vocabulary size has to be a clean multiplier of powers of two for GPU memory alignment; even if the fertility/cost math suggests something like 101K–113K as "optimal," the actual number gets padded up to a clean value like 131,072 anyway, because GPUs perform worse with misaligned memory blocks — "even if you don't use it fully, it will haunt you later." The instructor was explicit that these tradeoffs can't be fully resolved analytically: "we don't have data[set] only... after we have data[,] a tokenizer, we can merge both numbers, do a short compute, and figure out the right structure" — i.e., some of this is settled empirically with real proxy runs, not formula alone.

---

## 5. Weight tying

The cheapest available saving: since the input embedding and output projection are matrices of identical shape facing the same vocabulary, use **one matrix for both jobs**. Weight tying sets the output projection to the transpose of the input embedding, removing an entire billion-parameter tensor and about **17 GB of training state** at V5's reference shape. It also has a conceptual argument: since both matrices live in the same space, tying forces the geometry the model *reads* tokens with to be the same geometry it *predicts* tokens with.

**The argument against:** the two jobs aren't the same job. At the input, a row should be a good starting representation for a token about to be processed — rewarding *what the token means*. At the output, a row should produce a score that separates this token from every other plausible next token — rewarding *what distinguishes it from its competitors*. These objectives agree for most of the vocabulary and disagree exactly where it matters: frequent function words whose meaning is thin but whose prediction is constant.

**The empirical rule.** Tying pays when the token-facing matrices are a large share of the model (the saving is real, and regularization helps a small model that would otherwise overfit them). It stops paying as the model grows, because the saving becomes a rounding error while the geometric constraint doesn't. **GPT-2 tied, Gemma ties; Llama-2 at 7B does not.** The threshold isn't a magic number — it's the point where the embedding share of total parameters falls into the **low single digits**, and **V5 is well past it**.

**Live classroom notes.** The instructor gave the same empirical rule with a concrete cutoff: tying is worth it under roughly **3 billion total parameters**; above that, the RAM/parameter savings are a rounding error, while untying lets input and output develop the "slightly different" representations associated with stronger models ("differences that make a model behave like Opus or Gemini"). A student (Sachin) tried to argue weight tying is intuitively wrong because the output of the model is a "transformed contextual vector," different in kind from the raw input vector — the instructor rejected this as not a real counter-argument absent a stated claim being contradicted, and redirected to ask questions about material actually covered. Another student (Abhishek) asked about Gemma (~4B, described in-class as a "small model") using tying despite being close to the boundary — consistent with the stated rule that tying is common for models in the few-billion range and below.

---

## 6. Factorized embeddings, the on-ramp

Tying halves the problem. **Factorization** attacks what's left. The simplest form: the row a token is looked up as doesn't have to be as wide as the model. Look the token up in a *narrow* table, then project the narrow vector up to model width with a matrix shared by every token. The narrow table is `V × r`, the projection is `r × D`, so parameter count becomes `V·r + r·D` in place of `V·D`. At V5's shape with rank `r = 512`: `67.1M + 4.1M ≈ 71 million parameters` in place of 1.06 billion — a **~93% reduction**. This is the scheme **ALBERT** used.

**The cost is a rank bottleneck, and it is not soft.** Whatever the projection does, the set of vectors the embedding layer can produce is confined to an `r`-dimensional subspace of the `D`-dimensional space — because every output is the same `r × D` matrix applied to some narrow vector. The layer has `V` rows but at most `r` *directions*. If `r` is generous, invisible; if aggressive, the model is asked to represent 130,000 distinct tokens inside a few hundred directions, and **tokens start having to share**.

**The general lesson stated explicitly:** this is the honest shape of every compression technique in the session. You are not getting the same table for fewer parameters — you're getting a *structured* table, and the question is always whether the imposed structure happens to match the structure the data actually has.

**Interactive widget: factorized embedding builder.** Constructs a reference table with a known spectrum so reconstruction error of a rank-`r` approximation is exact rather than estimated; plots retained "energy" against rank alongside parameter savings, showing the knee where savings are large and error small, versus the region past it where every further saved parameter costs real capacity.

**Live classroom notes — rank, spectral decay, and the seed of LoRA.** The instructor built the concept of *rank* using a Spotify-vs-high-bitrate audio analogy and a JPEG-compression-ratio analogy: not every dimension of a representation is equally important, and you can often compress heavily with barely perceptible loss. Mathematically, `100×100` can be rewritten as `100×10` then `10×100` — cheaper, and nearly the same output. Critically, **how much you can compress isn't fixed — it changes as training progresses.** Early in training, useful "effective rank" is high (many dimensions matter, because the model hasn't specialized yet); as training matures, the *spectral decay* (how fast the signal drops off across dimensions) steepens, meaning far fewer dimensions end up mattering. This exact phenomenon — a fully trained network's meaningful update space collapsing to a small subspace — is explicitly named as **the intuition underlying LoRA** (Low-Rank Adaptation): find the small number of dimensions that matter, and only train those.

One resistant exception flagged live: **the output head resists low-rank compression far more than any other layer** — "that thing is never low rank, even after deployment... that is the most important layer, and that is one big issue... you can't lower rank those things" — because it must retain the ability to distinguish all 131,072 possible output tokens, unlike interior layers that can specialize on narrower subspaces.

---

## 7. Kronecker factorization

The technique Session 2 promised, and the one this lab releases. A dense table keeps one row per token. **Kronecker embeddings keep no rows at all** — they build each token's row out of the token's own bytes, in three steps.

**Step 1 — read the bytes.** Every token is a sequence of UTF-8 bytes ("the" is three bytes). Nothing is learned here.

**Step 2 — mark a grid.** Picture a grid with **256 rows** (one per possible byte value) and **32 columns** (one per byte position). For every byte in the token, mark the cell at (its value, its position). A three-byte token marks three cells. Flatten the grid and you have a vector of `256 × 32 = 8,192` numbers, almost all zero — the token's fixed, never-trained, always-identical **code**. Marking one cell is exactly a Kronecker product of two one-hot vectors (byte-value one-hot ⊗ position one-hot) — where the technique's name comes from:

`κ(b) = (1/√L) · vec( Σ over positions p of c[byte_p] ⊗ p[position_p] )`

The `1/√L` term divides by the square root of the token's byte length so short and long tokens land at a comparable scale; the result is then z-normalized.

**Step 3 — project.** One shared `Linear(8192, d_model)` turns the fixed code into the vector the model actually uses. **That projection is the only thing in the entire input path that is learned.**

**The parameter count that matters:** `8,192 × 8,096 = 66,322,432 parameters`, against a dense table's `1,061,158,912` — a **93.75% reduction**. Notice what's *missing*: **the vocabulary size**. There is no `V` anywhere in that multiplication. The input path costs the same whether the vocabulary holds thirty thousand tokens or five hundred thousand, because the grid always has 256 rows and 32 columns.

**What you gain:**
- **Cost stops depending on vocabulary.** Adding tokens is free on the input side — inverting Section 4's whole trade-off.
- **Unseen tokens still work.** A token the model never trained on still has bytes, so it gets a sensible code rather than an untrained random row.
- **Similar spellings start out similar.** "train," "training," and "trainer" share most of their bytes, so they begin near each other instead of unrelated.

**What you pay:**
- **The layer needs the token text.** It takes a tokenizer and keeps a byte buffer per token, reaching back across the seam Session 2 drew — a deliberate trade, not an accident, and why the constructor carries both `vocab_size` and `tokenizer`.
- **The code is frozen by construction.** The codec never learns anything. If two tokens produce the same code, no amount of training can pull them apart (Section 8 is about exactly when this happens).

**Terminology clarification.** A different, unrelated family in the literature also factorizes an embedding *table* as `A ⊗ B` (two *learned* matrices whose Kronecker product reconstructs the table's shape). **This is not that.** Here, the Kronecker product is between a byte value and a byte position, both factors are *one-hot* (not learned), and the full table is never reconstructed at all, even implicitly.

**Interactive widgets:**
- *Kronecker Microscope*: type any token in any script and watch the three steps run live — bytes read, grid marked, projected through the shared matrix. Parameter count updates live and doesn't move when vocabulary size changes. Try "training," भारत, and తెలుగు to see byte-per-character cost change from 1 to 3.
- *Module walkthrough*: shows which lines create a fixed, never-trained buffer (byte buffer, codec table — sized by vocabulary but zero trainable weight) vs. the single trainable `Parameter` (the projection — never sized by vocabulary). Forward pass is two lines. **Architectural point:** the module's entire visible behavior is `[B,T]` integers in, `[B,T,D]` floats out — how the row is manufactured internally is fully private, which is why this aggressive a compression decision can be made, measured, and reversed without touching attention, feedforward, loss, or the dataloader.

**Live classroom notes — origin story and philosophy (not in the official text).** This is the instructor's own prior published work, already shipped in V4. Motivating question: *"Can I create the 8,096-number vector for a token without keeping a 131,072 × 8,096 table at all?"* V4's actual implementation used `d_model = 4,096` (not 8,096), so its real projection was an 8,192→4,096 matrix, with a **~66,560**-parameter fixed (non-trainable) byte/position basis.

**Why 32 bytes specifically — the real reason, not stated in the official text:** `8,096 ÷ 32 ≈ 253 ≈ 256`, and UTF-8 has exactly 256 possible byte values. The window size is chosen so the flattened grid structure lines up cleanly against the model's own width, not picked arbitrarily. Doubling to 64 bytes would roughly double the required depth of the resulting representation — "a bigger cost."

**The philosophical justification (the heart of the idea):** the first few transformer layers' actual job is to convert a raw token signal into *contextual* meaning — meaning that changes anyway (typos, novel usage, out-of-context slang: "red is a beautiful word at Christmas, a bad word to hear in a hospital"). So why force the embedding table to permanently store a fixed "meaning" for each token, when meaning is inherently unstable and context-dependent? Instead, make the *raw* representation a deterministic function of spelling — cheap and consistent — and let the first trainable layer do the semantic work it was effectively always doing anyway. If the raw code is already good enough for some token, the network can learn to pass it through near-identity — the same "give it a mechanism and let it decide" philosophy from Section 2.

**Additional claimed benefits (explicitly caveated as unproven at scale):** robustness to typos, not just unseen tokens — "Amwrica" misspelled still produces a coherent, structured code, unlike an undefined row in a lookup table. Because the representation doesn't depend on a fixed BPE vocabulary, arbitrarily long phrases (e.g., "United States of America") could in principle collapse into a single token, since the encoder only needs bytes, not a pre-registered vocabulary entry — something the instructor states no standard tokenizer (e.g., tiktoken) can do. **Honest caveat, stated directly:** these benefits were validated only at small scale (~131M parameters); "I couldn't publish it... I have tested it, but it's not conclusive enough" at production scale.

**A student (Nitin) asked about the inspiration** for the fixed byte-to-vector matrix. The instructor reframed the real motivating problems: (1) LLMs don't see the spelling of a word, only an opaque integer; (2) the model has to spend capacity learning what an arbitrary integer even is; (3) sometimes multi-word phrases should be single concepts, and the current approach can't flexibly do that. The claim (from prior published work) is that the first layers of a transformer effectively re-derive "meaning" from a raw signal anyway — so making that raw signal deterministic and spelling-based, rather than an opaque learned integer identity, should be a strict improvement, testable but not yet conclusively proven at scale.

---

## 8. The thirty-two byte budget

The grid has **32 columns**, meaning only the first 32 bytes of a token are ever seen: `L = min(len(byte_seq), pos_dim)`. Bytes past position 32 are dropped. For English this is generous — ASCII spends **one byte per character**, so 32 bytes is 32 characters, more than almost any token needs.

**For Indic scripts it is not generous at all.** Devanagari, Telugu, Tamil, Bengali and their neighbours sit in a Unicode region UTF-8 encodes in **three bytes per character** — the same window now holds **ten characters**, not thirty-two. It gets tighter still: a conjunct such as क्ष is not one character to Unicode — it is **three code points** (क, halant, ष). At three bytes each, that single visual character costs **nine of the thirty-two bytes**. A word carrying three or four conjuncts has spent the entire budget before it finishes.

**And the failure is silent.** Two tokens agreeing on their first 32 bytes produce **exactly the same code**, therefore exactly the same embedding vector, **permanently**. The projection cannot separate them because it is never shown a difference. Nothing raises an error or a warning — the model simply cannot tell those two tokens apart, ever. **This is the sovereign risk in the technique, and unlike most such risks it is a number rather than an argument.**

**The resolution is empirical, not automatic.** The question isn't whether Kronecker embeddings save memory (they plainly do) — it's whether `pos_dim = 32` is the right window for the scripts that matter, answered by measurement: encode the real V5 vocabulary and count collisions per script. Raising `pos_dim` to 64 doubles `D` and doubles the projection to roughly **133M parameters** — still **eight times smaller** than a dense table. If the collision count justifies it, buy it. **This is the actual assignment task for anyone extending Section 8's measurement.**

**Interactive widget: byte budget lab.** Encodes real words live; shows how much of a word survives the window vs. is dropped, with the limit drawn as a line. Demonstrates two genuinely different Hindi words — **अंतर्राष्ट्रीयकरण** and **अंतर्राष्ट्रीयता** — colliding at the shipped `pos_dim = 32`: identical codes, identical vectors, indistinguishable to the model forever. Widening the window to 48 separates them.

---

## 9. The V4 scar: a frozen input path meets a mixture shift

A real incident, reported by Session 5 as a warning about mixture transitions and revisited here with a sharper diagnosis. A sudden increase in the Hindi share of the training mixture, arriving against **frozen** embeddings, drove the gradient norm up by roughly **one hundred and fifty times** over a short stretch. Session 5's original read ("don't change the blend too abruptly") is true but not the deepest lesson.

**The embedding layer is the only place in the model where the token distribution meets continuous computation.** Every distributional fact about the corpus — which tokens are common, which scripts are present, what the mixture currently is — enters the model through that one adapter. When the mixture changes, the statistics arriving there change, and the natural response is for the adapter to move. **If it is frozen it cannot, and the adjustment has to happen somewhere** — it happens in the layers above, which now have to absorb a shifted input distribution using parameters tuned for the old one, showing up as large updates that propagate. **This is not a learning-rate problem and does not have a learning-rate fix. It is an adaptation-boundary problem**: degrees of freedom were removed at the exact point where the change enters, and the consequence surfaced somewhere else instead.

**Kronecker embeddings sharpen this considerably — the part V4 did not anticipate.** A dense table has a *billion* degrees of freedom to absorb a distributional change, moving just the rows that need to move and leaving the rest alone. A Kronecker input path has **the projection and nothing else** — the codec is fixed by construction with zero parameters, so every token in the vocabulary adapts through one shared matrix, or does not adapt at all. There is no way to adjust how Hindi is represented without touching every other token simultaneously. **A compressed input path is a less capable adapter by construction**, which makes the case for keeping the projection trainable **stronger, not weaker**, than for a dense table. **Compressing the embedding and freezing it are two decisions that each look locally reasonable and are jointly dangerous.**

**The operational conclusions V5 actually adopts:**
- **The projection and factors stay trainable throughout.** The codec is already frozen by construction; freezing the projection too would leave the input path with *no adaptive capacity at all*.
- **Freezing, if it ever happens, is scheduled and logged**, never left on as a residue of an experiment.
- **Mixture transitions get the warmup band** Session 5 already mandated.
- **Gradient norms of the layers immediately above the embedding are monitored as a leading indicator** — the global norm averages the signal away and hides this.
- The ledger acquires a new field: **`embedding_policy_id`**, recording the embedding type, `char_dim`/`pos_dim`, the tokenizer hash the byte buffers were built from, the projection's trainable/frozen state, unfreeze schedule, position policy, and tying decision. The tokenizer hash matters especially here: **change the tokenizer and every token's bytes change, so every code changes, and the projection is now trained against a codec that no longer exists.** A checkpoint that cannot state what its input path was doing cannot be meaningfully compared against another one.

**Interactive widget: frozen input path lab.** Trains a real small language model over a two-domain vocabulary in-browser, with genuine forward passes, real cross-entropy, and hand-written gradient descent, on a stream whose mixture shifts partway through. Freezing the embedding causes a visible jump in both loss and the gradient norm of the layer above at the transition; leaving it trainable absorbs the same transition smoothly. A warmup control spreads the shift and shrinks the spike.

**Live classroom notes.** The instructor's own analogy for the shock: like "listening to Donald Trump all day, then suddenly Sachin Tendulkar starts speaking" — an abrupt register/vocabulary shift the shared 8k×8k (in V4's real numbers, 4k×8k) projection wasn't prepared for. The practical mitigation described live matches the ledger conclusions above: warmup steps after mixture changes, and *slow, gradual* curriculum transitions rather than sharp cuts — "if I change [the mixture] quickly, there's a big jump; if I change it slowly, it's better for the model." Learning rate also needs adjustment around these transition points.

---

## 10. Position: the model reads a set

Everything so far answered the first of two questions the model asks about an incoming token: **what the token is**. The second question is **where it is**, and the reason it needs a separate answer is **structural, not incidental**.

Attention computes a score between every pair of positions from their query and key vectors and mixes values accordingly. **Nothing in that computation refers to the order of the sequence.** Permute the input tokens and every pairwise score permutes with them, and the output permutes correspondingly — the mechanism is **equivariant to permutation**, a precise way of saying it cannot tell **"dog bites man" from "man bites dog."** **The transformer does not read a sequence. It reads a set**, and order has to be supplied to it deliberately.

**Proof, not just assertion (Session 2's experiment).** Two tiny models were trained side by side on a task rigged so every sequence appeared once per label under a two-token swap. The **token-only** model was pinned at chance — the two swapped cases were *literally identical inputs* to it. The **token-plus-position** model learned the rule correctly. That experiment is the foundation for this section, and the open question is what the position signal should actually be made of.

**Live classroom notes.** The instructor's live demonstration: a simple two-input, one-weight-each computation (`4 × w1 + 5 × w2`) produces the same output regardless of whether "a" or "b" arrives first at those input slots — mathematically, order isn't preserved unless something encodes it. When a student (Vardhan) asked why order matters if tokens are "presented in sequence anyway," the instructor clarified that all tokens are fed to the model *simultaneously*, not one at a time the way a human reads — so there's no implicit "earlier" without an explicit signal. One proposed intuition (from the instructor): borrow from complex/imaginary numbers — encode order as a second, orthogonal "axis" added to the real-valued token representation, so `4` and `4 + 2i` are distinguishable even though their "real" component is identical — a conceptual bridge to how position embeddings get added as a second signal alongside the token vector.

---

## 11. The absolute table, and the wall at max_position

The simplest fix mirrors the token table: a second lookup table indexed by position, added to the token's row before the stack sees either:

```python
token_embedding = nn.Embedding(vocab_size, d_model)
position_embedding = nn.Embedding(max_position, d_model)
x = token_embedding(token_ids) + position_embedding(torch.arange(T))
```

This is what **GPT-2** did, and it works. As a parameter block it's small enough to ignore next to the vocabulary table, since `max_position` is thousands, not hundreds of thousands — **so the objection to it is not cost.**

**The objection is structural.** A position table is gathered and scatter-added exactly like a token table, so row `t` learns only from steps where some sequence had a token at position `t`. If trained with a maximum position of **4,096**, rows 0–4,095 are trained and **row 7,000 does not exist**. Extend context at inference and there's no row to fetch; pre-allocate rows and never train them, and they sit at random initialization, injecting noise exactly where you extended the model to handle. **The table cannot extrapolate, and the reason is not subtle or fixable by better training** — there is no signal in the parameters connecting row 4,095 to row 4,096, because they were only ever *independent rows in a lookup table*.

**This is a hard wall** — the wall the entire modern positional-encoding literature exists to get past. The way past it is to **stop storing position and start computing it**, so position enters through a function of `t` defined for every `t`, rather than a row that exists only for values seen in training. The original Transformer's sinusoidal encoding was the first version of this idea; what the field converged on afterward (and what V5 will actually use) is Session 8's subject.

**Interactive widget: absolute position table, trained live.** A task whose answer depends on token position is trained on positions 0–7 with real gradients, learning the rule to near-perfect accuracy inside that range. Evaluated at positions 8–15, accuracy falls to chance — a coverage strip shows those rows never appeared in a batch, so the scatter-add never wrote to them; they hold their initialization. **The cliff at the trained boundary is measured, not illustrated.**

**Live classroom notes.** The instructor reinforced this is unconditional: "it doesn't matter how much you change the learning rate, those numbers are never going to be touched" — if a model needs to handle 1 million tokens of context, it must have actually trained on sequences using those specific positions; there's no shortcut around this with an absolute learned table.

---

## 12. The families, and what Session 8 builds

A map, not a mechanism — the mechanisms are next week's material. There are broadly **four places a position signal can be injected:**

| # | Family | Where it enters | What it's made of | Behavior past trained length |
|---|---|---|---|---|
| 1 | **Absolute learned** | Stored, added at the input | One row per position, trained by gather/scatter-add exactly like the token table | **Hard wall** — undefined/never-trained past max length |
| 2 | **Sinusoidal** | Computed, added at the input | A deterministic function of `t`, no learned parameters | Defined everywhere, but the model has never had to *use* the far part of the function |
| 3 | **Rotary (RoPE)** | Applied inside attention, to queries and keys | A computed rotation depending on the *offset* between positions | Degrades gracefully — "where the field has settled" |
| 4 | **Attention bias (ALiBi)** | Applied directly to attention scores | A fixed, distance-dependent penalty (one slope per head) | Extrapolates gracefully by design — the bias is defined for any distance |

**The trade running across all four:** how much the model is told vs. how much it has to infer. A **stored** signal is maximally expressive within its trained range and **useless outside it**. A **computed** signal generalizes by construction but **constrains what can be represented**. This is **the same trade this session made twice already** — once when a dense table became a factorized one, once when a factorized table became a Kronecker one. Not a coincidence: it's what happens every time structure is substituted for stored parameters.

Session 8 builds rotary embeddings and the scaling schemes that extend them, covers attention-side variants that change which pairs get computed at all, and treats long-context extension as its own engineering problem rather than a bigger config number.

**Interactive widget: position family map.** Shows where each scheme injects its signal, what it's made of, parameter/compute cost, and extension behavior; selecting a family traces the signal through the model to where attention consumes it.

**Live classroom notes.** The instructor was blunt that most public diagrams of transformer architecture (showing "positional encoding" added at the input, next to embeddings) depict the **obsolete** method — "nobody uses this [anymore]." RoPE is injected directly into the attention score computation (applied to query/key vectors, not summed with the token embedding at all) — cheaper and faster. ALiBi is injected even later, directly at the softmax/output stage, making it cheaper still.

---

## 13. The V5 embedding decision

*(This section states the actual, committed design choices for V5 — not just the space of options.)*

- **Vocabulary stays in the 131K class**, inherited from BrahmicTokenizer work, chosen against fertility across target scripts *and* the parameter arithmetic of Section 3 together — not either alone.
- **Input path is the released Kronecker byte codec plus a trainable projection**, with a **dense table kept as the control arm** (i.e., trained in parallel as a baseline for comparison, not discarded). The `pos_dim` window is chosen **by measurement, not by default**, starting from the shipped value of 32.
- **The projection is trainable throughout.** The codec is already frozen by construction; freezing the projection too would leave the input path with no adaptive capacity at all. Freezing, if it happens, is a scheduled and logged decision, never a leftover from an experiment.
- **Output head is untied**, because at V5's scale the saving from tying is small, the constraint on geometry is not, and a structured (Kronecker) input path and a dense output head aren't naturally the same object anyway.
- **The byte window is an architectural parameter, not an inherited default.** Since 32 bytes is 32 English characters but only 10 Indic ones, `pos_dim` is set by a **collision count measured on the real V5 vocabulary, per script.**
- **Position policy is deferred to Session 8 by design**, with the **absolute table explicitly ruled out** for the long-context target, on the evidence from Section 11.
- All of this is written into the ledger under **`embedding_policy_id`**, so any checkpoint can answer what its input path was doing when it was written.

**Every number above is a hypothesis.** Session 5 established the standard that "a proportion is an opinion until a proxy run has tested it" — and there's no reason an architecture decision should be held to a lower standard than a mixture decision. The byte window, projection width, and tying decision all get **proxy runs at the one-billion and three-billion scale** before being trusted at full scale.

**Interactive widget: V5 embedding design board.** Assembles the full policy (vocabulary, width, input path, rank, tying, freeze policy) and computes parameter count, training memory with full optimizer accounting, compression vs. dense, and degrees of freedom per token — from real arithmetic, not a score table. Failing gates are named with the specific number that failed them. Emits the `embedding_policy_id` record as JSON — the object the ledger stores, and what the assignment asks students to defend.

---

## 14. The assignment — toward "Kronecker Embedding V2"

Framed explicitly by the instructor as a potential joint research paper: *"This is the direction I am planning to write Kronecker Embedding V2, and these are the ideas... if you can [solve one], then we both will write a paper — and we can write a paper alone without taking my name [if you prefer]."* Pick **one** problem (each is separate — don't mix them), work out a solution (with agent help, including training a small transformer model to prove it), and submit code plus a README (a webapp with graphs/animations is welcome; a plain README is also fine).

**The five problems, verbatim in substance:**

1. **Mathematical structure in embeddings.** What if embeddings could store mathematical structure — such that the embedding of "9" has the meaning of 9 stored in absolute mathematical terms, so that computing `embedding(9) + embedding(9)` yields something whose mathematical-meaning component is literally `embedding(18)`, and `9 × 9` similarly yields `81`? How far can this be pushed — could whole mathematics and its operations be described this way? (Space for alphabets/words is preserved by using the existing 32 slots and appending new ones for this concept.)

2. **Multimodal extension.** What is the natural extension of Kronecker embeddings to represent images and audio as well — requiring preprocessing of image and audio patches — such that all three modalities (text, image, audio) could share one embedding scheme?

3. **Dynamic window size.** Kronecker currently reserves all 32 position slots for every token, even short ones like "a" — wasted space — while still hard-capping (and silently colliding) tokens longer than 32 bytes. How can the window be made dynamic without forcing truncation?

4. **A real Fourier alternative.** Why not represent each character as a Fourier wave and sum characters together to build a word's representation, instead of one-hot grid marks?

5. **Reversibility.** Kronecker is forward-deterministic (a word always produces the same embedding) — how can this be made reversible (an embedding maps back to the same word)? If solved, the final output head could be eliminated entirely, and the vocabulary could scale to 1M+ tokens with no per-token parameter cost on either side.

**Submission requirements:** state which problem you're solving, prove your solution works (train a real small model), submit code and a README.

**Live classroom notes — the color not in the official text.**

*Problem 4 was the instructor's original idea, not problem 5's Kronecker approach.* "My first idea was Fourier. I don't know if you know Fourier... I love this concept... I couldn't solve that, so I ended up with [Kronecker]... the Kronecker is actually a sub-idea." The Fourier motivation: representing anything (the example given live was a Fourier reconstruction of a portrait) as a sum of simple frequencies, and whether the same idea could represent characters/words as summed waves. Framed as the hardest problem on the list, but one that would **automatically solve problems 1, 2, and 3** if cracked. Problem 5 (reversibility) is explicitly a **separate, unrelated problem** to problems 1–4.

*On problem 5, the instructor's own attempted direction and why it stalled:* the motivation for reversibility is speed — if the mapping could be inverted, the model could predict a whole 32-character span in one shot instead of one token at a time, "we can speed up neural network or LLM like anything." The blocker: a trained neural network predicts an *approximation* of the exact fixed target vector (e.g., predicting `0.31, 0.18, ...` when the true deterministic code is `0.30, 0.20, ...`) — close, but never exactly equal, especially early in training when it's not even close. The instructor's own attempted fix, unresolved: borrow the **KL divergence** / **VAE (variational autoencoder)** approach — instead of predicting a single exact point in embedding space, predict a *distribution* (mean ± variance — "not a point, a point cloud"), so that approximate predictions are tolerated by construction rather than being a hard failure. This direction was explicitly left unsolved by the instructor.

*A student (Nitin) asked why not simply use cosine similarity* to find the nearest real token to an imperfect prediction, as a way around exact-match reversibility. **The instructor's direct answer:** cosine similarity only works well once vectors are already close together — i.e., after a model is reasonably well-trained. At random initialization, in high-dimensional space, essentially all vectors are roughly equidistant from one another, so cosine similarity gives no useful signal for bootstrapping an untrained or early-training model. (A related aside: in *low*-dimensional space cosine similarity remains informative even for less-trained representations; the failure mode is specific to high dimensions with near-random vectors.)

**Priority order given live:** solving #4 (Fourier) would automatically resolve #1–#3; #5 (reversibility) stands alone as a different, arguably even bigger unlock — eliminating the output head entirely.

---

## Appendix A — Key formulas and numbers

| Quantity | Value |
|---|---|
| V5/V4 vocabulary size | **131,072** tokens (BrahmicTokenizer) |
| V5 reference model width (`d_model`) | **8,096** (V4 actually used **4,096**) |
| Dense embedding table (input) | `131,072 × 8,096 = 1,061,158,912` parameters ≈ **1.06B**, **2.12 GB** in bf16 |
| Untied dense input + output | **2.12B parameters** total |
| Bytes per trained parameter (AdamW mixed precision) | **16 bytes** (2 bf16 weight + 2 bf16 grad + 4 fp32 master + 4+4 Adam moments) |
| Dense input table training memory | **16.98 GB** |
| Dense untied pair training memory | **~34 GB** (~42% of an 80GB accelerator) |
| ALBERT-style factorized embedding (rank 512) | `V·r + r·D` = 67.1M + 4.1M ≈ **71M params** (~93% reduction) |
| Kronecker grid | **256 rows** (byte values) × **32 columns** (byte positions) = 8,192-length flattened code |
| Kronecker projection (V5 shape) | `8,192 × 8,096 = 66,322,432` parameters (**93.75% reduction** vs. dense) |
| Kronecker projection (V4 actual shape) | 8,192 → 4,096; ~66,560 fixed non-trainable basis parameters |
| Kronecker `pos_dim = 64` alternative | ~133M parameters (still ~8x smaller than dense) |
| Reason `pos_dim = 32` | `8,096 ÷ 32 ≈ 253 ≈ 256` — matches UTF-8's 256 byte values |
| Global batch (example used) | 256 sequences × 8,192 tokens = **2,097,152 token positions** |
| Gradient contribution spread (Zipf) | Most frequent token: ~100,000 terms/step; tail token: ≤1 — **5–6 orders of magnitude** |
| V4 scar: gradient norm spike | **~150x** on a sudden Hindi-share mixture increase against frozen embeddings |
| Weight-tying threshold (rule of thumb) | Pays off under **~3B total parameters**; V5 is well past it |
| Fertility/cost-minimizing vocabulary size | English-only ≈ **53K**; V5 mix ≈ **101K**; Indic-heavy ≈ **113K** |
| Absolute position table | `nn.Embedding(max_position, d_model)`, added to token embedding — hard wall past `max_position` |

## Appendix B — Glossary

- **Gather**: reading one row from a table using an integer as an offset — a memory operation, not arithmetic.
- **Scatter-add**: the backward-pass mirror of a gather; accumulates one gradient contribution per occurrence of a token into that token's row, leaving all other rows untouched.
- **Zipf's law / Zipfian distribution**: frequency of an item is inversely proportional to its rank (frequency ∝ 1/rank^s); explains why a small number of tokens dominate usage while most of the vocabulary is rare.
- **Weight tying**: using the same matrix (transposed) for both the input embedding and the output projection.
- **Rank / low-rank factorization**: representing a large matrix as the product of two smaller matrices, exploiting the fact that not all dimensions of a representation carry independent signal; the basis for both ALBERT-style embedding factorization and LoRA.
- **Spectral decay**: how quickly the "importance" of successive dimensions drops off in a trained representation; steepens as training matures, enabling more aggressive compression later in training than early.
- **Kronecker embedding**: constructing a token's vector from a fixed (untrained) grid built from the token's own UTF-8 bytes and their positions, passed through one shared trainable projection — replacing a per-token lookup table entirely.
- **Kronecker product** (in this context): the outer-product-style combination of a one-hot byte-value vector and a one-hot position vector, used to mark one cell of the byte/position grid.
- **`embedding_policy_id`**: the ledger field recording every decision about how the embedding layer was built and trained for a given checkpoint (embedding type, `char_dim`/`pos_dim`, tokenizer hash, trainable/frozen state, tying decision, position policy).
- **Adaptation-boundary problem**: a failure mode where the point in the model meant to absorb a distributional change has had its degrees of freedom removed (e.g., via freezing), forcing the adjustment to surface elsewhere in the network instead.
- **RoPE (Rotary Position Embedding)**: a position-encoding scheme applied inside attention (to query/key vectors) rather than added at the input; degrades gracefully past the trained context length.
- **ALiBi (Attention with Linear Biases)**: a position-encoding scheme that adds a fixed, distance-dependent penalty directly to attention scores; extrapolates gracefully by design.
- **Fertility**: the average number of tokens a tokenizer needs to represent a given unit of text (e.g., a word); lower is more efficient.

## Appendix C — Assignment quick reference

| # | Problem | Difficulty/scope note (from live class) |
|---|---|---|
| 1 | Store mathematical structure in embeddings (9+9→18, 9×9→81) | Would be automatically solved by #4 |
| 2 | Extend Kronecker to images and audio | Would be automatically solved by #4 |
| 3 | Make the 32-byte window dynamic, not wasteful/truncating | Would be automatically solved by #4 |
| 4 | A real Fourier alternative to Kronecker | The instructor's original, unsolved idea; hardest, most encompassing |
| 5 | Make Kronecker embeddings reversible (embedding → token) | Separate problem from 1–4; would eliminate the output head entirely; instructor's attempted direction was KL-divergence/VAE-style distributional prediction, unresolved |
