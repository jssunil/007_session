"""Use-case tests adapted from classmates' Session 7 submissions.

CPU only, no training. Each test states which question it answers.
Run with -s to see the printed tables.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kronecker_v2.codec import KroneckerCodec
from kronecker_v2.config import PAD_IDX
from kronecker_v2.metrics import (
    Evaluator, active_slot_accuracy, bytegrid_token_scores, is_valid_utf8,
)

P = 32
LATIN = [f"word{i:03d}".encode() for i in range(150)]
DEVA = [("शब्द" + chr(0x0915 + i % 30) + chr(0x0915 + i // 30)).encode() for i in range(150)]
VOCAB = LATIN + DEVA
SCRIPTS = ["latin"] * len(LATIN) + ["devanagari"] * len(DEVA)
V = len(VOCAB)


def make_codec() -> KroneckerCodec:
    assert len(set(VOCAB)) == V, "test vocabulary must be collision-free"
    return KroneckerCodec(VOCAB, pos_dim=P)


def noisy_slot_logits(codec, targets, noise, seed=0):
    """Peaked logits on the target bytes plus Gaussian noise."""
    g = torch.Generator().manual_seed(seed)
    tb, _ = codec.target_bytes(targets)
    logits = torch.randn(len(targets), P, 257, generator=g) * noise
    logits.scatter_add_(2, tb.unsqueeze(-1), torch.full(tb.shape + (1,), 4.0))
    return logits


# 1. Free-decoding validity (Aneesha) ------------------------------------

def test_free_decoding_validity_vs_vocab_map():
    """Per-slot argmax can spell strings that are not tokens (even invalid
    UTF-8); the vocab-MAP decode never can."""
    torch.manual_seed(0)
    codec = make_codec()
    ev = Evaluator(codec, VOCAB, SCRIPTS, np.arange(V), torch.device("cpu"))
    targets = torch.arange(V)
    rows = []
    for noise in [0.5, 2.0, 4.0, 6.0]:
        logits = noisy_slot_logits(codec, targets, noise)
        decoded = codec.decode(logits.argmax(-1))
        utf8 = np.mean([is_valid_utf8(b) for b in decoded])
        real = np.mean([b in ev.byte_to_id for b in decoded])
        scores = bytegrid_token_scores(logits, codec.byte_buffer)
        pick = scores.argmax(dim=1)
        map_acc = (pick == targets).float().mean().item()
        argmax_acc = np.mean([ev.byte_to_id.get(b) == t for b, t in zip(decoded, targets.tolist())])
        rows.append((noise, utf8, real, argmax_acc, map_acc))
        assert (0 <= pick).all() and (pick < V).all()           # always a real token
        assert map_acc >= argmax_acc - 1e-9                      # MAP never worse
    print("\nnoise | valid-utf8 | real-token | argmax-acc | vocab-MAP-acc")
    for r in rows:
        print(f"{r[0]:5.1f} | {r[1]:10.3f} | {r[2]:10.3f} | {r[3]:10.3f} | {r[4]:.3f}")
    assert rows[0][2] == 1.0 and rows[0][4] == 1.0               # clean regime
    assert rows[-1][2] < 1.0                                     # argmax breaks, MAP does not
    assert rows[-1][4] > rows[-1][3]


# 2. Noise robustness of the code itself (Aneesha, Sathya) ---------------

def test_code_noise_robustness_argmax_vs_nearest_vocab():
    """Gaussian noise on genuine z-normalised codes, no training.
    Per-position argmax (true length given) vs nearest vocabulary code."""
    torch.manual_seed(0)
    codec = make_codec()
    ids = torch.arange(V)
    code = codec.encode(ids)                                     # [V, 256*P]
    lengths = codec.length_buffer
    all_codes = F.normalize(code, dim=1)
    g = torch.Generator().manual_seed(1)
    rows = []
    for sd in [0.0, 2.0, 4.0, 5.0, 6.0, 8.0]:
        noisy = code + torch.randn(code.shape, generator=g) * sd
        # nearest vocabulary code (cosine)
        nearest = (F.normalize(noisy, dim=1) @ all_codes.T).argmax(dim=1)
        near_acc = (nearest == ids).float().mean().item()
        # per-position argmax over the 256 byte rows (cell = byte*P + pos)
        grid = noisy.view(V, 256, P)
        pred = grid.argmax(dim=1)                                # [V, P]
        pred = torch.where(torch.arange(P) < lengths.unsqueeze(1), pred,
                           torch.full_like(pred, PAD_IDX))
        dec = codec.decode(pred)
        arg_acc = np.mean([codec.decode(codec.byte_buffer[i:i+1])[0] == d for i, d in enumerate(dec)])
        rows.append((sd, arg_acc, near_acc))
    print("\nnoise sd | per-position argmax | nearest vocab code")
    for r in rows:
        print(f"{r[0]:8.1f} | {r[1]:19.3f} | {r[2]:.3f}")
    assert rows[0][1] == 1.0 and rows[0][2] == 1.0               # exact when clean
    for _, a, n in rows:
        assert n >= a - 1e-9                                     # vocabulary prior never hurts
    assert rows[-1][2] > rows[-1][1]                             # and helps at high noise


# 5. Per-script accuracy, padding excluded (Nitin) -----------------------

def test_active_slot_accuracy_ignores_padding():
    codec = make_codec()
    targets = torch.tensor([0, 1, 2])
    tb, _ = codec.target_bytes(targets)
    # predictor gets every PAD slot right but every real byte wrong
    wrong = tb.clone()
    wrong[tb != PAD_IDX] = 1
    assert active_slot_accuracy(wrong, tb) == 0.0
    assert active_slot_accuracy(tb, tb) == 1.0
    # naive all-slot accuracy would be high for the same predictor
    assert (wrong == tb).float().mean() > 0.5


def test_per_script_breakdown_reports_both_scripts():
    codec = make_codec()
    ev = Evaluator(codec, VOCAB, SCRIPTS, np.arange(V), torch.device("cpu"))
    targets = np.arange(V)
    correct = np.array([s == "latin" for s in SCRIPTS])          # Latin right, Devanagari wrong
    b = ev.breakdown(correct, targets)
    assert b["by_script"]["latin"]["acc"] == 1.0
    assert b["by_script"]["devanagari"]["acc"] == 0.0
    assert b["by_script"]["devanagari"]["n"] == len(DEVA)


# 7. Scoring equivalence with an explicitly expanded matrix (Dattatreya) -

def test_gather_scores_equal_explicit_expanded_matrix():
    """The gather-sum token score equals <log-probs, one-hot token grid>
    computed with a dense [V, P, 257] matrix that is never used in the model."""
    torch.manual_seed(0)
    codec = make_codec()
    slot_logits = torch.randn(7, P, 257)
    fast = bytegrid_token_scores(slot_logits, codec.byte_buffer, chunk=3)
    onehot = F.one_hot(codec.byte_buffer, 257).float()           # [V, P, 257]
    lp = F.log_softmax(slot_logits, dim=-1)                      # [N, P, 257]
    dense = torch.einsum("npc,vpc->nv", lp, onehot)
    assert (fast - dense).abs().max() < 1e-4
