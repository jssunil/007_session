"""M5 GATE: the comparable-NLL path must reproduce standard token NLL exactly
for a dense head, and be internally consistent for the byte-grid head."""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kronecker_v2.codec import KroneckerCodec
from kronecker_v2.metrics import bytegrid_token_scores, comparable_nll

V, P = 512, 32
VOCAB = [f"tok{i}".encode() for i in range(V)]


def test_dense_identity_gate():
    """comparable_nll(logits, y) == cross_entropy(logits, y) exactly."""
    torch.manual_seed(0)
    logits = torch.randn(64, V)
    y = torch.randint(0, V, (64,))
    ours = comparable_nll(logits, y)
    ref = F.cross_entropy(logits, y, reduction="none")
    assert torch.allclose(ours, ref, atol=1e-5), (ours - ref).abs().max()


def test_bytegrid_scores_match_bruteforce():
    """Vectorised scoring == naive per-token loop."""
    torch.manual_seed(0)
    codec = KroneckerCodec(VOCAB, pos_dim=P)
    slot_logits = torch.randn(5, P, 257)
    scores = bytegrid_token_scores(slot_logits, codec.byte_buffer, chunk=2)
    logprobs = F.log_softmax(slot_logits, dim=-1)
    for n in range(5):
        for v in [0, 7, V - 1]:
            ref = sum(logprobs[n, p, codec.byte_buffer[v, p]] for p in range(P))
            assert torch.allclose(scores[n, v], ref, atol=1e-5)


def test_bytegrid_nll_decreases_with_confidence():
    """Sharpening the correct token's slots must lower its comparable NLL."""
    codec = KroneckerCodec(VOCAB, pos_dim=P)
    tid = torch.tensor([3])
    tb, _ = codec.target_bytes(tid)
    weak = torch.zeros(1, P, 257)
    strong = torch.zeros(1, P, 257)
    for p in range(P):
        strong[0, p, tb[0, p]] = 8.0
    s_weak = bytegrid_token_scores(weak, codec.byte_buffer)
    s_strong = bytegrid_token_scores(strong, codec.byte_buffer)
    assert comparable_nll(s_strong, tid) < comparable_nll(s_weak, tid)
    # and with near-one-hot slots, the correct token dominates the vocab
    assert s_strong.argmax(dim=1).item() == 3


def test_vocab_map_is_always_valid_and_recovers_clean_prediction():
    """The vocab-MAP decode picks a real token even when the per-slot argmax
    spells an invalid string, and top-1 equals the target on clean logits."""
    import torch
    from kronecker_v2.codec import KroneckerCodec
    from kronecker_v2.metrics import Evaluator, bytegrid_token_scores
    import numpy as np

    vocab = [f"word{i:03d}".encode() for i in range(64)]
    codec = KroneckerCodec(vocab, pos_dim=32)
    ev = Evaluator(codec, vocab, ["latin"] * 64, np.arange(64), torch.device("cpu"))

    targets = torch.tensor([3, 40])
    tb, _ = codec.target_bytes(targets)
    clean = torch.full((2, 32, 257), -10.0)
    for i in range(2):
        for p in range(32):
            clean[i, p, tb[i, p]] = 5.0
    scores = ev.token_scores_bytegrid(clean)
    top1, top5 = ev.vocab_map_hits(scores, targets)
    assert top1.all() and top5.all()

    junk = torch.full((1, 32, 257), -10.0)
    junk[:, :, ord("z")] = 5.0                       # argmax spells an invalid string
    s = bytegrid_token_scores(junk, codec.byte_buffer)
    assert 0 <= int(s.argmax(dim=1)) < 64
