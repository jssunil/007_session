"""M4: the byte-grid head can learn the mapping at all (overfit one batch),
and greedy/constrained decoding recover tokens."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kronecker_v2.codec import KroneckerCodec
from kronecker_v2.heads import ByteGridARHead, ByteGridHead
from kronecker_v2.trie import ByteTrie

V, D, P = 256, 64, 32
VOCAB = [f"word{i:03d}".encode() for i in range(V)]


def overfit(head, h, targets, steps=400, lr=1e-2, ar=False):
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = head.loss(h, targets)
        loss.backward()
        opt.step()
    return loss.item()


def test_bytegrid_head_overfits_tiny_batch():
    torch.manual_seed(0)
    codec = KroneckerCodec(VOCAB, pos_dim=P)
    head = ByteGridHead(D, P, codec)
    h = torch.randn(4, 8, D)                    # frozen fake trunk output
    targets = torch.randint(0, V, (4, 8))
    loss = overfit(head, h, targets)
    assert loss < 0.01
    pred = head.greedy_bytes(h)
    decoded = codec.decode(pred.reshape(-1, P))
    want = [VOCAB[t] for t in targets.reshape(-1).tolist()]
    acc = sum(d == w for d, w in zip(decoded, want)) / len(want)
    assert acc == 1.0, f"exact recovery {acc} after overfit"


def test_ar_head_overfits_tiny_batch():
    torch.manual_seed(0)
    codec = KroneckerCodec(VOCAB, pos_dim=P)
    head = ByteGridARHead(D, P, codec, d_local=32, n_layers=1, n_heads=2)
    h = torch.randn(2, 4, D)
    targets = torch.randint(0, V, (2, 4))
    loss = overfit(head, h, targets, steps=600, ar=True)
    assert loss < 0.05
    pred = head.greedy_bytes(h)
    decoded = codec.decode(pred.reshape(-1, P))
    want = [VOCAB[t] for t in targets.reshape(-1).tolist()]
    acc = sum(d == w for d, w in zip(decoded, want)) / len(want)
    assert acc == 1.0


def test_trie_constrained_decode_fixes_invalid():
    """Constrained decoding must always return a real token, and must agree
    with unconstrained decoding when the prediction is already valid."""
    torch.manual_seed(0)
    codec = KroneckerCodec(VOCAB, pos_dim=P)
    trie = ByteTrie(VOCAB, P)
    # logits favouring an INVALID byte string ("zzz..." not in vocab)
    n = 3
    logits = torch.full((n, P, 257), -10.0)
    for pos in range(P):
        logits[:, pos, ord("z")] = 5.0
    byte_vals, ids = trie.constrained_greedy(logits)
    for i in range(n):
        assert 0 <= ids[i] < V                  # a real token, always
        decoded = codec.decode(byte_vals[i : i + 1])[0]
        assert decoded == VOCAB[ids[i]][:P]
    # and when the argmax already spells a real token, constrained == greedy
    tid = 42
    logits2 = torch.full((1, P, 257), -10.0)
    tb, _ = codec.target_bytes(torch.tensor([tid]))
    for pos in range(P):
        logits2[0, pos, tb[0, pos]] = 5.0
    _, ids2 = trie.constrained_greedy(logits2)
    assert ids2[0] == tid
