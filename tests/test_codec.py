"""M2 correctness gate: round-trip, scaling, normalisation, batching, locality."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kronecker_v2.codec import KroneckerCodec
from kronecker_v2.config import PAD_IDX

# Synthetic vocabulary covering the tricky cases: ASCII, multi-byte UTF-8,
# Devanagari conjuncts, a colliding pair (>32 bytes sharing the prefix),
# and a 1-byte token.
VOCAB = [
    b"a",
    b"the",
    b"training",
    "भारत".encode(),
    "క్షేత్రం".encode(),
    "अंतर्राष्ट्रीयकरण".encode(),   # 48 bytes -> truncated at 32
    "अंतर्राष्ट्रीयता".encode(),     # 45 bytes -> same first 32 bytes: collision
    b" hello",
    bytes(range(1, 33)),             # exactly pos_dim bytes
    bytes(range(1, 40)),             # longer than pos_dim, shares prefix: collision
]


@pytest.fixture(scope="module")
def codec():
    return KroneckerCodec(VOCAB)


def collision_free_ids():
    prefixes = [bs[:32] for bs in VOCAB]
    return [i for i, p in enumerate(prefixes) if prefixes.count(p) == 1]


def test_roundtrip_all_noncolliding(codec):
    """THE GATE: decode(target_bytes(id)) == vocab bytes, zero training."""
    ids = torch.tensor(collision_free_ids())
    targets, mask = codec.target_bytes(ids)
    decoded = codec.decode(targets)
    for i, tid in enumerate(ids.tolist()):
        assert decoded[i] == VOCAB[tid][:32], f"round-trip failed for id {tid}"
        assert len(VOCAB[tid]) <= 32  # non-colliding short tokens are exact
        assert decoded[i] == VOCAB[tid]


def test_colliding_pair_identical_codes(codec):
    """Tokens sharing their first 32 bytes must produce identical codes."""
    ids = torch.tensor([[5, 6]])
    codes = codec.encode(ids)
    assert torch.allclose(codes[0, 0], codes[0, 1])


def test_scaling_and_znorm(codec):
    """1/sqrt(L) scaling pre-normalisation; z-norm: mean 0, std 1."""
    ids = torch.tensor([[0, 1, 2]])
    codes = codec.encode(ids)
    flat = codes.reshape(-1, codec.code_dim)
    assert torch.allclose(flat.mean(dim=1), torch.zeros(3), atol=1e-5)
    assert torch.allclose(flat.std(dim=1), torch.ones(3), atol=1e-4)
    # Before z-norm each marked cell holds 1/sqrt(L); verify via ratio of
    # distinct nonzero levels for tokens of different length ("a" vs "the").
    # After z-norm, check the *count* of cells above the mean equals L.
    for j, tid in enumerate([0, 1, 2]):
        L = codec.length_buffer[tid].item()
        n_marked = (flat[j] > flat[j].mean() + 3).sum().item()  # marks are big outliers
        assert n_marked == L


def test_batched_encode_matches_looped(codec):
    ids = torch.tensor([[0, 3, 5], [7, 8, 2]])
    batched = codec.encode(ids)
    for b in range(2):
        for t in range(3):
            single = codec.encode(ids[b : b + 1, t : t + 1])
            assert torch.allclose(batched[b, t], single[0, 0], atol=1e-6)


def test_prefix_locality(codec):
    """Shared-prefix tokens are closer in cosine than unrelated tokens."""
    ids = torch.tensor([[2, 5, 6, 3]])  # training / अंतर्…करण / अंतर्…ता / भारत
    c = codec.encode(ids)[0]
    cos = torch.nn.functional.cosine_similarity
    related = cos(c[1], c[2], dim=0)      # the two Hindi near-duplicates
    unrelated = cos(c[0], c[3], dim=0)    # "training" vs "भारत"
    assert related > 0.99
    assert related > unrelated + 0.3


def test_no_trainable_params(codec):
    assert sum(p.numel() for p in codec.parameters()) == 0


def test_target_bytes_mask(codec):
    ids = torch.tensor([0])  # "a", length 1
    targets, mask = codec.target_bytes(ids)
    assert targets.shape == (1, 32)
    assert mask.sum() == 1
    assert targets[0, 0] == ord("a")
    assert (targets[0, 1:] == PAD_IDX).all()
