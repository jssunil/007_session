"""Arm F: byte-grid head trained with a whole-token softmax (CPU only)."""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kronecker_v2.codec import KroneckerCodec
from kronecker_v2.config import ModelConfig
from kronecker_v2.heads import ByteGridTokenHead
from kronecker_v2.metrics import bytegrid_token_scores
from kronecker_v2.model import LM

V, D, P = 200, 48, 32
VOCAB = [f"word{i:03d}".encode() for i in range(V)]


def make(**kw):
    codec = KroneckerCodec(VOCAB, pos_dim=P)
    return codec, ByteGridTokenHead(D, P, codec, **kw)


def test_loss_matches_explicit_dense_ce_and_chunking():
    """Chunked/checkpointed loss == plain CE over an explicitly expanded
    one-hot matrix, with and without the token bias."""
    torch.manual_seed(0)
    freqs = np.random.default_rng(0).integers(1, 1000, V)
    codec, head = make(token_freqs=freqs, row_chunk=7)
    head.token_bias.data += torch.randn(V)                        # non-trivial bias
    h = torch.randn(3, 5, D)
    y = torch.randint(0, V, (3, 5))
    got = head.loss(h, y)
    onehot = F.one_hot(codec.byte_buffer, 257).float()            # [V, P, 257]
    logits = head.forward(h).reshape(-1, P, 257)
    ref_scores = torch.einsum("npc,vpc->nv", logits, onehot) + head.token_bias
    ref = F.cross_entropy(ref_scores, y.reshape(-1))
    assert torch.allclose(got, ref, atol=1e-4), (got.item(), ref.item())


def test_bias_init_is_log_unigram_probability():
    freqs = np.arange(1, V + 1)
    _, head = make(token_freqs=freqs)
    p = torch.softmax(head.token_bias.data, dim=0)
    assert torch.allclose(p, torch.tensor(freqs / freqs.sum(), dtype=torch.float32), atol=1e-6)
    _, nobias = make(use_token_bias=False)
    assert nobias.token_bias is None


def test_eval_scores_consistent_with_training_scores():
    """Eval path (per-slot log-softmax sum + bias) and training path (raw
    logit sum + bias) differ only by a per-row constant, so softmax over V
    is identical."""
    torch.manual_seed(1)
    codec, head = make(token_freqs=np.arange(1, V + 1))
    h = torch.randn(4, D)
    slot_logits = head.forward(h)
    train_scores = head.token_scores(slot_logits)
    eval_scores = bytegrid_token_scores(slot_logits, codec.byte_buffer) + head.token_bias
    assert torch.allclose(torch.log_softmax(train_scores, 1), torch.log_softmax(eval_scores, 1), atol=1e-4)


def test_arm_f_overfits_and_always_decodes_to_real_token():
    torch.manual_seed(0)
    codec, head = make(token_freqs=np.ones(V))
    h = torch.randn(2, 8, D)
    y = torch.randint(0, V, (2, 8))
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    for _ in range(300):
        opt.zero_grad()
        loss = head.loss(h, y)
        loss.backward()
        opt.step()
    assert loss.item() < 0.05
    pick = head.token_scores(head.forward(h).reshape(-1, P, 257)).argmax(1)
    assert (pick == y.reshape(-1)).all()


def test_lm_builds_with_arm_f_and_counts_bias_params():
    cfg = ModelConfig(d_model=48, n_layers=1, n_heads=2, seq_len=16, vocab_size=V,
                      input_path="kronecker", output_head="bytegrid_token", pos_dim=P)
    codec = KroneckerCodec(VOCAB, pos_dim=P)
    model = LM(cfg, codec, token_freqs=np.ones(V))
    x = torch.randint(0, V, (2, 8))
    loss = model.loss(x, x)
    assert torch.isfinite(loss)
    assert model.param_counts()["params_head"] == D * P * 257 + V   # proj + 1 bias/token


def test_fast_scores_match_gather_scores_values_and_grads():
    """The embedding_bag path must equal the P-gather reference, in value and
    in the gradient reaching the slot logits and the token bias."""
    torch.manual_seed(0)
    codec, head = make(token_freqs=np.arange(1, V + 1))
    head.token_bias.data += torch.randn(V)
    logits = torch.randn(9, P, 257, requires_grad=True)
    w = torch.randn(9, V)                                          # random upstream gradient

    a = head.token_scores(logits)
    (a * w).sum().backward()
    g_logits_sparse, g_bias_sparse = logits.grad.clone(), head.token_bias.grad.clone()

    logits.grad = None
    head.token_bias.grad = None
    b = head.token_scores_gather(logits)
    (b * w).sum().backward()

    assert torch.allclose(a, b, atol=1e-5)
    assert torch.allclose(g_logits_sparse, logits.grad, atol=1e-5)
    assert torch.allclose(g_bias_sparse, head.token_bias.grad, atol=1e-5)


def test_cell_index_picks_one_cell_per_slot_and_is_not_saved():
    codec, head = make()
    idx = head._cell_index(torch.device("cpu"))
    assert idx.shape == (V, P)
    assert (idx // 257 == torch.arange(P)).all()                   # column p lives in slot p
    assert (idx % 257 == codec.byte_buffer).all()                  # and selects that token's byte
    assert not any("index" in k for k in head.state_dict())
