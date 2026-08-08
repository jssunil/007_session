"""M4: every input/head combination has correct shapes; param counts match
the formulas in CLAUDE.md exactly."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kronecker_v2.codec import KroneckerCodec
from kronecker_v2.config import HEAD_CHAR_DIM, ModelConfig
from kronecker_v2.model import LM

V, D, P = 512, 64, 32
VOCAB = [f"tok{i}".encode() for i in range(V)]


def make(input_path, output_head, tie=False):
    cfg = ModelConfig(d_model=D, n_layers=2, n_heads=2, seq_len=32, vocab_size=V,
                      input_path=input_path, output_head=output_head,
                      tie_projection=tie, ar_d_local=32, ar_n_layers=1, ar_n_heads=2)
    codec = KroneckerCodec(VOCAB, pos_dim=P)
    return LM(cfg, codec)


ARMS = [
    ("dense", "dense", False),        # A
    ("kronecker", "dense", False),    # B
    ("kronecker", "bytegrid", False), # C
    ("kronecker", "bytegrid_ar", False),  # D
    ("kronecker", "bytegrid", True),  # E
]


@pytest.mark.parametrize("inp,out,tie", ARMS)
def test_forward_shapes_and_loss(inp, out, tie):
    model = make(inp, out, tie)
    x = torch.randint(0, V, (2, 16))
    y = torch.randint(0, V, (2, 16))
    h = model.trunk(x)
    assert h.shape == (2, 16, D)
    loss = model.loss(x, y)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()  # gradients flow


def test_param_count_formulas():
    code_dim = 256 * P
    a = make("dense", "dense").param_counts()
    assert a["params_input"] == V * D
    assert a["params_head"] == D * V

    c = make("kronecker", "bytegrid").param_counts()
    assert c["params_input"] == code_dim * D
    assert c["params_head"] == D * P * HEAD_CHAR_DIM

    e = make("kronecker", "bytegrid", tie=True).param_counts()
    assert e["params_head"] == P * D            # only the PAD logits are new
    # E total = C total - untied head + PAD logits
    assert e["params_total"] == c["params_total"] - D * P * HEAD_CHAR_DIM + P * D


def test_head_v_independence():
    """The byte-grid head's parameter count must not change with V."""
    small = make("kronecker", "bytegrid").param_counts()["params_head"]
    big_vocab = [f"token{i}".encode() for i in range(4 * V)]
    cfg = ModelConfig(d_model=D, n_layers=2, n_heads=2, seq_len=32, vocab_size=4 * V,
                      input_path="kronecker", output_head="bytegrid")
    big = LM(cfg, KroneckerCodec(big_vocab, pos_dim=P)).param_counts()
    assert big["params_head"] == small
    assert big["params_input"] == 256 * P * D  # input also V-independent


def test_tied_weight_is_shared():
    model = make("kronecker", "bytegrid", tie=True)
    assert model.head.tied_projection.weight is model.embedding.projection.weight
