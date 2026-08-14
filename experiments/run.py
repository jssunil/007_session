"""Single-arm training runner (TASKS.md M6).

Usage: python experiments/run.py experiments/configs/arm_c_kron_bytegrid.yaml

Everything an arm needs is in its YAML config. Logs one JSONL line per eval
to results/logs/<name>.jsonl and a final metrics blob to
results/metrics/<name>.json; checkpoint to results/checkpoints/<name>.pt.

Evaluation protocol (identical across arms):
  - val batches are drawn from a batcher re-seeded to 0 at every eval, so
    every arm at every step evaluates on the SAME batches.
  - own_loss: the arm's training objective on the val split (NOT comparable
    across head types; logged for curve-shape only).
  - comparable_nll / ppl: token-level NLL on the common basis (metrics.py).
    For the AR head this is exact but expensive, so it runs on the first
    `ar_nll_positions` positions of the eval set (recorded in the output).
  - exact-recovery (byte heads): greedy unconstrained + trie-constrained;
    dense arms report argmax-over-V accuracy in the same columns.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kronecker_v2.codec import KroneckerCodec
from kronecker_v2.config import RunConfig
from kronecker_v2.data import TokenBatcher, load_vocab_meta
from kronecker_v2.metrics import (
    Evaluator,
    ar_token_scores,
    bytegrid_token_scores,
    comparable_nll,
    train_memory_gb,
)
from kronecker_v2.model import LM
from kronecker_v2.trie import ByteTrie
from kronecker_v2.utils import JsonlLogger, get_device, seed_everything

# eval sizing (identical across arms; subsets exist purely for tractability)
NLL_BATCHES = 8            # batches used for comparable-NLL (dense + parallel byte)
EXACT_BATCHES = 4          # batches used for greedy exact-recovery
CONSTRAINED_POSITIONS = 4096   # positions for trie-constrained decoding (python loop)
AR_NLL_POSITIONS = 1024        # positions for the AR head's exact comparable-NLL


def lr_at(step: int, cfg) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    t = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    floor = cfg.lr * cfg.min_lr_frac
    return floor + 0.5 * (cfg.lr - floor) * (1 + math.cos(math.pi * t))


@torch.no_grad()
def evaluate(model: LM, cfg: RunConfig, evaluator: Evaluator, trie: ByteTrie,
             device: torch.device, final: bool = False) -> dict:
    model.eval()
    m = cfg.model
    val = TokenBatcher(cfg.data.val_bin, m.seq_len, cfg.train.batch_size,
                       dtype=cfg.data.token_dtype, seed=0, device=device)
    is_byte = m.output_head in ("bytegrid", "bytegrid_ar")
    is_ar = m.output_head == "bytegrid_ar"

    own_losses, nlls = [], []
    correct_u, correct_c, valid_u = [], [], []
    tgt_u, tgt_c = [], []

    for bi in range(cfg.train.eval_batches):
        x, y = val.next_batch()
        with torch.autocast("cuda", torch.bfloat16, enabled=cfg.train.amp and device.type == "cuda"):
            h = model.trunk(x)
            own_losses.append(model.head.loss(h, y).item())
        h = h.float()
        yf = y.reshape(-1)

        if bi < NLL_BATCHES:
            if not is_byte:
                logits = model.head(h).reshape(-1, m.vocab_size)
                nlls.append(evaluator.token_nll_dense(logits, yf).cpu())
            elif not is_ar:
                slot_logits = model.head(h).reshape(-1, m.pos_dim, 257)
                nlls.append(evaluator.token_nll_bytegrid(slot_logits, yf).cpu())
            elif bi == 0:
                hf = h.reshape(-1, m.d_model)[:AR_NLL_POSITIONS]
                scores = ar_token_scores(model.head, hf, model.codec.byte_buffer)
                nlls.append(comparable_nll(scores, yf[:AR_NLL_POSITIONS]).cpu())

        if bi < EXACT_BATCHES:
            if is_byte:
                if is_ar:
                    pred = model.head.greedy_bytes(h).reshape(-1, m.pos_dim)
                else:
                    pred = model.head.greedy_bytes(h).reshape(-1, m.pos_dim)
                cu, vu = evaluator.exact_recovery(pred, yf)
                correct_u.append(cu)
                valid_u.append(vu)
                tgt_u.append(yf.cpu())
                if bi == 0 and (final or not is_ar):
                    # constrained decoding on a subset (python trie loop)
                    k = min(CONSTRAINED_POSITIONS, yf.shape[0])
                    if is_ar:
                        # constrained AR: use teacher-forced logits as proxy is
                        # invalid; run true constrained greedy stepwise
                        slot_lp = None
                        cc = constrained_ar(model.head, h.reshape(-1, m.d_model)[:k], trie)
                        correct_c.append(torch.tensor(cc) == yf[:k].cpu())
                    else:
                        slot_logits = model.head(h).reshape(-1, m.pos_dim, 257)[:k]
                        _, ids = trie.constrained_greedy(slot_logits)
                        correct_c.append(ids == yf[:k].cpu())
                    tgt_c.append(yf[:k].cpu())
            else:
                logits = model.head(h).reshape(-1, m.vocab_size)
                cu = (logits.argmax(dim=-1) == yf).cpu()
                correct_u.append(cu)
                valid_u.append(torch.ones_like(cu))
                correct_c.append(cu)
                tgt_u.append(yf.cpu())
                tgt_c.append(yf.cpu())

    nll = torch.cat(nlls).mean().item() if nlls else float("nan")
    cu = torch.cat(correct_u).numpy()
    vu = torch.cat(valid_u).numpy()
    cc = torch.cat(correct_c).numpy() if correct_c else np.array([False])
    out = {
        "own_loss": float(np.mean(own_losses)),
        "comparable_nll": nll,
        "ppl": math.exp(min(nll, 20)) if nlls else float("nan"),
        "exact_acc_unconstrained": float(cu.mean()),
        "exact_acc_constrained": float(cc.mean()),
        "invalid_rate": float(1 - vu.mean()),
        "nll_positions": int(sum(t.numel() for t in nlls)) if nlls else 0,
    }
    if final:
        tgt = torch.cat(tgt_u).numpy()
        out["breakdown_unconstrained"] = evaluator.breakdown(cu, tgt)
        tgtc = torch.cat(tgt_c).numpy()
        out["breakdown_constrained"] = evaluator.breakdown(cc, tgtc)
    model.train()
    return out


@torch.no_grad()
def constrained_ar(head, h_flat: torch.Tensor, trie: ByteTrie) -> list[int]:
    """True stepwise constrained greedy decode for the AR head."""
    from kronecker_v2.config import HEAD_CHAR_DIM, PAD_IDX
    n = h_flat.shape[0]
    parallel = head._parallel_logits(h_flat).view(n, head.pos_dim, HEAD_CHAR_DIM)
    prev = torch.full((n, 1), head.bos_idx, dtype=torch.int64, device=h_flat.device)
    nodes = [trie.root] * n
    done = [False] * n
    ids = [0] * n
    for p in range(head.pos_dim):
        step_logits = (parallel[:, p] + head._slot_logits(h_flat, prev)[:, -1]).cpu()
        nxt = torch.full((n,), PAD_IDX, dtype=torch.int64)
        for i in range(n):
            if done[i]:
                continue
            node = nodes[i]
            allowed = list(node.keys())
            if id(node) in trie.terminal:
                allowed.append(PAD_IDX)
            if not allowed:
                done[i] = True
                continue
            choice = allowed[int(step_logits[i, allowed].argmax())]
            if choice == PAD_IDX:
                done[i] = True
            else:
                nxt[i] = choice
                nodes[i] = node[choice]
        prev = torch.cat([prev, nxt.to(h_flat.device).unsqueeze(1)], dim=1)
        if all(done):
            break
    for i in range(n):
        ids[i] = trie.terminal.get(id(nodes[i]), 0)
    return ids


def save_checkpoint(path: Path, model: LM, opt: torch.optim.Optimizer,
                    train: TokenBatcher, cfg: RunConfig, step: int,
                    tokens_seen: int) -> None:
    """Atomic full-state checkpoint: crash during save leaves the old file intact."""
    state = {
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "config": cfg.to_dict(),
        "step": step,
        "tokens_seen": tokens_seen,
        "train_rng_state": train.rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    tmp = path.with_suffix(".pt.tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def try_resume(path: Path, model: LM, opt: torch.optim.Optimizer,
               train: TokenBatcher, cfg: RunConfig, device: torch.device) -> tuple[int, int]:
    """Load a full-state checkpoint if present; return (start_step, tokens_seen).

    Only checkpoints that carry optimizer + RNG state are resumed (older
    model-only checkpoints can't continue the run faithfully and are ignored).
    """
    if not path.exists():
        return 0, 0
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "opt" not in ckpt or "train_rng_state" not in ckpt:
        print(f"[{cfg.name}] checkpoint {path} lacks optimizer/RNG state; starting fresh")
        return 0, 0
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["opt"])
    train.rng.bit_generator.state = ckpt["train_rng_state"]
    torch.set_rng_state(ckpt["torch_rng_state"].cpu())
    if ckpt.get("cuda_rng_state") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in ckpt["cuda_rng_state"]])
    print(f"[{cfg.name}] resuming from {path} at step {ckpt['step']}")
    return ckpt["step"], ckpt["tokens_seen"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=str)
    ap.add_argument("--fresh", action="store_true",
                    help="ignore any existing checkpoint and start from step 0")
    args = ap.parse_args()
    cfg = RunConfig.load_yaml(args.config)
    seed_everything(cfg.train.seed)
    device = get_device(cfg.train.device)
    if cfg.train.device == "auto" and device.type != "cuda":
        # After a driver crash (TDR) CUDA silently disappears and 'auto' would
        # burn the arm on a ~200x slower CPU run. Opt in with device: cpu.
        raise RuntimeError(
            f"[{cfg.name}] device=auto but CUDA is unavailable. "
            "Refusing to train a full arm on CPU; set device: cpu explicitly "
            "if that is really intended."
        )

    byte_strings, scripts, freqs = load_vocab_meta(cfg.data.tokenizer_dir)
    assert len(byte_strings) == cfg.model.vocab_size
    codec = KroneckerCodec(byte_strings, pos_dim=cfg.model.pos_dim).to(device)
    model = LM(cfg.model, codec).to(device)
    counts = model.param_counts()
    evaluator = Evaluator(codec, byte_strings, scripts, freqs, device)
    trie = ByteTrie(byte_strings, cfg.model.pos_dim)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                            betas=cfg.train.betas, weight_decay=cfg.train.weight_decay)
    train = TokenBatcher(cfg.data.train_bin, cfg.model.seq_len, cfg.train.batch_size,
                         dtype=cfg.data.token_dtype, seed=cfg.train.seed, device=device)

    ckpt_dir = Path(cfg.out_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{cfg.name}.pt"

    start_step, tokens_seen = (0, 0) if args.fresh else \
        try_resume(ckpt_path, model, opt, train, cfg, device)

    logger = JsonlLogger(Path(cfg.out_dir) / "logs" / f"{cfg.name}.jsonl")
    logger.log({"event": "start" if start_step == 0 else "resume",
                "config": cfg.to_dict(), "param_counts": counts,
                "device": str(device), "torch": torch.__version__,
                "start_step": start_step})
    print(f"[{cfg.name}] params: {counts}")

    t_start = time.time()
    window_tokens, window_t0 = 0, time.time()
    tokens_per_sec = 0.0
    model.train()

    for step in range(start_step, cfg.train.max_steps):
        lr = lr_at(step, cfg.train)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = train.next_batch()
        with torch.autocast("cuda", torch.bfloat16, enabled=cfg.train.amp and device.type == "cuda"):
            loss = model.loss(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        tokens_seen += x.numel()
        window_tokens += x.numel()

        if (step + 1) % cfg.train.eval_every == 0 or step == cfg.train.max_steps - 1:
            if device.type == "cuda":
                torch.cuda.synchronize()
            tokens_per_sec = window_tokens / (time.time() - window_t0)
            final = step == cfg.train.max_steps - 1
            ev = evaluate(model, cfg, evaluator, trie, device, final=final)
            rec = {"event": "eval", "step": step + 1, "lr": lr,
                   "train_loss": loss.item(), "tokens_seen": tokens_seen,
                   "tokens_per_sec": round(tokens_per_sec), **ev}
            logger.log(rec)
            print(f"[{cfg.name}] step {step+1} train {loss.item():.4f} "
                  f"own {ev['own_loss']:.4f} nll {ev['comparable_nll']:.4f} "
                  f"ppl {ev['ppl']:.1f} exactU {ev['exact_acc_unconstrained']:.4f} "
                  f"exactC {ev['exact_acc_constrained']:.4f} inv {ev['invalid_rate']:.4f} "
                  f"tok/s {tokens_per_sec:.0f}")
            window_tokens, window_t0 = 0, time.time()

        if (step + 1) % cfg.train.checkpoint_every == 0 or step == cfg.train.max_steps - 1:
            save_checkpoint(ckpt_path, model, opt, train, cfg, step + 1, tokens_seen)

    final_eval = evaluate(model, cfg, evaluator, trie, device, final=True)
    blob = {
        "name": cfg.name,
        "seed": cfg.train.seed,
        "resumed_from_step": start_step,
        "wall_clock_sec": round(time.time() - t_start, 1),
        "tokens_seen": tokens_seen,
        "tokens_per_sec_last": round(tokens_per_sec),
        **counts,
        "train_memory_gb_16B_accounting": round(train_memory_gb(counts["params_total"]), 3),
        **final_eval,
        "eval_protocol": {
            "eval_batches": cfg.train.eval_batches,
            "nll_batches": NLL_BATCHES,
            "exact_batches": EXACT_BATCHES,
            "constrained_positions": CONSTRAINED_POSITIONS,
            "ar_nll_positions": AR_NLL_POSITIONS,
        },
    }
    logger.log({"event": "final", **blob})
    out = Path(cfg.out_dir) / "metrics" / f"{cfg.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    logger.close()
    print(f"[{cfg.name}] done in {blob['wall_clock_sec']}s -> {out}")


if __name__ == "__main__":
    main()
