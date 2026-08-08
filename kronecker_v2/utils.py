"""Seeding and JSONL run logging (TASKS.md M0)."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed python, numpy, and torch. Called once at the start of every run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        # cuDNN determinism; matmul kernels stay default (TF32 off keeps runs
        # bit-comparable across arms on the same hardware).
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


class JsonlLogger:
    """One JSONL file per run: one line per eval step, plus a final metrics blob."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")

    def log(self, record: dict) -> None:
        record = dict(record)
        record.setdefault("wall_time", time.time())
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def get_device(pref: str = "auto") -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)
