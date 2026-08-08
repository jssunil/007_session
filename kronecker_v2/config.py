"""Dataclass configs, loadable from / dumpable to YAML.

A run is fully described by one RunConfig. The YAML files under
experiments/configs/ are the committed source of truth for every arm.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import yaml

CHAR_DIM = 256                      # UTF-8 byte values
DEFAULT_POS_DIM = 32                # byte positions (the window)
PAD_IDX = 256                       # padding category for the output head
HEAD_CHAR_DIM = CHAR_DIM + 1        # 257: 256 byte values + PAD


@dataclass
class ModelConfig:
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    d_ff_mult: int = 4
    seq_len: int = 512
    dropout: float = 0.0
    vocab_size: int = 32768
    input_path: str = "dense"       # "dense" | "kronecker"
    output_head: str = "dense"      # "dense" | "bytegrid" | "bytegrid_ar"
    tie_projection: bool = False
    pos_dim: int = DEFAULT_POS_DIM
    # Local AR decoder over byte slots (arm D only)
    ar_d_local: int = 128
    ar_n_layers: int = 2
    ar_n_heads: int = 4


@dataclass
class TrainConfig:
    seed: int = 1337
    max_steps: int = 12000
    batch_size: int = 32
    grad_accum: int = 1
    lr: float = 3e-4
    warmup_steps: int = 500
    min_lr_frac: float = 0.1        # cosine floor as a fraction of lr
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    eval_every: int = 500
    eval_batches: int = 40
    checkpoint_every: int = 4000
    amp: bool = True                # bf16 autocast on CUDA
    device: str = "auto"            # "auto" | "cuda" | "cpu"


@dataclass
class DataConfig:
    data_dir: str = "data"
    tokenizer_dir: str = "data/tokenizer"
    train_bin: str = "data/train.bin"
    val_bin: str = "data/val.bin"
    token_dtype: str = "uint16"     # uint16 is enough for V<=65536


@dataclass
class RunConfig:
    name: str = "unnamed"
    out_dir: str = "results"
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # ---- YAML round-trip ----

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RunConfig":
        model = ModelConfig(**d.get("model", {}))
        train_d = dict(d.get("train", {}))
        if "betas" in train_d:
            train_d["betas"] = tuple(train_d["betas"])
        train = TrainConfig(**train_d)
        data = DataConfig(**d.get("data", {}))
        return cls(
            name=d.get("name", "unnamed"),
            out_dir=d.get("out_dir", "results"),
            model=model,
            train=train,
            data=data,
        )

    def save_yaml(self, path: str | Path) -> None:
        d = self.to_dict()
        d["train"]["betas"] = list(d["train"]["betas"])  # YAML has no tuples
        Path(path).write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")

    @classmethod
    def load_yaml(cls, path: str | Path) -> "RunConfig":
        d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(d)
