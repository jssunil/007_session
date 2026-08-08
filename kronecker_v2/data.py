"""Corpus loading, tokenizer, script labels, and batching (SPEC.md §7).

Corpus: Wikipedia (wikimedia/wikipedia, snapshot 20231101), English + Hindi.
Exact shards used are documented in experiments/download_corpus.py. All arms
share one tokenizer, one memmapped token array, and one fixed train/val split.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


def train_bpe_tokenizer(text_files: list[str], vocab_size: int, save_dir: str | Path) -> Tokenizer:
    """Byte-level BPE. No <unk> needed: byte-level covers everything."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>"],
        show_progress=True,
    )
    tokenizer.train(text_files, trainer)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(save_dir / "tokenizer.json"))
    return tokenizer


def load_tokenizer(save_dir: str | Path) -> Tokenizer:
    return Tokenizer.from_file(str(Path(save_dir) / "tokenizer.json"))


def _bytelevel_char_to_byte() -> dict[str, int]:
    """Inverse of the GPT-2 ByteLevel byte->printable-unicode alphabet."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(0xA1, 0xAC + 1))
        + list(range(0xAE, 0xFF + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def vocab_byte_strings(tokenizer: Tokenizer) -> list[bytes]:
    """Exact UTF-8 byte string of every token id.

    The ByteLevel pre-tokenizer stores each token as a string over a printable
    byte-alphabet; we invert that alphabet character by character. This is
    exact even for tokens that split a UTF-8 character mid-sequence (where
    `tokenizer.decode` would substitute U+FFFD and lose the true bytes).
    Special tokens (all-ASCII, e.g. <|endoftext|>) pass through unchanged.
    """
    inv = _bytelevel_char_to_byte()
    id_to_token = {i: t for t, i in tokenizer.get_vocab().items()}
    out: list[bytes] = []
    for i in range(tokenizer.get_vocab_size()):
        token = id_to_token[i]
        out.append(bytes(inv[ch] for ch in token))
    return out


# ---------------------------------------------------------------------------
# Script labels
# ---------------------------------------------------------------------------


def script_of_codepoint(ch: str) -> str:
    """Unicode-block based script label used for the per-script breakdown."""
    cp = ord(ch)
    if cp < 0x80:
        return "latin"          # ASCII
    if 0x0900 <= cp <= 0x097F:
        return "devanagari"
    if 0x0C00 <= cp <= 0x0C7F:
        return "telugu"
    return "other"


def script_label(token_bytes: bytes) -> str:
    """Label a token by the script of its first non-ASCII codepoint.

    Tokens that are pure ASCII are 'latin'. Undecodable byte fragments
    (BPE pieces that split a UTF-8 character) are labelled by attempting a
    lenient decode; if nothing decodes, 'other'.
    """
    try:
        text = token_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = token_bytes.decode("utf-8", errors="ignore")
        if not text:
            return "other"
    for ch in text:
        if ord(ch) >= 0x80:
            return script_of_codepoint(ch)
    return "latin"


def compute_script_labels(byte_strings: list[bytes]) -> list[str]:
    return [script_label(b) for b in byte_strings]


# ---------------------------------------------------------------------------
# Tokenized corpus (memmap) and batching
# ---------------------------------------------------------------------------


def tokenize_to_memmap(
    tokenizer: Tokenizer,
    text_file: str | Path,
    out_path: str | Path,
    dtype: str = "uint16",
    eot_token: str = "<|endoftext|>",
    chunk_lines: int = 20000,
) -> int:
    """Tokenize a text file (one document per line) into a flat binary array.

    Documents are joined with the <|endoftext|> token. Returns token count.
    """
    eot_id = tokenizer.token_to_id(eot_token)
    assert eot_id is not None
    np_dtype = np.dtype(dtype)
    total = 0
    with open(text_file, encoding="utf-8") as fh, open(out_path, "wb") as out:
        batch: list[str] = []

        def flush(batch: list[str]) -> int:
            n = 0
            encodings = tokenizer.encode_batch(batch)
            ids: list[int] = []
            for enc in encodings:
                ids.extend(enc.ids)
                ids.append(eot_id)
            arr = np.asarray(ids, dtype=np_dtype)
            arr.tofile(out)
            n += arr.size
            return n

        for line in fh:
            line = line.strip()
            if line:
                batch.append(line)
            if len(batch) >= chunk_lines:
                total += flush(batch)
                batch = []
        if batch:
            total += flush(batch)
    return total


def token_frequencies(bin_path: str | Path, vocab_size: int, dtype: str = "uint16") -> np.ndarray:
    """Corpus frequency count per token id (from the train split)."""
    data = np.memmap(bin_path, dtype=np.dtype(dtype), mode="r")
    counts = np.bincount(data, minlength=vocab_size)
    return counts.astype(np.int64)


class TokenBatcher:
    """Standard causal-LM batching: random contiguous chunks of seq_len+1.

    Deterministic given its RNG seed. Same class serves train and val;
    val uses a fixed seed so every arm evaluates on identical batches.
    """

    def __init__(self, bin_path: str | Path, seq_len: int, batch_size: int,
                 dtype: str = "uint16", seed: int = 0, device: str | torch.device = "cpu"):
        self.data = np.memmap(bin_path, dtype=np.dtype(dtype), mode="r")
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.device = torch.device(device)
        assert len(self.data) > seq_len + 1, "corpus smaller than one sequence"

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        ix = self.rng.integers(0, len(self.data) - self.seq_len - 1, size=self.batch_size)
        x = np.stack([self.data[i : i + self.seq_len] for i in ix]).astype(np.int64)
        y = np.stack([self.data[i + 1 : i + 1 + self.seq_len] for i in ix]).astype(np.int64)
        xt = torch.from_numpy(x)
        yt = torch.from_numpy(y)
        if self.device.type == "cuda":
            xt = xt.pin_memory().to(self.device, non_blocking=True)
            yt = yt.pin_memory().to(self.device, non_blocking=True)
        return xt, yt


# ---------------------------------------------------------------------------
# Vocabulary metadata bundle (saved once, loaded by every arm)
# ---------------------------------------------------------------------------


def save_vocab_meta(
    out_dir: str | Path,
    byte_strings: list[bytes],
    scripts: list[str],
    frequencies: np.ndarray,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "byte_strings_hex": [b.hex() for b in byte_strings],
        "scripts": scripts,
        "frequencies": frequencies.tolist(),
    }
    (out_dir / "vocab_meta.json").write_text(json.dumps(meta), encoding="utf-8")


def load_vocab_meta(out_dir: str | Path) -> tuple[list[bytes], list[str], np.ndarray]:
    meta = json.loads((Path(out_dir) / "vocab_meta.json").read_text(encoding="utf-8"))
    byte_strings = [bytes.fromhex(h) for h in meta["byte_strings_hex"]]
    return byte_strings, meta["scripts"], np.asarray(meta["frequencies"], dtype=np.int64)
