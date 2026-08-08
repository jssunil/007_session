"""M1: corpus extraction, tokenizer training, tokenization, vocab metadata.

Steps (all deterministic):
  1. Extract text from the parquet shards into train/val text files.
     Doc-level split: every 50th document goes to val (2%). Newlines inside a
     document become spaces; one document per line.
     Byte caps keep the corpus small enough for few-hour training runs.
  2. Train a byte-level BPE tokenizer, vocab_size=32768, on the TRAIN text only.
  3. Extract vocab byte strings, script labels, corpus frequencies.
  4. Tokenize train/val into flat binary memmaps (uint16).
  5. Acceptance report: lossless round-trip per script, token counts,
     per-script token share, fertility (tokens per whitespace word).

Run:  python experiments/prepare_data.py [--vocab-size 32768 --tag ""]
The --tag suffix produces a parallel dataset build (used for the 64K anchor:
 --vocab-size 65536 --tag _64k --dtype uint32 reuses the same text files).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kronecker_v2.data import (
    compute_script_labels,
    load_tokenizer,
    save_vocab_meta,
    token_frequencies,
    tokenize_to_memmap,
    train_bpe_tokenizer,
    vocab_byte_strings,
)

RAW = {
    "en": ["data/raw/en-00000.parquet"],
    "hi": ["data/raw/hi-00000.parquet", "data/raw/hi-00001.parquet"],
}
# UTF-8 byte caps on extracted TRAIN text. Hindi needs ~3x the bytes of
# English for comparable token counts (3-byte Devanagari chars).
BYTE_CAPS = {"en": 120_000_000, "hi": 300_000_000}
VAL_EVERY = 50  # every 50th document -> val


def extract_text(force: bool = False) -> None:
    for lang, shards in RAW.items():
        train_path = Path(f"data/{lang}_train.txt")
        val_path = Path(f"data/{lang}_val.txt")
        if train_path.exists() and not force:
            print(f"skip extract {lang} (exists)")
            continue
        cap = BYTE_CAPS[lang]
        written = 0
        n_docs = 0
        with open(train_path, "w", encoding="utf-8") as tr, open(val_path, "w", encoding="utf-8") as va:
            for shard in shards:
                if written >= cap:
                    break
                table = pq.read_table(shard, columns=["text"])
                for text in table.column("text").to_pylist():
                    line = " ".join(text.split()) + "\n"
                    out = va if (n_docs % VAL_EVERY == VAL_EVERY - 1) else tr
                    out.write(line)
                    n_docs += 1
                    if out is tr:
                        written += len(line.encode("utf-8"))
                        if written >= cap:
                            break
        print(f"extracted {lang}: {written/1e6:.1f} MB train text, {n_docs} docs")


def fertility(tokenizer, text: str) -> float:
    words = text.split()
    ids = tokenizer.encode(text).ids
    return len(ids) / max(1, len(words))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-size", type=int, default=32768)
    ap.add_argument("--tag", type=str, default="", help="suffix for tokenizer/bin outputs")
    ap.add_argument("--dtype", type=str, default="uint16")
    ap.add_argument("--force-extract", action="store_true")
    args = ap.parse_args()
    assert args.dtype == "uint32" or args.vocab_size <= 65536

    t0 = time.time()
    extract_text(force=args.force_extract)

    tok_dir = Path(f"data/tokenizer{args.tag}")
    if not (tok_dir / "tokenizer.json").exists():
        print(f"training BPE tokenizer V={args.vocab_size} ...")
        train_bpe_tokenizer(
            ["data/en_train.txt", "data/hi_train.txt"], args.vocab_size, tok_dir
        )
    tokenizer = load_tokenizer(tok_dir)
    V = tokenizer.get_vocab_size()
    print(f"tokenizer loaded, V={V}")

    print("extracting vocab byte strings + script labels ...")
    byte_strings = vocab_byte_strings(tokenizer)
    scripts = compute_script_labels(byte_strings)

    stats: dict = {"vocab_size": V, "seed_note": "data pipeline is deterministic (no RNG)"}

    for split in ["train", "val"]:
        bin_path = Path(f"data/{split}{args.tag}.bin")
        if not bin_path.exists():
            total = 0
            tmp = bin_path.with_suffix(".tmp")
            # concatenate en+hi for this split into one array
            with open(tmp, "wb") as out:
                pass
            counts = {}
            for lang in ["en", "hi"]:
                part = Path(f"data/{split}{args.tag}_{lang}.part")
                n = tokenize_to_memmap(tokenizer, f"data/{lang}_{split}.txt", part, dtype=args.dtype)
                counts[lang] = n
                total += n
                with open(tmp, "ab") as out, open(part, "rb") as src:
                    out.write(src.read())
                part.unlink()
            tmp.rename(bin_path)
            stats[f"{split}_tokens"] = total
            stats[f"{split}_tokens_per_lang"] = counts
            print(f"{split}: {total:,} tokens {counts}")
        else:
            data = np.memmap(bin_path, dtype=np.dtype(args.dtype), mode="r")
            stats[f"{split}_tokens"] = int(len(data))
            print(f"skip tokenize {split} (exists, {len(data):,} tokens)")

    print("computing per-token frequencies (train split) ...")
    freqs = token_frequencies(f"data/train{args.tag}.bin", V, dtype=args.dtype)
    save_vocab_meta(tok_dir, byte_strings, scripts, freqs)

    # per-script vocab + token share
    scripts_arr = np.asarray(scripts)
    share = {}
    for s in ["latin", "devanagari", "telugu", "other"]:
        mask = scripts_arr == s
        share[s] = {
            "vocab_entries": int(mask.sum()),
            "token_share": float(freqs[mask].sum() / max(1, freqs.sum())),
        }
    stats["script_share"] = share

    # Acceptance: lossless round-trip on held-out paragraphs + fertility
    samples = {
        "latin": "The quick brown fox jumps over the lazy dog; 1234 & symbols!",
        "devanagari": "अंतर्राष्ट्रीयकरण और अंतर्राष्ट्रीयता में अंतर है। क्षेत्रीय भाषाएँ महत्वपूर्ण हैं।",
    }
    rt = {}
    for name, text in samples.items():
        decoded = tokenizer.decode(tokenizer.encode(text).ids)
        rt[name] = {"lossless": decoded == text, "fertility": round(fertility(tokenizer, text), 3)}
        assert decoded == text, f"round-trip FAILED for {name}: {decoded!r}"
    stats["roundtrip"] = rt
    stats["wall_seconds"] = round(time.time() - t0, 1)

    out = Path(f"results/metrics/data_stats{args.tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print("M1 acceptance PASSED")


if __name__ == "__main__":
    main()
