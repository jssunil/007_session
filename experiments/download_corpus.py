"""Download the exact corpus shards used by this project.

Source: Hugging Face dataset `wikimedia/wikipedia`, snapshot 20231101.
Shards (exact files, sizes as of download on 2026-08-08):
  - 20231101.hi/train-00000-of-00002.parquet  (134.9 MB)  -> data/raw/hi-00000.parquet
  - 20231101.hi/train-00001-of-00002.parquet  (103.0 MB)  -> data/raw/hi-00001.parquet
  - 20231101.en/train-00000-of-00041.parquet  (420.3 MB)  -> data/raw/en-00000.parquet

Uses the system `curl` binary (bundled with Windows 10+/macOS/most Linux)
because it trusts the OS certificate store; python-requests fails behind
TLS-intercepting setups.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

BASE = "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main"
SHARDS = {
    "hi-00000.parquet": f"{BASE}/20231101.hi/train-00000-of-00002.parquet",
    "hi-00001.parquet": f"{BASE}/20231101.hi/train-00001-of-00002.parquet",
    "en-00000.parquet": f"{BASE}/20231101.en/train-00000-of-00041.parquet",
}


def main() -> None:
    raw = Path("data/raw")
    raw.mkdir(parents=True, exist_ok=True)
    for name, url in SHARDS.items():
        dest = raw / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"skip {name} (exists, {dest.stat().st_size/1e6:.1f} MB)")
            continue
        print(f"downloading {name} ...")
        subprocess.run(["curl", "-sL", "-o", str(dest), url], check=True)
        print(f"  done ({dest.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
