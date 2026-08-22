"""Fetch a TinyStories slice comparable to the wiki --bytes cap."""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

URL = ("https://huggingface.co/datasets/roneneldan/TinyStories/"
       "resolve/main/TinyStoriesV2-GPT4-train.txt")
OUT = Path("data/_tinystories_train.txt")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=int, default=30_000_000)
    ap.add_argument("--url", default=URL)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size >= args.bytes:
        print(f"have {out} ({out.stat().st_size} bytes) — skip")
        return 0
    print(f"GET {args.url}  cap {args.bytes}")
    req = urllib.request.Request(args.url, headers={"User-Agent": "TapeLM-414"})
    with urllib.request.urlopen(req, timeout=120) as r:
        buf = r.read(args.bytes)
    out.write_bytes(buf)
    print(f"wrote {out} ({len(buf)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
