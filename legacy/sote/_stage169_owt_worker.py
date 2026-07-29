"""
Stage169 helper: stream OpenWebText → local token cache (isolates HF/datasets crashes).

Writes lines of space-separated tokens to OUT path until max_tokens.

  python _stage169_owt_worker.py --out data/_owt_tokens_cache.txt --max-tokens 400000000
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?|[.,!?;:\"'()\[\]{}\-%/$]")
PUNCT_CHARS = set(".,!?;:\"'()[]{}-%/$")
MAX_WORD_LEN = 24


def tokenize_rich(line: str) -> list[str]:
    out = []
    for t in TOKEN_RE.findall(line):
        if t in PUNCT_CHARS:
            out.append(t)
        elif len(re.sub(r"[^a-z0-9]", "", t.lower())) <= MAX_WORD_LEN:
            out.append(t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, required=True)
    ap.add_argument("--log-every", type=int, default=2_000_000)
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = out.with_suffix(out.suffix + ".meta")

    # Resume if meta says we already have enough
    if out.exists() and meta.exists():
        try:
            have = int(meta.read_text(encoding="utf-8").strip().split()[0])
            if have >= args.max_tokens:
                print(f"[owt-worker] cache already complete have={have}", flush=True)
                return 0
        except Exception:
            pass

    from datasets import load_dataset

    print(f"[owt-worker] load Skylion007/openwebtext streaming ...", flush=True)
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    n = 0
    t0 = time.time()
    # append mode if partial
    mode = "a" if out.exists() else "w"
    if mode == "a" and meta.exists():
        try:
            n = int(meta.read_text(encoding="utf-8").strip().split()[0])
        except Exception:
            n = 0
    with out.open(mode, encoding="utf-8") as f:
        for row in ds:
            text = row.get("text") or ""
            for line in text.splitlines():
                toks = tokenize_rich(line)
                if not toks:
                    continue
                left = args.max_tokens - n
                if left <= 0:
                    break
                if len(toks) > left:
                    toks = toks[:left]
                f.write(" ".join(toks) + "\n")
                n += len(toks)
                if n % args.log_every < len(toks):
                    meta.write_text(f"{n}\n", encoding="utf-8")
                    f.flush()
                    print(
                        f"[owt-worker] tokens={n}/{args.max_tokens} "
                        f"wall={(time.time()-t0)/60:.1f}m",
                        flush=True,
                    )
                if n >= args.max_tokens:
                    break
            if n >= args.max_tokens:
                break
    meta.write_text(f"{n}\n", encoding="utf-8")
    print(f"[owt-worker] DONE n={n} path={out}", flush=True)
    return 0 if n > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
