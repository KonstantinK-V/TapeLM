"""414 on two bigger tapes: wiki×10 and TinyStories. Same arms, no Phi."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SEEDS = (1337, 8642, 2890)
PY = sys.executable
AUDIT = "_audit414_construct.py"
JOBS = (
    ("wiki4k",
     Path("data/_wikitext103_train.txt"),
     Path("results/_stage414_wiki4k.json"),
     ["--min-line", "80", "--window-lines", "4000"]),
    ("stories",
     Path("data/_tinystories_train.txt"),
     Path("results/_stage414_stories.json"),
     ["--min-line", "20", "--window-lines", "4000"]),
)


def run(job, corpus, out, extra) -> int:
    if not corpus.exists():
        print(f"SKIP {job}: no {corpus}")
        return 1
    rc = 0
    for seed in SEEDS:
        cmd = [PY, AUDIT, "--corpus", str(corpus), "--out", str(out),
               "--seed", str(seed), *extra]
        print("\n==", job, "seed", seed, "==")
        print(" ", " ".join(cmd))
        p = subprocess.run(cmd)
        if p.returncode != 0:
            rc = p.returncode
    return rc


def main() -> int:
    rc = 0
    for job, corpus, out, extra in JOBS:
        r = run(job, corpus, out, extra)
        if r:
            rc = r
    print("\nDone. Compare window vs left inside each json — not across tapes.")
    print("  wiki4k   results/_stage414_wiki4k.json")
    print("  stories  results/_stage414_stories.json")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
