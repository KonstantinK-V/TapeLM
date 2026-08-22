"""528 stories ladder: ceiling first, then learner. Wiki/news thin-ceil = VOID."""
from __future__ import annotations

import subprocess
import sys

PY = sys.executable
ST = "data/_tinystories_train.txt"
NEWS = "data/_stage254_news.txt"
SEEDS = (1337, 8642, 2890)


def run(cmd):
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    p = subprocess.run(cmd)
    print(f"exit {p.returncode}", flush=True)
    return p.returncode


def main() -> int:
    rc = run([PY, "_check528_step.py"])
    # wiki 2400 → expect VOID thin ceiling
    rc |= run([PY, "_audit528_step.py", "--seed", "1337", "--window-lines", "2400"])
    # news 2400 → VOID
    rc |= run([PY, "_audit528_step.py", "--corpus", NEWS, "--seed", "1337",
               "--window-lines", "2400"])
    # stories: 400 → 1200 → 2400, 3 seeds
    for L in (400, 1200, 2400):
        for s in SEEDS:
            rc |= run([PY, "_audit528_step.py", "--corpus", ST, "--seed", str(s),
                       "--window-lines", str(L), "--epochs", "6"])
    print("\n528 ladder done", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
