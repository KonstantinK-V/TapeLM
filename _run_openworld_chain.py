"""Run open-world chain: smoke, cross-xfer 526, 525 counter."""
from __future__ import annotations

import subprocess
import sys

SEEDS = (1337, 8642, 2890)
NEWS = "data/_stage254_news.txt"
STORIES = "data/_tinystories_train.txt"
WIKI = "data/_wikitext103_train.txt"


def run(cmd: list[str]) -> str:
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True)
    tail = "\n".join((p.stdout + p.stderr).strip().splitlines()[-12:])
    print(tail, flush=True)
    if p.returncode:
        print(f"EXIT {p.returncode}", flush=True)
    return tail


def main() -> int:
    steps = []
    for s in SEEDS:
        steps.append(["python", "_contract_v1.py", "--corpus", NEWS, "--seed", str(s)])
    for s in SEEDS:
        steps.append(["python", "_contract_v1.py", "--corpus", STORIES, "--seed", str(s)])
    for s in SEEDS:
        steps.append(["python", "_contract_v1.py", "--home-corpus", WIKI,
                      "--corpus", NEWS, "--seed", str(s)])
    for s in SEEDS:
        steps.append(["python", "_audit526_openxfer.py", "--home-corpus", WIKI,
                      "--corpus", NEWS, "--seed", str(s)])
    for s in SEEDS:
        steps.append(["python", "_audit526_openxfer.py", "--home-corpus", WIKI,
                      "--corpus", STORIES, "--seed", str(s)])
    for corpus, tag in ((NEWS, "news"), (STORIES, "stories")):
        for s in SEEDS:
            steps.append(["python", "_audit525_confirm.py", "--corpus", corpus,
                          "--window-lines", "4800", "--seed", str(s)])
    fails = 0
    for cmd in steps:
        tail = run(cmd)
        if "EXIT" in tail and "high d2" not in tail:
            fails += 1
    print(f"\nchain done ({len(steps)} runs)")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
