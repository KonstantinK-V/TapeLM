"""Until ~morning: after 489+, run 494–498 signal hunts (multi-seed).

Fills the rest of the night. Not a ladder. Any lift/drop is a clue.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

LOG = Path("results/_stage_morning_night.log")
WAIT = Path("results/_stage489_plus_night.log")
PY = sys.executable


def wait_flag(path: Path, needle: str, timeout_s: int = 14400) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if path.exists() and needle in path.read_text(encoding="utf-8", errors="ignore"):
            print(f"saw {needle!r}, continuing", flush=True)
            return
        time.sleep(40)
    print(f"timeout waiting {path}, continuing anyway", flush=True)


def run(cmd: list[str]) -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"\n$ {' '.join(cmd)}\n")
        f.flush()
        return subprocess.Popen(
            cmd, stdout=f, stderr=subprocess.STDOUT,
            env={**{k: v for k, v in __import__("os").environ.items()},
                 "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
        ).wait()


def main() -> int:
    wait_flag(WAIT, "489+ done")
    LOG.write_text(f"=== morning pack {time.strftime('%Y-%m-%dT%H:%M:%S')} ===\n",
                   encoding="utf-8")
    rc = 0
    jobs = []
    # 491 already in 489+; reinforce with seed 2024
    jobs += [
        [PY, "-u", "_audit491_shuffle.py", "--seed", "2024", "--steps", "2500",
         "--log-every", "500"],
    ]
    for s in (1337, 2024, 7):
        jobs.append([PY, "-u", "_audit494_rarity.py", "--seed", str(s),
                     "--steps", "2500", "--log-every", "500"])
    for s in (1337, 7):
        jobs.append([PY, "-u", "_audit495_curriculum.py", "--seed", str(s),
                     "--train-steps", "2500", "--eval-steps", "1500",
                     "--log-every", "500"])
    for s in (1337, 2024, 7):
        jobs.append([PY, "-u", "_audit496_const.py", "--seed", str(s),
                     "--steps", "2500", "--log-every", "500"])
    for s in (1337, 2024, 7):
        jobs.append([PY, "-u", "_audit497_refuse.py", "--seed", str(s),
                     "--steps", "3000", "--log-every", "500"])
    for s in (1337, 7):
        jobs.append([PY, "-u", "_audit498_budget.py", "--seed", str(s),
                     "--steps", "2000", "--log-every", "500"])
    # extra wiki + transfer directions
    jobs += [
        [PY, "-u", "_audit493_wiki.py", "--seed", "7", "--steps", "2500",
         "--log-every", "500"],
        [PY, "-u", "_audit489_transfer.py", "--train-seed", "2024",
         "--eval-seed", "7", "--train-steps", "2500", "--eval-steps", "1500"],
        [PY, "-u", "_audit486_unique_only.py", "--seed", "42", "--steps", "4000",
         "--log-every", "500"],
        [PY, "-u", "_audit486_unique_only.py", "--seed", "99", "--steps", "4000",
         "--log-every", "500"],
        # larger windows / bigger read — stress unique density
        [PY, "-u", "_audit486_unique_only.py", "--seed", "1337", "--steps", "2000",
         "--window", "1200", "--bytes", "60000000", "--log-every", "400",
         "--out", "results/_stage486_unique_wide.json"],
        [PY, "-u", "_audit487_arrive.py", "--seed", "42", "--steps", "3000",
         "--log-every", "400"],
        [PY, "-u", "_audit488_chain.py", "--seed", "42", "--steps", "3000",
         "--log-every", "400"],
        # max-soft free swim (last freestyle)
        [PY, "-u", "_audit499_free_swim.py", "--seed", "1337", "--steps", "2500",
         "--max-hops", "5", "--log-every", "500"],
        [PY, "-u", "_audit499_free_swim.py", "--seed", "7", "--steps", "2500",
         "--max-hops", "5", "--log-every", "500"],
        [PY, "-u", "_audit499_free_swim.py", "--seed", "2024", "--steps", "2000",
         "--max-hops", "8", "--log-every", "400"],
        [PY, "-u", "_audit500_maxloose.py", "--seed", "1337", "--steps", "3000",
         "--max-hops", "8", "--log-every", "500"],
        [PY, "-u", "_audit500_maxloose.py", "--seed", "2024", "--steps", "3000",
         "--max-hops", "8", "--log-every", "500"],
        [PY, "-u", "_audit500_maxloose.py", "--seed", "7", "--steps", "3000",
         "--max-hops", "12", "--log-every", "500"],
        [PY, "-u", "_audit500_maxloose.py", "--seed", "42", "--steps", "4000",
         "--max-hops", "12", "--window", "1000", "--log-every", "500"],
    ]

    for i, cmd in enumerate(jobs, 1):
        print(f"[{i}/{len(jobs)}] {' '.join(cmd[2:5])} ...", flush=True)
        r = run(cmd)
        rc = rc or r
        print(f"  exit {r}", flush=True)

    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"\n=== morning pack done {time.strftime('%Y-%m-%dT%H:%M:%S')} ===\n")
    # digest
    for p in sorted(Path("results").glob("_stage49*.json")):
        try:
            keys = list(json.loads(p.read_text(encoding="utf-8")).keys())
            print(p.name, keys, flush=True)
        except Exception as e:
            print(p.name, "err", e, flush=True)
    print("morning pack finished", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
