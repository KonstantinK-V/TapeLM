"""542 ladder (3 seeds) + ~5h freestyle from existing hunt arms.

Not a gate queue after 542 — anything with lift/drop is a clue.
Log: results/_stage542_freestyle_night.log
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

LOG = Path("results/_stage542_freestyle_night.log")
PY = sys.executable


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
    LOG.write_text(f"=== 542+freestyle {time.strftime('%Y-%m-%dT%H:%M:%S')} ===\n",
                   encoding="utf-8")
    jobs: list[list[str]] = []

    # --- ladder: 542 (paired windows + null + counted work) ---
    jobs.append([PY, "-u", "_check542_curric.py"])
    for s in (1337, 8642, 2890):
        jobs.append([PY, "-u", "_audit542_curric.py", "--seed", str(s),
                     "--train-steps", "2500", "--eval-steps", "1500"])

    # --- freestyle: signal hunts (not ladder gates) ---
    jobs.append([PY, "-u", "_audit491_shuffle.py", "--seed", "2024",
                 "--steps", "2500", "--log-every", "500"])
    for s in (1337, 8642, 2890):
        jobs.append([PY, "-u", "_audit494_rarity.py", "--seed", str(s),
                     "--steps", "2500", "--log-every", "500"])
    for s in (1337, 7, 2024):
        jobs.append([PY, "-u", "_audit496_const.py", "--seed", str(s),
                     "--steps", "2500", "--log-every", "500"])
    for s in (1337, 8642, 2890):
        jobs.append([PY, "-u", "_audit497_refuse.py", "--seed", str(s),
                     "--steps", "3000", "--log-every", "500"])
    for s in (1337, 2890):
        jobs.append([PY, "-u", "_audit498_budget.py", "--seed", str(s),
                     "--steps", "2000", "--log-every", "500"])
    jobs.append([PY, "-u", "_audit493_wiki.py", "--seed", "7",
                 "--steps", "2500", "--log-every", "500"])
    jobs.append([PY, "-u", "_audit489_transfer.py", "--train-seed", "2024",
                 "--eval-seed", "7", "--train-steps", "2500", "--eval-steps", "1500"])
    for s in (1337, 8642, 2890, 42):
        jobs.append([PY, "-u", "_audit486_unique_only.py", "--seed", str(s),
                     "--steps", "4000", "--log-every", "500"])
    jobs.append([PY, "-u", "_audit486_unique_only.py", "--seed", "1337",
                 "--steps", "2000", "--window", "1200", "--bytes", "60000000",
                 "--log-every", "400", "--out", "results/_stage486_unique_wide.json"])
    for s in (1337, 8642):
        jobs.append([PY, "-u", "_audit487_arrive.py", "--seed", str(s),
                     "--steps", "3000", "--log-every", "400"])
    for s in (1337, 2890):
        jobs.append([PY, "-u", "_audit488_chain.py", "--seed", str(s),
                     "--steps", "3000", "--log-every", "400"])
    for s, hops in ((1337, 5), (8642, 8), (2890, 8)):
        jobs.append([PY, "-u", "_audit499_free_swim.py", "--seed", str(s),
                     "--steps", "2500", "--max-hops", str(hops), "--log-every", "500"])
    for s, hops in ((1337, 8), (8642, 12), (2890, 12), (42, 12)):
        jobs.append([PY, "-u", "_audit500_maxloose.py", "--seed", str(s),
                     "--steps", "3000", "--max-hops", str(hops), "--log-every", "500"])
    jobs.append([PY, "-u", "_audit500_maxloose.py", "--seed", "42",
                 "--steps", "4000", "--max-hops", "12", "--window", "1000",
                 "--log-every", "500"])

    rc = 0
    t0 = time.time()
    for i, cmd in enumerate(jobs, 1):
        tag = " ".join(cmd[2:5])
        print(f"[{i}/{len(jobs)}] {tag} ...", flush=True)
        r = run(cmd)
        rc = rc or r
        elapsed = time.time() - t0
        print(f"  exit {r}  elapsed {elapsed / 60:.1f}m", flush=True)

    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"\n=== 542+freestyle done {time.strftime('%Y-%m-%dT%H:%M:%S')} "
                f"({(time.time() - t0) / 60:.1f}m) ===\n")
    for p in sorted(Path("results").glob("_stage54*.json")):
        try:
            keys = list(json.loads(p.read_text(encoding="utf-8")).keys())
            print(p.name, keys, flush=True)
        except Exception as e:
            print(p.name, "err", e, flush=True)
    print("542+freestyle finished", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
