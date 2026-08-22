"""More night freestyle after 487/488: transfer, shuffle-control, wiki hunt."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

LOG = Path("results/_stage489_plus_night.log")
DONE = Path("results/_stage487_488_night.log")
PY = sys.executable


def wait_extend(timeout_s: int = 7200) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if DONE.exists() and "extend done" in DONE.read_text(encoding="utf-8", errors="ignore"):
            print("487/488 done, starting 489+", flush=True)
            return
        time.sleep(30)
    print("extend wait timeout — starting 489+ anyway", flush=True)


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
    wait_extend()
    LOG.write_text(f"=== 489+ {time.strftime('%Y-%m-%dT%H:%M:%S')} ===\n",
                   encoding="utf-8")
    rc = 0
    jobs = [
        [PY, "-u", "_audit491_shuffle.py", "--seed", "1337", "--steps", "2500",
         "--log-every", "500"],
        [PY, "-u", "_audit491_shuffle.py", "--seed", "7", "--steps", "2500",
         "--log-every", "500"],
        [PY, "-u", "_audit489_transfer.py", "--train-seed", "1337",
         "--eval-seed", "2024", "--train-steps", "2500", "--eval-steps", "1500"],
        [PY, "-u", "_audit489_transfer.py", "--train-seed", "7",
         "--eval-seed", "1337", "--train-steps", "2500", "--eval-steps", "1500"],
        [PY, "-u", "_audit493_wiki.py", "--seed", "1337", "--steps", "2500",
         "--log-every", "500"],
        [PY, "-u", "_audit493_wiki.py", "--seed", "2024", "--steps", "2500",
         "--log-every", "500"],
    ]
    for cmd in jobs:
        print("run", " ".join(cmd[2:6]), "...", flush=True)
        r = run(cmd)
        rc = rc or r
        print("exit", r, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"\n=== 489+ done {time.strftime('%Y-%m-%dT%H:%M:%S')} ===\n")
    # tiny summary
    for p in ("results/_stage491_shuffle.json", "results/_stage489_transfer.json",
              "results/_stage493_wiki.json"):
        path = Path(p)
        if path.exists():
            print(p, list(json.loads(path.read_text(encoding="utf-8")).keys()), flush=True)
    print("489+ queue finished", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
