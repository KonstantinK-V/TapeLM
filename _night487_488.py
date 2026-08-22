"""Extend night after 485/486: 487 soft-arrive + 488 chain. 3 seeds each."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

LOG = Path("results/_stage487_488_night.log")
PY = sys.executable
FLAG = Path("results/_stage486_unique_only.json")


def wait_486(timeout_s: int = 1800) -> None:
    """Wait until seed 7 final (not partial) exists in 486 json."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if FLAG.exists():
            import json
            d = json.loads(FLAG.read_text(encoding="utf-8"))
            if "7" in d and not d["7"].get("partial"):
                print("486 complete, starting extend", flush=True)
                return
        time.sleep(20)
    print("486 wait timeout — starting extend anyway", flush=True)


def run(cmd: list[str]) -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"\n$ {' '.join(cmd)}\n")
        f.flush()
        p = subprocess.Popen(
            cmd, stdout=f, stderr=subprocess.STDOUT,
            env={**{k: v for k, v in __import__("os").environ.items()},
                 "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
        )
        return p.wait()


def main() -> int:
    wait_486()
    LOG.write_text(f"=== extend {time.strftime('%Y-%m-%dT%H:%M:%S')} ===\n",
                   encoding="utf-8")
    rc = 0
    for s in (1337, 2024, 7):
        print(f"487 seed {s}...", flush=True)
        r = run([PY, "-u", "_audit487_arrive.py", "--seed", str(s),
                 "--steps", "3000", "--bytes", "40000000", "--window", "700",
                 "--log-every", "400"])
        rc = rc or r
        print(f"487 seed {s} exit {r}", flush=True)
    for s in (1337, 2024, 7):
        print(f"488 seed {s}...", flush=True)
        r = run([PY, "-u", "_audit488_chain.py", "--seed", str(s),
                 "--steps", "3000", "--bytes", "40000000", "--window", "700",
                 "--log-every", "400"])
        rc = rc or r
        print(f"488 seed {s} exit {r}", flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"\n=== extend done {time.strftime('%Y-%m-%dT%H:%M:%S')} ===\n")
    print("extend queue finished", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
