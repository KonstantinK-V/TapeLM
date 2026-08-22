"""Night driver: 485 hunt (3 seeds) then 486 unique-only (3 seeds). UTF-8 log."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

LOG = Path("results/_stage485_hunt_night.log")
PY = sys.executable


def run(cmd: list[str]) -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"\n$ {' '.join(cmd)}\n")
        f.flush()
        p = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env={**dict(**{k: v for k, v in __import__("os").environ.items()}),
                 "PYTHONUNBUFFERED": "1",
                 "PYTHONIOENCODING": "utf-8"},
        )
        return p.wait()


def main() -> int:
    LOG.write_text(f"=== night restart {time.strftime('%Y-%m-%dT%H:%M:%S')} ===\n",
                   encoding="utf-8")
    rc = 0
    for s in (1337, 2024, 7):
        c = [
            PY, "-u", "_audit485_hunt.py",
            "--seed", str(s), "--steps", "5000",
            "--bytes", "40000000", "--window", "700",
            "--frame-max", "4", "--hunt-budget", "16",
            "--narrow-k", "8", "--log-every", "500",
        ]
        print(f"485 seed {s}...", flush=True)
        r = run(c)
        rc = rc or r
        print(f"485 seed {s} exit {r}", flush=True)

    for s in (1337, 2024, 7):
        c = [
            PY, "-u", "_audit486_unique_only.py",
            "--seed", str(s), "--steps", "3000",
            "--bytes", "40000000", "--window", "700",
            "--frame-max", "4", "--budget", "16",
            "--log-every", "400",
        ]
        print(f"486 seed {s}...", flush=True)
        r = run(c)
        rc = rc or r
        print(f"486 seed {s} exit {r}", flush=True)

    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"\n=== night done {time.strftime('%Y-%m-%dT%H:%M:%S')} ===\n")
    print("night queue finished", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
