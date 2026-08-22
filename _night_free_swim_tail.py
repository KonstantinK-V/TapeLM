"""Free-swim tail: 499 + 500 (morning pack missed these — process started before edit)."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

LOG = Path("results/_stage499_500_night.log")
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
    LOG.write_text(f"=== free-swim tail {time.strftime('%Y-%m-%dT%H:%M:%S')} ===\n",
                   encoding="utf-8")
    # wait for morning pack to finish so we don't fight CPU hard
    morning = Path("results/_stage_morning_night.log")
    t0 = time.time()
    while time.time() - t0 < 7200:
        if morning.exists() and "morning pack done" in morning.read_text(
                encoding="utf-8", errors="ignore"):
            break
        time.sleep(30)
    else:
        print("morning still running — starting free-swim anyway", flush=True)

    jobs = [
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
    rc = 0
    for i, cmd in enumerate(jobs, 1):
        print(f"[{i}/{len(jobs)}] {' '.join(cmd[2:5])} ...", flush=True)
        r = run(cmd)
        rc = rc or r
        print(f"  exit {r}", flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"\n=== free-swim done {time.strftime('%Y-%m-%dT%H:%M:%S')} ===\n")
    print("free-swim finished", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
