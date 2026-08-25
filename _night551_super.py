"""551 super night: all exits from 550 filter / 551 rerank dead-end.

Checks, baselines (550/551), full sweep (15 modes x mate splits),
multi-seed, wide-window spot checks. Not a gate queue — any GO is a clue.

Log:  results/_stage551_super_night.log
Index: results/_NIGHT551_SUPER_INDEX.txt

    python -u _night551_super.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

LOG = Path("results/_stage551_super_night.log")
INDEX = Path("results/_NIGHT551_SUPER_INDEX.txt")
PY = sys.executable
STORIES = "data/_tinystories_train.txt"
SEEDS = (1337, 8642, 2890, 42, 7)
SWEEP_MODES = (
    "gl,filter,rn_mg,rn_m,rn_stable,rn_anchor,rn_allenv,rn_jacc,rn_union,"
    "rn_cond,rn_idf,rn_boost,rn_peaked,rn_allgo,rn_distgo"
)


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
    LOG.write_text(f"=== 551 super night {time.strftime('%Y-%m-%dT%H:%M:%S')} ===\n",
                   encoding="utf-8")
    jobs: list[list[str]] = []

    jobs.append([PY, "-u", "_check550_bundle.py"])
    jobs.append([PY, "-u", "_check551_rerank.py"])

    for s in SEEDS:
        jobs.append([PY, "-u", "_audit550_bundle.py", "--seed", str(s),
                     "--corpus", STORIES])
        jobs.append([PY, "-u", "_audit551_rerank.py", "--seed", str(s),
                     "--corpus", STORIES])
        jobs.append([PY, "-u", "_audit551_sweep.py", "--seed", str(s),
                     "--corpus", STORIES, "--modes", SWEEP_MODES])

    # wide window / more lines — does rerank need more context?
    for s in (1337, 8642):
        jobs.append([PY, "-u", "_audit551_sweep.py", "--seed", str(s),
                     "--corpus", STORIES, "--modes", SWEEP_MODES,
                     "--window-lines", "800", "--n-win", "16", "--lines", "40000",
                     "--out", "results/_stage551_sweep_wide.json"])
        jobs.append([PY, "-u", "_audit551_rerank.py", "--seed", str(s),
                     "--corpus", STORIES, "--window-lines", "800", "--n-win", "16",
                     "--lines", "40000",
                     "--out", "results/_stage551_rerank_wide.json"])

    # related exits: 549 depth, 536 rnd reorder control
    for s in (1337, 8642, 2890):
        jobs.append([PY, "-u", "_audit549_depth.py", "--seed", str(s),
                     "--corpus", STORIES])
    for s in (1337, 2890):
        jobs.append([PY, "-u", "_audit536_rnd.py", "--seed", str(s),
                     "--corpus", STORIES])

    # fallback corpus if stories path missing in env
    if Path("data/external_tinystories_mini.txt").exists():
        jobs.append([PY, "-u", "_audit551_sweep.py", "--seed", "1337",
                     "--corpus", "data/external_tinystories_mini.txt",
                     "--modes", SWEEP_MODES, "--bytes", "20000000",
                     "--out", "results/_stage551_sweep_mini.json"])

    rc = 0
    t0 = time.time()
    for i, cmd in enumerate(jobs, 1):
        tag = " ".join(cmd[2:6])
        print(f"[{i}/{len(jobs)}] {tag} ...", flush=True)
        r = run(cmd)
        rc = rc or r
        print(f"  exit {r}  elapsed {(time.time() - t0) / 60:.1f}m", flush=True)

    summary_lines = [
        f"# 551 super night done {time.strftime('%Y-%m-%dT%H:%M:%S')} "
        f"({(time.time() - t0) / 60:.1f}m)",
        "",
        "## Baselines",
        "- 550 filter   results/_stage550_bundle.json",
        "- 551 rerank   results/_stage551_rerank.json",
        "",
        "## Sweep (15 modes x any/anchor/allenv mate defs)",
        "- results/_stage551_sweep.json",
        "- wide         results/_stage551_sweep_wide.json",
        "- mini         results/_stage551_sweep_mini.json",
        "",
        "## Related",
        "- 549 depth    results/_stage549_depth.json",
        "- 536 rnd      results/_stage536_rnd.json",
        "",
        "Morning: grep gate.*true in sweep JSON; best d_gl per seed.",
    ]
    INDEX.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"\n=== done {(time.time() - t0) / 60:.1f}m ===\n")
        for p in ("_stage551_sweep.json", "_stage551_sweep_wide.json",
                  "_stage550_bundle.json", "_stage551_rerank.json"):
            fp = Path("results") / p
            if not fp.exists():
                continue
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
                f.write(f"\n--- {p} ---\n")
                for seed, rec in data.items():
                    if "arms" in rec:
                        best = rec.get("best", "?")
                        bd = rec.get("best_d_gl", 0)
                        ng = rec.get("n_gate", 0)
                        f.write(f"  seed {seed}: best={best} d={bd:+.4f} gates={ng}\n")
                    else:
                        gate = rec.get("gate", rec.get("d_gl", "?"))
                        f.write(f"  seed {seed}: gate={gate} d_gl={rec.get('d_gl', '?')}\n")
            except Exception as e:
                f.write(f"  {p}: err {e}\n")

    print("551 super night finished", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
