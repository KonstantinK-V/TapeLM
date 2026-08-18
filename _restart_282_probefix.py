"""After probe-index fix: 282 smoke -> full -> --no-probe. m2b already done."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"


def log(msg: str) -> None:
    print(msg, flush=True)


def run(cmd: list[str], out_path: Path) -> int:
    log(f"RUN {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with out_path.open("w", encoding="utf-8", errors="replace") as f:
        p = subprocess.Popen(
            cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env,
        )
        assert p.stdout is not None
        for line in p.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            f.write(line)
            f.flush()
        return p.wait()


def latest_decision() -> Path | None:
    files = sorted(RES.glob("stage282_decision*.json"), key=lambda p: p.stat().st_mtime)
    # prefer non-archive names last by mtime anyway
    return files[-1] if files else None


def summarize(dec: Path | None, tag: str) -> None:
    if not dec:
        log(f"{tag}: no decision")
        return
    d = json.loads(dec.read_text(encoding="utf-8"))
    h = d.get("held_out") or {}
    log(f"{tag} overall={d.get('overall')} rew={h.get('reward_total')} "
        f"teach={h.get('teacher_reward_total')} cov={h.get('coverage_all')} "
        f"probe_hit={h.get('probe_hit_rate')} typed_conflict={h.get('conflict_when_tie')}")
    log(f"  gates={json.dumps(d.get('gates'), ensure_ascii=True)}")
    dest = RES / f"stage282_decision_{tag}.json"
    dest.write_text(json.dumps(d, indent=2), encoding="utf-8")


def main() -> int:
    log("=== 282 probe-fix restart ===")
    code = run([sys.executable, "_stage282_mind.py", "--smoke"], RES / "_stage282_smoke.out")
    summarize(latest_decision(), "smoke_probefix")
    if code != 0:
        log(f"smoke hard fail exit={code}")
        return code
    # Soft INVALID on thin smoke is ok; still run full as planned.
    log("launching full")
    run([sys.executable, "_stage282_mind.py",
         "--bc-episodes", "4000", "--rl-episodes", "3000", "--min-mentions", "2"],
        RES / "_stage282_full_m2.out")
    summarize(latest_decision(), "full_m2_probefix")

    log("launching --no-probe")
    run([sys.executable, "_stage282_mind.py",
         "--bc-episodes", "4000", "--rl-episodes", "3000", "--min-mentions", "2",
         "--no-probe"],
        RES / "_stage282_full_m2_noprobe.out")
    summarize(latest_decision(), "full_m2_noprobe_probefix")
    log("=== done ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
