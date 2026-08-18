"""Overnight: wait m2b -> 282 smoke -> if not hard-fail, full then --no-probe."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
M2B = RES / "stage280_decision_fp_m2b.json"
SMOKE_OUT = RES / "_stage282_smoke.out"
REVIEW = RES / "_stage282_overnight_review.txt"
FULL_OUT = RES / "_stage282_full_m2.out"
NOPROBE_OUT = RES / "_stage282_full_m2_noprobe.out"


def log(msg: str) -> None:
    print(msg, flush=True)


def m2b_alive() -> bool:
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Select-Object -ExpandProperty CommandLine"],
            text=True, stderr=subprocess.DEVNULL, cwd=str(ROOT),
        )
    except Exception:
        return False
    for line in out.splitlines():
        if "_stage280_raw_exam.py" in line and "m2b" in line:
            return True
    return False


def run(cmd: list[str], out_path: Path) -> int:
    log(f"RUN {' '.join(cmd)}")
    with out_path.open("w", encoding="utf-8", errors="replace") as f:
        import os
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
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
    return files[-1] if files else None


def summarize_m2b() -> None:
    d = json.loads(M2B.read_text(encoding="utf-8"))
    h = d.get("held_out") or {}
    t = h.get("tie") or {}
    log(f"m2b overall={d.get('overall')} ceil={d.get('teacher_ceiling_reward')} "
        f"rew={h.get('reward_total')}")
    log(f"tie n={t.get('n')} teacher_abstain={t.get('teacher_abstain')} "
        f"policy_abstain={t.get('abstain')} stall={h.get('stall_rate')}")


def main() -> int:
    log(f"=== overnight 282 orchestrator start ===")
    log("waiting for m2b...")
    while True:
        if (not m2b_alive()) and M2B.exists():
            break
        time.sleep(90)
    log("m2b done")
    summarize_m2b()

    code = run([sys.executable, "_stage282_mind.py", "--smoke"], SMOKE_OUT)
    log(f"282 smoke exit={code}")
    dec = latest_decision()
    tail = SMOKE_OUT.read_text(encoding="utf-8", errors="replace")[-4000:] if SMOKE_OUT.exists() else ""
    hard = False
    reason = ""
    if code != 0:
        hard, reason = True, f"smoke exit {code}"
    for needle in ("Traceback", "CUDA out of memory", "No module named"):
        if needle in tail:
            hard, reason = True, f"found {needle}"
            break
    if dec is None:
        hard, reason = True, "no stage282_decision*.json"

    lines = [f"smoke_exit={code}", f"decision={dec}", f"hard_fail={hard} reason={reason}"]
    if dec is not None:
        d = json.loads(dec.read_text(encoding="utf-8"))
        lines.append(f"overall={d.get('overall')}")
        lines.append(f"gates={json.dumps(d.get('gates'), ensure_ascii=True)}")
        (RES / "stage282_decision_smoke.json").write_text(
            json.dumps(d, indent=2), encoding="utf-8")
    lines.append("---- smoke tail ----")
    lines.append(tail)
    REVIEW.write_text("\n".join(lines), encoding="utf-8")
    log("\n".join(lines[:8]))

    if hard:
        log(f"HARD FAIL — stop before fulls. See {REVIEW}")
        return 2

    # Soft fail (MIND_NO / PARTIAL / TEACHER_UNUSABLE) still runs fulls.
    log("soft-ok — launching 282 full (min-mentions 2)")
    run([sys.executable, "_stage282_mind.py",
         "--bc-episodes", "4000", "--rl-episodes", "3000", "--min-mentions", "2"], FULL_OUT)
    dec = latest_decision()
    if dec:
        (RES / "stage282_decision_full_m2.json").write_text(
            dec.read_text(encoding="utf-8"), encoding="utf-8")

    log("launching 282 full --no-probe")
    run([sys.executable, "_stage282_mind.py",
         "--bc-episodes", "4000", "--rl-episodes", "3000", "--min-mentions", "2",
         "--no-probe"], NOPROBE_OUT)
    dec = latest_decision()
    if dec:
        (RES / "stage282_decision_full_m2_noprobe.json").write_text(
            dec.read_text(encoding="utf-8"), encoding="utf-8")

    log("=== overnight 282 orchestrator done ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
