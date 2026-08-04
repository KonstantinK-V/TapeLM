"""
Bedtime GPU queue (updated plan):

  255 ablate wiki12 (skip if done) → 260f full → 257 → 258 → 261 (smoke→full) → 242 → 212b

  python _run_queue_night_full.py --skip-255
  python _run_queue_night_full.py --skip-255 --wait-260f --from-stage 257   # 260f already running
  python _run_queue_night_full.py --from-stage 260f
  python _run_queue_night_full.py --force

Logs: results/_run_queue_night_full.log
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
QLOG = RES / "_run_queue_night_full.log"

OUT_255 = RES / "_stage255_ablate_no_wq_wiki12_full.out"
DEC_255 = RES / "stage255_decision_ablate_no_wq_wiki12.json"
OUT_260F = RES / "_stage260f_full.out"
OUT_257 = RES / "_stage257_full.out"
DEC_257 = RES / "stage257_decision.json"
OUT_258 = RES / "_stage258_full.out"
DEC_258 = RES / "stage258_decision.json"
DEC_261 = RES / "stage261_decision.json"
OUT_261 = RES / "_stage261_full.out"
OUT_261_SMOKE = RES / "_stage261_smoke.out"
SCRIPT_261 = ROOT / "_stage261_nl_query.py"
OUT_242 = RES / "_stage242_full.out"
DEC_242 = RES / "stage242_decision.json"
OUT_212B = RES / "_stage212b_full.out"
DEC_212B = RES / "stage212b_decision.json"

ABL_W12_CHUNKS = 12
STREAM_COMMON = [
    "--chunk-lines",
    "25000",
    "--epochs-per-chunk",
    "1.0",
    "--replay-frac",
    "0.2",
    "--ckpt-every",
    "2",
]

STAGE_ORDER = ("255", "260f", "257", "258", "261", "242", "212b")


def log(msg: str) -> None:
    line = msg if msg.endswith("\n") else msg + "\n"
    try:
        print(line, end="", flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), end="", flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    with QLOG.open("a", encoding="utf-8") as f:
        f.write(line)


def read_dec(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def ablate_wiki12_done() -> bool:
    d = read_dec(DEC_255)
    if d and d.get("summary", {}).get("chunks", 0) >= ABL_W12_CHUNKS:
        return True
    if OUT_255.exists():
        tail = OUT_255.read_text(encoding="utf-8", errors="ignore")[-20000:]
        if "schedule exhausted" in tail and "STREAM_INGEST" in tail:
            return True
    return False


def run_cmd(cmd: list[str], out_path: Path, label: str, *, append: bool = False) -> int:
    log(f"starting {label}")
    log(f"  cmd: {' '.join(cmd)}")
    mode = "a" if append else "w"
    with out_path.open(mode, encoding="utf-8") as out:
        if append:
            out.write(f"\n--- queue restart {time.strftime('%Y-%m-%dT%H:%M:%S')} ---\n")
        return subprocess.call(cmd, cwd=str(ROOT), stdout=out, stderr=subprocess.STDOUT)


def ingest_ablate_wiki12(*, resume: bool) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / "_stage255_stream_ingest.py"),
        "--schedule",
        "wiki:12",
        *STREAM_COMMON,
        "--run-tag",
        "ablate_no_wq_wiki12",
        "--no-query-train",
    ]
    if resume:
        cmd.append("--resume")
    return cmd


def nl261_smoke_warrants_full(d: dict | None) -> bool:
    """Run full after smoke if the claim is still live (not NO / lexical-only)."""
    if not d or d.get("stage") != 261:
        return False
    return d.get("overall") in ("NL_QUERY_OK", "NL_QUERY_PARTIAL", "NL_QUERY_NWAY_ONLY")


def stage_script(name: str) -> Path | None:
    if name == "261":
        return SCRIPT_261 if SCRIPT_261.is_file() else None
    p = ROOT / f"_stage{name}_open_gate.py" if name in ("260b", "260c", "260d", "260e", "260f") else ROOT / f"_stage{name}_instance_sem.py" if name == "212b" else ROOT / f"_stage{name}_rehearsal_dose.py" if name == "242" else ROOT / f"_stage{name}_fp_compose.py" if name == "257" else ROOT / f"_stage{name}_semantic_query.py" if name == "258" else None
    if p is None:
        return None
    return p if p.is_file() else None


def done(name: str, force: bool) -> bool:
    if force:
        return False
    if name == "255":
        return ablate_wiki12_done()
    if name == "260f":
        d = read_dec(RES / "stage260f_decision.json")
        return bool(d and d.get("stage") == "260f" and int(d.get("steps", 0)) >= 2500)
    if name == "257":
        d = read_dec(DEC_257)
        return bool(d and d.get("stage") == 257 and int(d.get("steps", 0)) >= 800)
    if name == "258":
        d = read_dec(DEC_258)
        return bool(d and d.get("stage") == 258 and int(d.get("n_subjects", 0)) >= 64)
    if name == "261":
        d = read_dec(DEC_261)
        return bool(d and d.get("stage") == 261 and int(d.get("steps", 0)) >= 800)
    if name == "242":
        d = read_dec(DEC_242)
        if not d or d.get("stage") != 242:
            return False
        return "dose_wall_s" in d and len(d.get("curve") or {}) >= 6
    if name == "212b":
        d = read_dec(DEC_212B)
        return bool(d and d.get("stage") in ("212b", 212) and d.get("overall"))
    return False


def run_stage(name: str) -> int:
    if name == "255":
        resume = OUT_255.exists() or (RES / "stream255" / "ablate_no_wq_wiki12").exists()
        return run_cmd(
            ingest_ablate_wiki12(resume=resume),
            OUT_255,
            "255 ablate_no_wq_wiki12",
            append=resume and OUT_255.exists(),
        )
    if name == "260f":
        return run_cmd(
            [sys.executable, str(ROOT / "_stage260f_open_gate.py")],
            OUT_260F,
            "260f full",
        )
    if name == "257":
        return run_cmd([sys.executable, str(ROOT / "_stage257_fp_compose.py")], OUT_257, "257 full")
    if name == "258":
        rc = run_cmd([sys.executable, str(ROOT / "_stage258_semantic_query.py")], OUT_258, "258 full")
        d = read_dec(DEC_258)
        if d:
            gpt = (d.get("summary") or {}).get("gpt_control") or {}
            up = (gpt.get("unseen_para") or {}).get("sel_acc")
            cs = ((d.get("summary") or {}).get("curve_sem") or {}).get("unseen_para") or {}
            cu = cs.get("sel_acc")
            if up is not None and cu is not None:
                log(f"258 unseen_para: curve={cu:.3f} gpt={up:.3f} delta={cu - up:+.3f}")
        return rc
    if name == "261":
        script = stage_script("261")
        if script is None:
            log("261: _stage261_nl_query.py not in repo — skip")
            return 0
        rc = run_cmd([sys.executable, str(script), "--smoke"], OUT_261_SMOKE, "261 smoke")
        if rc != 0:
            return rc
        d = read_dec(DEC_261)
        overall = (d or {}).get("overall", "?")
        if not nl261_smoke_warrants_full(d):
            log(f"261 smoke overall={overall} — skip full (need NL_QUERY_OK or NL_QUERY_PARTIAL)")
            return 0
        log(f"261 smoke overall={overall} — running full")
        return run_cmd([sys.executable, str(script)], OUT_261, "261 full")
    if name == "242":
        return run_cmd([sys.executable, str(ROOT / "_stage242_rehearsal_dose.py")], OUT_242, "242 full grid")
    if name == "212b":
        return run_cmd([sys.executable, str(ROOT / "_stage212b_instance_sem.py")], OUT_212B, "212b full")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-255", action="store_true")
    ap.add_argument(
        "--wait-260f",
        action="store_true",
        help="poll until stage260f full (steps>=2500) then continue queue (use with --from-stage 257)",
    )
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--from-stage",
        type=str,
        default="255",
        choices=STAGE_ORDER,
        help="first stage to run (255 skipped if --skip-255)",
    )
    args = ap.parse_args()

    QLOG.write_text("", encoding="utf-8")
    log(f"night queue: from={args.from_stage} skip_255={args.skip_255} wait_260f={args.wait_260f} force={args.force}")

    if args.wait_260f:
        log("waiting for 260f full (decision steps>=2500)...")
        while not done("260f", args.force):
            time.sleep(120)
            log("  still waiting for stage260f_decision.json")
        log("260f complete — continuing queue")

    start_i = STAGE_ORDER.index(args.from_stage)
    for name in STAGE_ORDER[start_i:]:
        if name == "255" and args.skip_255:
            log("255 skipped (--skip-255)")
            continue
        if done(name, args.force):
            log(f"{name} already complete — skip")
            continue
        rc = run_stage(name)
        log(f"{name} exit={rc}")
        if rc != 0:
            return rc

    log("night queue finished OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
