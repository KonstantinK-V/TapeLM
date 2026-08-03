"""
After wiki:12 (run-tag wiki12) finishes, queue overnight stream tests on GPU.

  python _run_queue_after_wiki12.py
  python _run_queue_after_wiki12.py --skip-ablation

1. Poll until results/stage255_decision_wiki12.json has 12 chunks (or log tail).
2. Multi-domain stream: wiki:6,med:3,news:3 -> run-tag stream_wmn_v1
3. Optional ablation: wiki:2,med:2 with --no-query-train -> run-tag ablate_no_wq_4ch

Logs:
  results/_run_queue_after_wiki12.log
  results/_stage255_stream_wmn_full.out
  results/_stage255_ablate_no_wq_full.out
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG_W12 = ROOT / "results" / "_stage255_wiki12_full.out"
DEC_W12 = ROOT / "results" / "stage255_decision_wiki12.json"
OUT_WMN = ROOT / "results" / "_stage255_stream_wmn_full.out"
OUT_ABL = ROOT / "results" / "_stage255_ablate_no_wq_full.out"
QLOG = ROOT / "results" / "_run_queue_after_wiki12.log"
POLL_S = 180
WMN_CHUNKS = 10
ABL_CHUNKS = 4


def log(msg: str) -> None:
    line = msg if msg.endswith("\n") else msg + "\n"
    try:
        print(line, end="", flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), end="", flush=True)
    try:
        with QLOG.open("a", encoding="utf-8") as f:
            f.write(line)
    except OSError:
        pass


def wiki12_done() -> bool:
    if DEC_W12.exists():
        try:
            d = json.loads(DEC_W12.read_text(encoding="utf-8"))
            if d.get("run_tag") == "wiki12" and d.get("summary", {}).get("chunks", 0) >= 12:
                return True
            if d.get("summary", {}).get("chunks", 0) >= 12:
                return True
        except (json.JSONDecodeError, OSError):
            pass
    if LOG_W12.exists():
        tail = LOG_W12.read_text(encoding="utf-8", errors="ignore")[-15000:]
        if "schedule exhausted" in tail and "STREAM_INGEST" in tail:
            return True
    return False


def stream_done(dec_name: str, min_chunks: int) -> bool:
    p = ROOT / "results" / dec_name
    if not p.exists():
        return False
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return d.get("summary", {}).get("chunks", 0) >= min_chunks
    except (json.JSONDecodeError, OSError):
        return False


def run_stage(cmd: list[str], out_path: Path, label: str) -> int:
    if out_path.exists() and (time.time() - out_path.stat().st_mtime) < 7200:
        log(f"skip {label}: {out_path.name} updated in last 2h (already running or just finished?)")
        return 0
    log(f"starting {label}")
    log(f"  cmd: {' '.join(cmd)}")
    try:
        with out_path.open("w", encoding="utf-8") as out:
            return subprocess.call(cmd, cwd=str(ROOT), stdout=out, stderr=subprocess.STDOUT)
    except PermissionError:
        log(f"failed {label}: could not open {out_path.name} (locked)")
        return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-ablation", action="store_true", help="only stream_wmn_v1, no --no-query-train 4-chunk run")
    ap.add_argument("--poll-s", type=int, default=POLL_S)
    args = ap.parse_args()

    log(f"waiting for wiki:12 to finish (poll {args.poll_s}s)")
    while not wiki12_done():
        if LOG_W12.exists():
            lines = [ln for ln in LOG_W12.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
            tail = (lines[-1][:120] if lines else "...").encode("ascii", "replace").decode("ascii")
            log(f"  wiki12 tail: {tail}")
        time.sleep(args.poll_s)
    log("wiki:12 done — queueing stream_wmn_v1")

    wmn_cmd = [
        sys.executable,
        str(ROOT / "_stage255_stream_ingest.py"),
        "--schedule",
        "wiki:6,med:3,news:3",
        "--chunk-lines",
        "25000",
        "--epochs-per-chunk",
        "1.0",
        "--replay-frac",
        "0.2",
        "--ckpt-every",
        "2",
        "--run-tag",
        "stream_wmn_v1",
    ]
    rc = run_stage(wmn_cmd, OUT_WMN, "stream_wmn_v1")
    log(f"stream_wmn_v1 exit={rc}")

    if args.skip_ablation:
        log("ablation skipped (--skip-ablation)")
        return rc

    if not stream_done("stage255_decision_stream_wmn_v1.json", WMN_CHUNKS):
        log("stream_wmn_v1 did not reach 12 chunks in decision — skipping ablation")
        return rc

    log("stream_wmn_v1 done — starting ablation (no W_q, 4 chunks)")
    abl_cmd = [
        sys.executable,
        str(ROOT / "_stage255_stream_ingest.py"),
        "--schedule",
        "wiki:2,med:2",
        "--chunk-lines",
        "25000",
        "--epochs-per-chunk",
        "1.0",
        "--replay-frac",
        "0.2",
        "--ckpt-every",
        "2",
        "--run-tag",
        "ablate_no_wq_4ch",
        "--no-query-train",
    ]
    rc2 = run_stage(abl_cmd, OUT_ABL, "ablate_no_wq_4ch")
    log(f"ablate_no_wq_4ch exit={rc2}")
    return rc2 if rc2 else rc


if __name__ == "__main__":
    raise SystemExit(main())
