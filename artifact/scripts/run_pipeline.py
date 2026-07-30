#!/usr/bin/env python3
"""Run TapeLM extension pipeline stages 213–220 from repo root."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# (stage_id, script, decision_file)
PIPELINE = [
    (213, "_stage213_arc_enc_freeze_finetune.py", "stage213_decision.json"),
    (214, "_stage214_recency_ctx.py", "stage214_decision.json"),
    (215, "_stage215_domain_adapter.py", "stage215_decision.json"),
    (216, "_stage216_split_arc_ff.py", "stage216_decision.json"),
    (217, "_stage217_slow_endpoint_slots.py", "stage217_decision.json"),
    (218, "_stage218_snap_hop.py", "stage218_decision.json"),
    (219, "_stage219_stream_decay.py", "stage219_decision.json"),
    (220, "_stage220_sem_sidecar.py", "stage220_decision.json"),
    (221, "_stage221_fp_remap_adapter.py", "stage221_decision.json"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description="TapeLM extension pipeline 213–220")
    ap.add_argument("--from", dest="from_stage", type=int, default=213)
    ap.add_argument("--to", dest="to_stage", type=int, default=220)
    ap.add_argument("--smoke", action="store_true", help="Pass --smoke to each stage script")
    ap.add_argument("--force", action="store_true", help="Run even if decision.json exists")
    ap.add_argument("--list", action="store_true", help="List stages and exit")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.list:
        for sid, script, dec in PIPELINE:
            path = REPO / script
            ok = "ok" if path.exists() else "MISSING"
            print(f"  {sid}  {script}  [{ok}]")
        print(f"\nDocs: {REPO / 'docs/EXTENSION_PIPELINE.md'}")
        return 0

    rc = 0
    for sid, script, dec_name in PIPELINE:
        if sid < args.from_stage or sid > args.to_stage:
            continue
        script_path = REPO / script
        dec_path = REPO / "results" / dec_name
        if not script_path.exists():
            print(f"[{sid}] SKIP missing {script}")
            continue
        if dec_path.exists() and not args.force:
            print(f"[{sid}] SKIP existing {dec_name} (use --force)")
            continue
        cmd = [sys.executable, str(script_path)]
        if args.smoke:
            cmd.append("--smoke")
        print(f"[{sid}] RUN {' '.join(cmd)}")
        if args.dry_run:
            continue
        r = subprocess.run(cmd, cwd=str(REPO))
        if r.returncode != 0:
            print(f"[{sid}] FAILED exit {r.returncode}")
            rc = r.returncode
            break
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
