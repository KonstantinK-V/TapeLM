#!/usr/bin/env python3
"""
TapeLM product entrypoint — recommended path for new users.

  python artifact/scripts/run_product.py              # memory demo (default)
  python artifact/scripts/run_product.py --all        # memory + full scorecard (196)
  python artifact/scripts/run_product.py --scorecard-only
  python artifact/scripts/run_product.py --check-only
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
P1 = REPO / "checkpoints" / "stage191_p1_curve.pt"


def run(script: str, extra: list[str] | None = None) -> int:
    cmd = [sys.executable, str(SCRIPTS / script), *(extra or [])]
    return subprocess.run(cmd, cwd=str(REPO)).returncode


def main() -> int:
    ap = argparse.ArgumentParser(description="TapeLM product demo")
    ap.add_argument("--all", action="store_true", help="Memory demo then Stage 196 scorecard")
    ap.add_argument("--scorecard-only", action="store_true", help="Stage 196 only")
    ap.add_argument("--check-only", action="store_true", help="Environment + checkpoints only")
    ap.add_argument("--smoke", action="store_true", help="Pass --smoke to memory demo")
    ap.add_argument("--skip-cross-domain", action="store_true")
    args = ap.parse_args()

    print("TapeLM variant A — one encoder, operable fingerprint memory\n")
    print("Shipping trunk: 221 -> 227 -> 228c -> 230 -> 226c (see README)\n")

    rc = run("check_env.py")
    if rc != 0:
        return rc

    if args.check_only:
        return 0

    if not P1.exists():
        print("\nDownload weights:")
        print("  python artifact/scripts/download_checkpoints.py")
        print("\nBrowse verdicts without GPU:")
        print("  python artifact/scripts/show_map.py")
        return 2

    mem_extra: list[str] = []
    if args.smoke:
        mem_extra.append("--smoke")
    if args.skip_cross_domain:
        mem_extra.append("--skip-cross-domain")

    if args.scorecard_only:
        return run("run_demo.py")

    rc = run("run_memory_demo.py", mem_extra)
    if rc != 0:
        return rc

    if args.all:
        print("\nRunning full assemble scorecard (Stage 196)…")
        return run("run_demo.py")

    print("\nNext: python artifact/scripts/run_product.py --all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
