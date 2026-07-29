#!/usr/bin/env python3
"""Run TapeLM scorecard (Stage 196) from repo root if checkpoints exist."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STAGE = REPO / "_stage196_tapelm.py"
P1 = REPO / "checkpoints" / "stage191_p1_curve.pt"


def main() -> int:
    if not STAGE.exists():
        print(f"Missing {STAGE}")
        return 1
    if not P1.exists():
        print("Checkpoints not found. See docs/CHECKPOINTS.md")
        print("Meanwhile, browse results without GPU:")
        print("  python artifact/scripts/show_map.py")
        return 2
    print(f"Running {STAGE.name} …")
    r = subprocess.run([sys.executable, str(STAGE)], cwd=str(REPO))
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
