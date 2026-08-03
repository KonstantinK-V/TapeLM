#!/usr/bin/env python3
"""Refresh artifact/decisions/ from results/stage*_decision.json (curated list)."""
from __future__ import annotations

import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEC = REPO / "artifact" / "decisions"
TAGS = ["196", "192", "204", "205", "203", "207", "207_max", "208", "209", "210", "211", "212", "213", "214", "215", "216", "217", "218", "219", "220", "221", "222", "223", "224", "225", "226", "226b", "226c", "227", "229", "230", "228a", "228b", "228c", "221_probe", "231", "232", "233", "234", "235", "236", "237", "238", "239", "240", "241", "242", "243", "244", "245", "246", "247", "248", "249", "250", "251", "252", "253", "254", "255", "256"]

def main() -> int:
    DEC.mkdir(parents=True, exist_ok=True)
    n = 0
    for t in TAGS:
        for suffix in ("_decision.json", "_mini.md"):
            src = REPO / "results" / f"stage{t}{suffix}"
            if src.exists():
                shutil.copy2(src, DEC / src.name)
                n += 1
    print(f"Synced {n} files -> {DEC}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
