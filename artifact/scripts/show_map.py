#!/usr/bin/env python3
"""Print curated stage verdicts from artifact/decisions/. No GPU required."""
from __future__ import annotations

import json
from pathlib import Path

ART = Path(__file__).resolve().parents[1]
DEC = ART / "decisions"

CURATED = [
    ("196", "TapeLM assemble (main demo)"),
    ("192", "FP lexicon calibration"),
    ("204", "Noise / OOV vs fair RAG"),
    ("205", "Unlearn / provenance"),
    ("203", "Internal hops (structured)"),
    ("207", "Variant B smoke"),
    ("207_max", "Variant B @ wiki scale"),
    ("208", "Hybrid rare head"),
    ("209", "Semantic scaling + teacher"),
    ("210", "Hops in forward, token answers"),
    ("211", "Internal slow tape cross-doc"),
    ("212", "Instance channel"),
]


def main() -> int:
    print("TapeLM - curated verdicts\n")
    if not DEC.is_dir():
        print(f"Missing {DEC}. Run from a full checkout.")
        return 1
    for tag, label in CURATED:
        path = DEC / f"stage{tag}_decision.json"
        if not path.exists():
            print(f"  stage{tag:8}  —  (no file)")
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        overall = d.get("overall", d.get("verdict", "?"))
        print(f"  stage{tag:8}  {overall:32}  {label}")
    print(f"\nDocs: {ART / 'OVERVIEW.md'}")
    print(f"Architecture: {ART.parent / 'docs/ARCHITECTURE.md'}")
    print(f"GitHub topics: {ART.parent / 'docs/PUBLISHING.md'}")
    print(f"Full plan: {ART.parent / 'results/plan_curve_dynamics.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
