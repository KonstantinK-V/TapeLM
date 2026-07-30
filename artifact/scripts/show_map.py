#!/usr/bin/env python3
"""Print curated stage verdicts from artifact/decisions/. No GPU required."""
from __future__ import annotations

import json
from pathlib import Path

ART = Path(__file__).resolve().parents[1]
DEC = ART / "decisions"

CURATED = [
    ("191", "P1 generation parity vs GPT"),
    ("192", "FP lexicon calibration"),
    ("194", "Fact memory hop1"),
    ("195", "Hop2 chains + binding"),
    ("197", "One-shot knowledge edit"),
    ("196", "TapeLM assemble (full scorecard)"),
    ("204", "Noise / OOV vs fair RAG"),
    ("205", "Unlearn / provenance"),
    ("203", "Structured external hops"),
    ("221", "W-remap adapter"),
    ("227", "Canonical slots + W@read"),
    ("228c", "Official fp decode (4-way)"),
    ("230", "Contradiction resolution"),
    ("226c", "Cross-domain gen + memory"),
    ("207", "Variant B (scope)"),
    ("210", "Internalization frontier (scope)"),
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
    print(f"\nQuickstart: {ART / 'QUICKSTART.md'}")
    print(f"Product demo: python artifact/scripts/run_product.py")
    print(f"Docs: {ART / 'OVERVIEW.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
