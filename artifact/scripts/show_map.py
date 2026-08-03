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
    ("231", "Temporal W (matched qmap)"),
    ("232", "Stream decay + W version"),
    ("233", "Tool binding keys"),
    ("234", "Compositional W (228 algebra)"),
    ("235", "Mixed L1 pretrain probe"),
    ("236", "Compositional W fixed exam"),
    ("237", "Mixed L1 overnight continue"),
    ("238", "Mixed L1 from scratch"),
    ("239", "CF sequential A->B vs GPT"),
    ("240", "CF vs GPT+RAG surprise"),
    ("241", "Harmful W vs no-W"),
    ("242", "GPT rehearsal dose"),
    ("243", "Carrier drift slots vs weights"),
    ("244", "Forget cleanliness"),
    ("245", "Mixed no-W vs P1+W"),
    ("246", "Domain curriculum retention"),
    ("247", "Ingest forks CE vs slots"),
    ("251", "CPC understand ladder (CAL+CPC)"),
    ("252", "Joint CE + lambda*CPC arms"),
    ("253", "Scale joint 16M tokens"),
    ("254", "Continual curriculum shared upper"),
    ("255", "Stream ingest engine (chunked)"),
    ("256", "Slot-bias glue (copy mixture decode)"),
    ("207", "Variant B (scope)"),
    ("210", "Internalization frontier (scope)"),
]


def main() -> int:
    print("Inprint v0.1 — curated verdicts (TapeLM codebase)\n")
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
    print(f"Product demo: python artifact/scripts/run_inprint.py demo")
    print(f"Docs: {ART / 'OVERVIEW.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
