"""Print 415 rawlit blocks from decision dumps. Per seed, no pool."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def fmt(x, nd=3):
    if x is None:
        return "nan"
    return f"{x:.{nd}f}"


paths = [Path(a) for a in sys.argv[1:]]
if not paths:
    paths = sorted(Path("out").glob("_stage289_decision_415*.json"))

print("file  seed  ge2_n  mind   count  dlt    one_n  ref1   ref_ge2  beat  ref_gate")
ob = or_ = n = 0
for p in paths:
    if not p.exists():
        print(p, "MISSING")
        continue
    d = json.loads(p.read_text(encoding="utf-8"))
    h = ((d.get("reach") or {}).get("rawlit") or {}).get("held_out") or {}
    g, o = h.get("ge2") or {}, h.get("one") or {}
    beat = bool(h.get("gate_mind_beats_count"))
    rg = bool(h.get("gate_refuse_one_gt_ge2"))
    n += 1
    ob += int(beat)
    or_ += int(rg)
    mh, ch = g.get("mind_hit"), g.get("count_hit")
    dlt = (mh - ch) if mh is not None and ch is not None else float("nan")
    seed = d.get("seed", "?")
    print(
        f"{p.name}  {seed}  {g.get('n')}  {fmt(mh)}  {fmt(ch)}  {dlt:+.3f}  "
        f"{o.get('n')}  {fmt(o.get('refuse'))}  {fmt(g.get('refuse'))}  {beat}  {rg}"
    )
if n:
    print(f"GATE mind>count: {ob}/{n}   refuse_one>ge2: {or_}/{n}")
