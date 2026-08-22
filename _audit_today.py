"""Today's wrap: read existing stage JSONs, print the ladder. No new run.

    python _audit_today.py
"""
from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path("results")
STAGES = (
    ("477", "_stage477_cxfer.json", "C  pin->address->pin, iso names"),
    ("478", "_stage478_online.json", "D  tape + trains next episode"),
    ("480", "_stage480_planes.json", "planes  DEAD-retry; H dies"),
    ("481", "_stage481_mark.json", "WM  tape[H]=LIVE|DEAD"),
    ("482", "_stage482_nexthop.json", "WM->hop3  reads LIVE, not if h2"),
)


def load(name):
    p = RESULTS / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def three(d):
    if not d:
        return None
    seeds = [d[k] for k in ("1337", "8642", "2890") if k in d]
    if len(seeds) < 3:
        return f"{len(seeds)}/3 files"
    g = sum(1 for s in seeds if s.get("gate"))
    v = sum(1 for s in seeds if s.get("void"))
    return f"GO {g}/3" + (f"  VOID {v}" if v else "")


def main() -> int:
    print("TapeLM  21 Aug  -- wrap, no new science")
    print("-" * 56)
    n_go = 0
    missing = 0
    for num, fn, note in STAGES:
        d = load(fn)
        st = three(d)
        if st is None:
            missing += 1
            print(f"  {num}  (no {fn})")
        else:
            if st.startswith("GO 3"):
                n_go += 1
            print(f"  {num}  {st:12}  {note}")
    print("-" * 56)
    print("stands:")
    print("  walk transfers; word / H does not")
    print("  LIVE/DEAD meaning transfers; hop3 reads the mark")
    print("  not GPT, not wiki, not role-chain 483")
    print("walls: 479.2 pre-bucket, 474 post-Q no action")
    if missing:
        print(f"\n{missing} json missing.")
    print(f"anchors with 3/3: {n_go}/5  -- rest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
