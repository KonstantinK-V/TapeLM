"""Check of 408: the size rival uses the same value as 407. No torch, no wiki."""
from __future__ import annotations

from pathlib import Path

import _audit390_address as A
import _audit408_size as S

SRC = Path("_audit408_size.py")

# unique pads per line - shared pad0 would glue unrelated places
LINES = [
    "aa the cat sat bb",
    "cc the zebra sat dd the fox saw ee2",
    "ee the zebra ran ff",
    "gg the moose ran hh",
    "ii the dog sat jj",
    "kk one pig sat ll",
    "mm one cow sat nn",
    "oo xx cat yy pp",
    "qq xx bird yy rr",
    "s1 one elk ran s2",
    "s3 one ibex ran s4",
    "t1 the wolf saw t2",
]


def designed():
    return A.build_tape(LINES, frame_max=1, min_fillers=1)


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "qprof = Counter(toks[x] for x in slots if x != s)" not in src:
        f.append("1. value does not use 407's leak discipline")
    if "ok / len(slots)" not in src:
        f.append("3. value is not a share of the place's own holes")
    if "def oracle_of(rows, key, B):" not in src:
        f.append("2. oracle_of is missing")
    if 'ranked = sorted(rows, key=lambda r: (-r[key], r["pid"]))' not in src:
        f.append("2. rivals are not top-B by the declared key")
    if "capture = v_gain - sz_gain" not in src:
        f.append("4. capture is not value_gain - size_gain")
    if "size_takes = (sz_gain > 0.05) and (capture < 0.05)" not in src:
        f.append("4. size_takes gate is not declared")
    if "def const_value(" not in src:
        f.append("5. const hubs are not measured")
    if "if all(toks[s] in {toks[x] for x in slots if x != s} for s in slots):" not in src:
        f.append("5. const hubs are not filtered to own-only places")
    pads = {L.split()[0] for L in LINES} | {L.split()[-1] for L in LINES}
    if len(pads) < 2 * len(LINES) - 2:
        f.append("0. line pads are not unique - shared pads glue places")
    return f


MUTANTS = (
    ("value counts holes not their share",
     "        rows.append({\"pid\": pid, \"value\": ok / len(slots), \"size\": len(slots),",
     "        rows.append({\"pid\": pid, \"value\": float(ok), \"size\": len(slots),", "3."),
    ("size oracle ranks by value instead of size",
     '    ranked = sorted(rows, key=lambda r: (-r[key], r["pid"]))',
     '    ranked = sorted(rows, key=lambda r: (-r["value"], r["pid"]))', "2."),
    ("capture subtracts the wrong way",
     "    capture = v_gain - sz_gain", "    capture = sz_gain - v_gain", "4."),
    ("const hubs are not filtered to own-only places",
     "        if all(toks[s] in {toks[x] for x in slots if x != s} for s in slots):",
     "        if len(slots) >= 0:", "5."),
)


def main() -> int:
    src = SRC.read_text(encoding="utf-8")
    fails = props()
    for name, old, new, tag in MUTANTS:
        if src.count(old) != 1:
            fails.append(f"MUTATION {tag} ({name}): anchor count {src.count(old)}")
            continue
        if not any(g.startswith(tag) for g in props(src.replace(old, new, 1))):
            fails.append(f"MUTATION {tag} ({name}): check did not fire")
    for x in fails:
        print("FAIL " + x)
    print(f"{len(fails)} failures" if fails else
          f"all properties hold, and all {len(MUTANTS)} re-introduced failures were caught")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
