"""Check of 406: today's write is throwaway; write-back is a marked cell hop 2 can stand on."""
from __future__ import annotations
from pathlib import Path
import _audit406_writeback as A

SRC = Path("_audit406_writeback.py")

def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    today, back = A.measure(False), A.measure(True)
    if today["hop2_sees_write"] != 0:
        f.append("1. TODAY hop 2 already stands on a marked cell")
    if today["truth"]["tmp_fill"] is not True or today["truth"]["working_cells"] != 0:
        f.append("1. TODAY is not a throwaway fill")
    if back["hop2_sees_write"] != 1 or back["truth"]["working_cells"] != 1:
        f.append("2. WRITE-BACK hop 2 does not stand on the marked cell")
    if not back["truth"]["corpus_untouched"] or back["truth"]["tmp_fill"]:
        f.append("2. WRITE-BACK mutated the corpus or still throwaway-fills")
    if "fill_conjecture" not in src:
        f.append("3. conjecture fill is gone")
    if not back["wrong_walks_elsewhere"] or not today["wrong_walks_elsewhere"]:
        f.append("4. OTHER and XARWIN walk the same homes")
    if 'w["marked"] = set(q["marked"]) | {("work", len(w["working"]) - 1)}' not in src:
        f.append("5. write-back does not mark the new cell")
    return f

MUTANTS = (
    ("today already appends",
     '    w["working"] = list(q["working"])\n    w["marked"] = set(q["marked"])\n    return w',
     '    w["working"] = list(q["working"]) + ["XARWIN"]\n'
     '    w["marked"] = set(q["marked"]) | {("work", 0)}\n    return w', "1."),
    ("write-back forgets to append",
     '    w["working"] = list(q["working"]) + [said]',
     '    w["working"] = list(q["working"])', "2."),
    ("the mark is dropped",
     '    w["marked"] = set(q["marked"]) | {("work", len(w["working"]) - 1)}',
     '    w["marked"] = set(q["marked"])', "5."),
    ("wrong and true share a cluster",
     '    "the OTHER club played a different sport on a different field in a long padded line",',
     '    "the XARWIN club played a different sport on a different field in a long padded line",',
     "4."),
)

def main() -> int:
    src = SRC.read_text(encoding="utf-8")
    fails = props()
    for name, old, new, tag in MUTANTS:
        if src.count(old) != 1:
            fails.append(f"MUTATION {tag} ({name}): anchor occurs {src.count(old)} times")
            continue
        saved = dict(A.__dict__)
        exec(compile(src.replace(old, new, 1), "<mutant>", "exec"), A.__dict__)
        try:
            got = props(src=src.replace(old, new, 1))
        except Exception as e:
            got = [f"{tag} {type(e).__name__}"]
        finally:
            A.__dict__.clear(); A.__dict__.update(saved)
        if not any(g.startswith(tag) for g in got):
            fails.append(f"MUTATION {tag} ({name}): check did not fire")
    for x in fails:
        print("FAIL " + x)
    print(f"{len(fails)} failures" if fails else
          f"all properties hold, and all {len(MUTANTS)} re-introduced failures were caught")
    return 1 if fails else 0

if __name__ == "__main__":
    raise SystemExit(main())
