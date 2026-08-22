"""Check of 413: window words retrieve via left/right index. No torch, no wiki.

  1. THE HOLE IS NOT A KEY. `window_keys` skips the hole token; indexing never uses filler `w`.
  2. INDEX IS LEFT/RIGHT, NOT FILLER. `by_ctx_of` walks addrs halves only.
  3. SAME-LINE DROP. Places sharing the hole's line are out of retrieve.
  4. TASTY -> APPLES. On the designed tape, the rare window word retrieves the place that
     holds the truth as filler under that context handle.
  5. DOUBLE GATE: window - stay > 0.05 AND window - random > 0.05, after VOID on nonempty.

    python _check413_winpin.py
"""
from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import random

import _audit390_address as A
import _audit413_winpin as M

SRC = Path("_audit413_winpin.py")

# hole = OTHER; window word tasty indexes a different place whose filler is APPLES
DESIGNED = [
    "p0 tasty APPLES right0 more padding words for the line length aa",
    "p1 tasty APPLES right0 more padding words for the line length bb",
    "p2 tasty OTHER right1 more padding words for the line length cc",
    "p3 tasty OTHER right1 more padding words for the line length dd",
    "p4 common word word word word word word word word word word ee",
    "p5 common word word word word word word word word word word ff",
]


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    T = A.build_tape(DESIGNED, 3, 1)
    if not T["places"] or "addrs" not in T:
        return ["0. designed tape has no places/addrs"]

    # 1 + 2: hole out of keys; index is halves
    if "if v == hole or v in seen:" not in src:
        f.append("1. window_keys does not exclude the hole")
    if "for tok in set(L) | set(R):" not in src:
        f.append("2. by_ctx_of is not left/right — filler w may be in the index")
    if "_w, L, R" not in src:
        f.append("2. addrs are not unpacked with w set aside")

    # 3: same-line drop
    if 'drop = set(T["on_line"][T["owner"][s]])' not in src:
        f.append("3. same-line places are not dropped")
    if "drop.discard(qpid)" not in src:
        f.append("3. the asking place is dropped from itself")

    # 4: tasty retrieves APPLES on designed tape (hole is OTHER, not APPLES)
    hide = next((s for s, t in enumerate(T["toks"]) if t == "OTHER"), None)
    if hide is None:
        f.append("4. designed tape has no OTHER hole")
    else:
        keys = M.window_keys(T, hide)
        if "tasty" not in keys:
            f.append("4. tasty is not among window keys of the OTHER hole")
        if "OTHER" in keys:
            f.append("1. the hole OTHER entered the window keys")
        ix = M.by_ctx_of(T)
        qpid = T["place_of"][hide]
        drop = set(T["on_line"][T["owner"][hide]])
        drop.discard(qpid)
        own = M.own_of(T, hide)
        win_p = M.retrieve(T, ix, keys, qpid, drop, 8)
        win = M.offer_of(T, win_p, own)
        if "APPLES" not in win[:8]:
            f.append("4. tasty→APPLES: window retrieve did not offer APPLES")

    # 5: gate
    if 'void = rep["nonempty"] <= 0.05' not in src:
        f.append("5. VOID is not nonempty <= 0.05")
    if ('rep["window_minus_stay"] > 0.05 and rep["window_minus_random"] > 0.05'
            not in src):
        f.append("5. gate is not the double bar vs stay AND vs random")

    args = Namespace(cap=8, topm=8, max_q=40)
    rep = M.measure(T, args, random.Random(0))
    if rep is None:
        f.append("0. designed tape produced no questions")
    elif rep["working_cells"] != 0:
        f.append("0. working cells are not 0")
    return f


MUTANTS = (
    ("the hole enters the keys",
     "        if v == hole or v in seen:",
     "        if v in seen:",
     "1."),
    ("index includes filler w",
     "        for tok in set(L) | set(R):",
     "        for tok in set(L) | set(R) | {_w}:",
     "2."),
    ("same-line places stay in",
     '        drop = set(T["on_line"][T["owner"][s]])',
     "        drop = set()",
     "3."),
    ("VOID reads n",
     '    void = rep["nonempty"] <= 0.05',
     '    void = rep["n"] <= 0.05',
     "5."),
    ("gate drops stay",
     "    gate = (not void) and rep[\"window_minus_stay\"] > 0.05 and "
     "rep[\"window_minus_random\"] > 0.05",
     "    gate = (not void) and rep[\"window_minus_random\"] > 0.05",
     "5."),
)


def main() -> int:
    src = SRC.read_text(encoding="utf-8")
    fails = props()
    for name, old, new, tag in MUTANTS:
        if src.count(old) != 1:
            fails.append(f"MUTATION {tag} ({name}): its anchor occurs {src.count(old)} times")
            continue
        saved = dict(M.__dict__)
        mutated = src.replace(old, new, 1)
        try:
            exec(compile(mutated, "<mutant>", "exec"), M.__dict__)
            got = props(src=mutated)
        except Exception as e:
            got = [f"{tag} the mutant raised {type(e).__name__}"]
        finally:
            M.__dict__.clear()
            M.__dict__.update(saved)
        if not any(g.startswith(tag) for g in got):
            fails.append(f"MUTATION {tag} ({name}): the failure was re-introduced and check "
                         f"{tag} did not fire")
    for x in fails:
        print("FAIL " + x)
    print(f"{len(fails)} failures" if fails else
          f"all properties hold, and all {len(MUTANTS)} re-introduced failures were caught")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
