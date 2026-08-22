"""Check of 412: dumb neighbours, pin by address, not by fillers."""
from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import random

import _audit390_address as A
import _audit412_addrpin as M

SRC = Path("_audit412_addrpin.py")


def _pad(k):
    return " " + " ".join(f"p{k}x{j}" for j in range(24))


DESIGNED = [
    "the XARWIN team won the opening match of the season" + _pad(0),
    "the XARWIN team lost the closing match of the season" + _pad(1),
    "the XARWIN club played a different sport on a field" + _pad(2),
    "the XARWIN club played another season on a field xx" + _pad(3),
    "also ZEBRA here extra words for a foreign frame now" + _pad(4),
    "also ZEBRA here extra words for a foreign frame two" + _pad(5),
]


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    T = A.build_tape(DESIGNED, 3, 1)
    if not T["places"] or "addrs" not in T:
        return ["0. designed tape has no places/addrs"]
    toks = T["toks"]
    hide = next(s for s, t in enumerate(toks)
                if t == "XARWIN" and T["owner"][s] == 1)
    qpid = T["place_of"][hide]
    cands = M.nearby(T, hide, radius=400, cap=16)
    if not cands:
        f.append("2. designed hole has no neighbours inside radius")
    else:
        ja = M.pick_addr(T, qpid, cands)
        club = [j for j in cands
                if any(toks[x] == "XARWIN" for x in T["places"][j]) and j != qpid]
        zebra = [j for j in cands
                 if any(toks[x] == "ZEBRA" for x in T["places"][j])]
        if club and zebra and ja in zebra and ja not in club:
            f.append("6. address pinned the foreign ZEBRA frame over shared-left XARWIN")
        if club and M.addr_score(T, qpid, club[0]) <= 0:
            f.append("6. shared-left club scored 0 — handles are not the address")

    args = Namespace(radius=400, cand=16, topm=8, max_q=40)
    rep = M.measure(T, args, random.Random(0))
    if rep is None:
        return ["0. designed tape produced no questions"]
    if rep["working_cells"] != 0:
        f.append("0. working cells are not 0")

    if "_w1, L1, R1" not in src and "_w1, L1, R1 = T[\"addrs\"][pid]" not in src:
        f.append("1. addr_score does not unpack w aside — it may be using the filler")
    if "T[\"addrs\"][pid]" not in src:
        f.append("1. addr_score is not reading addrs")
    if "at_value" in src.split("def nearby")[1].split("def pick_addr")[0]:
        f.append("2. nearby walks at_value — that is filler search, not a dumb step")
    if "hide=s" not in src:
        f.append("3. stay does not exclude the hole")
    if "own_of" not in src or "fillers_place(T, ja, own)" not in src:
        f.append("3. pin offer does not exclude asking-place own")
    if 'void = rep["scored"] <= 0.05' not in src:
        f.append("4. VOID is not the share of holes with a positive address score")
    if 'rep["addr_minus_random"] > 0.05 and rep["addr_minus_stay"] > 0.05' not in src:
        f.append("5. gate is not the double bar vs random AND vs stay")
    return f


MUTANTS = (
    ("address reads the filler w",
     "    _w1, L1, R1 = T[\"addrs\"][pid]\n    _w2, L2, R2 = T[\"addrs\"][j]",
     "    w1, L1, R1 = T[\"addrs\"][pid]\n    w2, L2, R2 = T[\"addrs\"][j]\n"
     "    if w1 == w2:\n        return 9.0",
     "1."),
    ("nearby is filler search",
     "    n = len(T[\"toks\"])\n    lo, hi = max(0, s - radius), min(n, s + radius + 1)",
     "    n = len(T[\"toks\"])\n    lo, hi = max(0, s - radius), min(n, s + radius + 1)\n"
     "    _ = T[\"at_value\"]",
     "2."),
    ("stay keeps the hole",
     "        stay_h += hit_offer(fillers_place(T, qpid, own, hide=s), truth, args.topm)",
     "        stay_h += hit_offer(fillers_place(T, qpid, own, hide=None), truth, args.topm)",
     "3."),
    ("VOID reads n",
     '    void = rep["scored"] <= 0.05',
     '    void = rep["n"] <= 0.05',
     "4."),
    ("gate drops stay",
     "    gate = (not void) and rep[\"addr_minus_random\"] > 0.05 and "
     "rep[\"addr_minus_stay\"] > 0.05",
     "    gate = (not void) and rep[\"addr_minus_random\"] > 0.05",
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
