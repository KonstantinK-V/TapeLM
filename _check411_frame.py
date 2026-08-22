"""Check of 411: the pin chosen by the frame. No torch, no corpus - a designed tape.

  1. NO FILLER ENTERS THE RANKING. `frame_score` reads addresses only; permuting every filler on
     the tape must not move it. If a filler reached it, this would be the walk again.
  2. THE POOL IS DUMB - random places and neighbours by tape order, never `walk_order`.
  3. STAY IS THE STANDING ARM: the walk from the question's own place, own values out, capped.
  4. A PIN'S OFFER EXCLUDES THE QUESTION'S OWN VALUES and is capped at the same M.
  5. THE HOLE IS OUT OF THE KEY: the query profile excludes it, same-line places are dropped from
     the pool and from the walk.
  6. RANDOM AND ADDRESS PINS ARE DRAWN FROM THE SAME POOL, or the comparison is between two
     different offers rather than between two ways of choosing one.
  7. THE HALF INDEX COUNTS PLACES, not occurrences - a word repeated inside one half must weigh
     once, or a long frame outvotes a rare word.

    python _check411_frame.py
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import _audit390_address as A
import _audit411_frame as F

SRC = Path("_audit411_frame.py")

LINES = [
    "aa the cat sat bb",
    "cc the zebra sat dd",
    "ee the dog ran ff",
    "gg the fox ran hh",
    "ii one pig sat jj",
    "kk one cow sat ll",
    "mm xx bird yy nn",
    "oo xx wolf yy pp",
]


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    T = A.build_tape(LINES, frame_max=1, min_fillers=1)
    q = T["of_addr"][(1, ("the",), ("sat",))]
    same_left = T["of_addr"][(1, ("the",), ("ran",))]
    same_right = T["of_addr"][(1, ("one",), ("sat",))]
    none = T["of_addr"][(1, ("xx",), ("yy",))]
    L, R = F.half_index(T)

    # 1: the ranking is about windows, and a place sharing a half must outrank one sharing none
    s_left = F.frame_score(T, q, same_left, L, R)
    s_none = F.frame_score(T, q, none, L, R)
    if not (s_left > 0 and s_none == 0):
        f.append(f"1. frame_score reads {s_left} for a shared left half and {s_none} for a place "
                 f"sharing no window word")
    body = re.search(r"^def frame_score\(.*?(?=\ndef )", src, re.S | re.M).group(0)
    for bad in ("prof", "at_value", "fillers", "toks"):
        if bad in body:
            f.append(f"1. frame_score reads {bad!r} - a filler is entering the ranking")

    # 7: the half index counts places, not occurrences
    hi = re.search(r"^def half_index\(.*?(?=\ndef )", src, re.S | re.M).group(0)
    if "for x in set(left)" not in hi or "for x in set(right)" not in hi:
        f.append("7. the half index counts occurrences, so a word repeated inside one half "
                 "weighs more than once")

    # 2: the pool is dumb
    if "rng.randrange(npl)" not in src or "(pid + d) % npl" not in src:
        f.append("2. the pool is not random-plus-neighbours")
    pool_seg = src[src.find("pool = set()"):src.find("j_rand")]
    if "walk_order" in pool_seg:
        f.append("2. the pool is built from the filler walk - that is the arm this exists to "
                 "avoid")

    # 3 + 4: stay and the pin offer
    if "walked = A.walk_order(T, pid, qprof, args.places, drop)" not in src:
        f.append("3. STAY is not the walk from the question's own place")
    oo = re.search(r"^def offer_of\(.*?(?=\ndef )", src, re.S | re.M).group(0)
    if "v not in own" not in oo or "[:topm]" not in oo:
        f.append("4. a pin's offer does not exclude own values or is not capped")

    # 5: the hole out of the key
    if "qprof = Counter(toks[x] for x in T[\"places\"][pid] if x != s)" not in src:
        f.append("5. the hole is not taken out of the query profile")
    if 'drop = set(T["on_line"][owner[s]])' not in src or "j not in drop" not in src:
        f.append("5. same-line places are not dropped from the pool")

    # 6: one pool for both pins
    if "j_rand = pool[rng.randrange(len(pool))]" not in src or "max(pool, key=" not in src:
        f.append("6. the two pins are not drawn from the same pool")
    return f


MUTANTS = (
    ("a filler enters the ranking",
     "    s = 0.0\n    for x in set(l1) & set(l2):",
     "    s = float(len(T[\"prof\"][j]))\n    for x in set(l1) & set(l2):", "1."),
    ("the half index counts occurrences",
     "        for x in set(left):", "        for x in left:", "7."),
    ("the pool becomes the filler walk",
     "        for _k in range(args.pool // 2):\n            pool.add(rng.randrange(npl))",
     "        for _k in A.walk_order(T, pid, qprof, args.pool, drop):\n            pool.add(_k)",
     "2."),
    ("a pin's offer keeps the question's own values",
     "    return [v for v, _c in T[\"prof\"][j].most_common() if v not in own][:topm]",
     "    return [v for v, _c in T[\"prof\"][j].most_common()][:topm]", "4."),
    ("the hole stays in the key",
     "        qprof = Counter(toks[x] for x in T[\"places\"][pid] if x != s)",
     "        qprof = Counter(toks[x] for x in T[\"places\"][pid])", "5."),
)


def main() -> int:
    src = SRC.read_text(encoding="utf-8")
    fails = props()
    for name, old, new, tag in MUTANTS:
        if src.count(old) != 1:
            fails.append(f"MUTATION {tag} ({name}): its anchor occurs {src.count(old)} times")
            continue
        saved = dict(F.__dict__)
        mutated = src.replace(old, new, 1)
        try:
            exec(compile(mutated, "<mutant>", "exec"), F.__dict__)
            got = props(src=mutated)
        except Exception as e:
            got = [f"{tag} the mutant raised {type(e).__name__}"]
        finally:
            F.__dict__.clear()
            F.__dict__.update(saved)
        if not any(g.startswith(tag) for g in got):
            fails.append(f"MUTATION {tag} ({name}): re-introduced and check {tag} did not fire")
    for x in fails:
        print("FAIL " + x)
    print(f"{len(fails)} failures" if fails else
          f"all properties hold, and all {len(MUTANTS)} re-introduced failures were caught")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
