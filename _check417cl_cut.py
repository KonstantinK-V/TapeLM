"""Check of 417cl: cut the decoy. No torch, no corpus - a designed tape.

  1. THE DECOY IS HARD BY CONSTRUCTION: it comes from a place sharing ONE half of the position's
     frame and not the other, and it is NEVER a word the tape allows at that frame. A random
     token is found by rarity, and this project has been eaten by that prior three times.
  2. THE LABEL IS THE INSERTED SLOT. Insert at i, target i, and the original word shifts right.
  3. HARD AND EASY RUN ON THE SAME LINE AND POSITION, or the hardness number compares two
     different populations instead of two decoys.
  4. THE FLOOR IS 1/positions and accuracy is 1/|argmax| under a tie - exact, seed-free.
  5. THREE DISTINCT RIVALS: frequency, frame-membership, half-locality. If they were one, "best
     count" would mean nothing.
  6. THE ENDS OF THE LINE ARE EXCLUDED - they have no frame.
  7. NO PHI ANYWHERE: this file measures whether a gap EXISTS, not who wins it.

    python _check417cl_cut.py
"""
from __future__ import annotations

import random
import re
from pathlib import Path

import _audit417cl_cut as C

SRC = Path("_audit417cl_cut.py")

# every frame must occur TWICE to be kept, so each appears on two lines with different fillers
LINES = [
    "aa the cat sat bb cc dd ee",
    "ff the dog sat gg hh ii jj",
    "kk the dog ran ll mm nn oo",
    "pp the owl ran qq rr ss tt",
    "uu one fox sat vv ww xx yy",
    "zz one elk sat a1 a2 a3 a4",
]


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    T = C.build(LINES, frame_max=1, min_fillers=1)
    words = LINES[0].split()          # aa the cat sat bb ...
    i = 2                             # `cat`, frame (the | sat)
    if C.frame_at(words, i) != (("the",), ("sat",)):
        f.append(f"6. frame_at reads {C.frame_at(words, i)}, not (the|sat)")
    legal = T["fills"].get((("the",), ("sat",)), set())
    if "cat" not in legal or "dog" not in legal:
        f.append(f"0. the designed tape is not designed: (the|sat) holds {sorted(legal)}")

    # 1: the decoy is never legal at the frame, and it is drawn from a half-sharing place
    rng = random.Random(0)
    seen = set()
    for _k in range(200):
        d = C.hard_decoy(T, words, i, rng)
        if d is not None:
            seen.add(d)
    if not seen:
        f.append("1. no hard decoy could be drawn on the designed tape")
    bad = seen & legal
    if bad:
        f.append(f"1. the hard decoy can be a word the tape ALLOWS at this frame: {sorted(bad)} - "
                 f"then the label is wrong and every rival is scored against it")
    if not (seen & {"owl", "fox", "elk"}):
        f.append(f"1. the decoy pool {sorted(seen)} does not come from half-sharing places")

    # 2 + 3: the label and the shared position
    if "w2.insert(i, dec)" not in src or "t = pos.index(i)" not in src:
        f.append("2. the target is not the inserted slot")
    seg = src[src.find("for tag, dec in"):src.find("c[\"n\"] += 1")]
    if "i = rng.randrange" in seg:
        f.append("3. the position is redrawn between the hard and the easy arm")

    # 4: floor and tie-breaking
    if "1.0 / len(pos)" not in src:
        f.append("4. the floor is not 1/positions")
    if "(1.0 / len(best)) if target_ix in best else 0.0" not in src:
        f.append("4. accuracy is not 1/|argmax| under a tie")

    # 5: three distinct rivals
    sc = re.search(r"^def scores\(.*?(?=\ndef )", src, re.S | re.M).group(0)
    for want in ('T["freq"].get(w, 0)', 'w in T["fills"].get(key, ())', 'T["by_half"].get('):
        if want not in sc:
            f.append(f"5. a rival is missing or merged: {want!r}")

    # 8: the construction must not BE a rival
    md = re.search(r"^def matched_decoy\(.*?(?=\ndef )", src, re.S | re.M)
    if md and "by_half" in re.sub(r'"""(?:.|\n)*?"""', "", md.group(0)):
        f.append("8. matched_decoy excludes words by HALF-SHARING - that is exactly what the "
                 "`local` rival counts, so the decoy would be defined as the word `local` flags")
    if md and "by_band" not in md.group(0):
        f.append("8. matched_decoy is not frequency-banded, so the `rare` prior is not removed")

    # 6: the ends
    if "range(1, len(w2) - 1)" not in src:
        f.append("6. the ends of the line are scored, and they have no frame")

    # 7: no Phi
    for bad_s in ("import torch", "net.phi", "Deriver"):
        if bad_s in src:
            f.append(f"7. {bad_s} appears - this file measures the gap, not who wins it")
    return f


MUTANTS = (
    ("the matched decoy is defined by the rival's own criterion",
     "    bad = set(T[\"fills\"].get((left, right), ())) | {true}\n    b = T[\"freq\"]",
     "    bad = set(T[\"fills\"].get((left, right), ())) | {true}\n"
     "    for _t, _h in ((\"L\", left), (\"R\", right)):\n"
     "        for _k in T[\"by_half\"].get((_t, _h), ()):\n            bad |= T[\"fills\"][_k]\n"
     "    b = T[\"freq\"]", "8."),
    ("the decoy may be a legal word for the frame",
     "    legal = T[\"fills\"].get((left, right), set()) | {words[i]}",
     "    legal = {words[i]}", "1."),
    # `key != (left, right)` is redundant while `legal` stands - a word of the same frame is
    # legal by definition - so it has no mutation of its own. It stays as the second lock.
    ("the ends of the line are scored",
     "        pos = list(range(1, len(w2) - 1))", "        pos = list(range(len(w2)))", "6."),
    ("a tie is scored as a win",
     "    return (1.0 / len(best)) if target_ix in best else 0.0",
     "    return 1.0 if target_ix in best else 0.0", "4."),
)


def main() -> int:
    src = SRC.read_text(encoding="utf-8")
    fails = props()
    for name, old, new, tag in MUTANTS:
        if src.count(old) != 1:
            fails.append(f"MUTATION {tag} ({name}): its anchor occurs {src.count(old)} times")
            continue
        saved = dict(C.__dict__)
        mutated = src.replace(old, new, 1)
        try:
            exec(compile(mutated, "<mutant>", "exec"), C.__dict__)
            got = props(src=mutated)
        except Exception as e:
            got = [f"{tag} the mutant raised {type(e).__name__}"]
        finally:
            C.__dict__.clear()
            C.__dict__.update(saved)
        if not any(g.startswith(tag) for g in got):
            fails.append(f"MUTATION {tag} ({name}): re-introduced and check {tag} did not fire")
    for x in fails:
        print("FAIL " + x)
    print(f"{len(fails)} failures" if fails else
          f"all properties hold, and all {len(MUTANTS)} re-introduced failures were caught")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
