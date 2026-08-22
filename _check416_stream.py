"""Check of 416 stream contract. No torch run — properties on the source.

  1. Key is window halves, not the token (window_parts / addrs, not toks[slot] in the key).
  2. Mind scores places (+ refuse), never a word list / offer.
  3. Tape resolve = majority filler of the chosen place.
  4. Refuse reward is +1 iff freq==1 on the tape.
  5. Loss is expected reward over place actions — not CE over vocab.
  6. Verdict has no GATE-WO.
  7. Same-line places dropped from candidates (390).

    python _check416_stream.py
"""
from __future__ import annotations

from pathlib import Path

SRC = Path("_audit416_stream.py")


def props(src: str | None = None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "window_parts" not in src or 'T["addrs"][pid]' not in src:
        f.append("1. window key is not the address halves")
    if 'for s in T["places"][pid]]' in src and "window_parts" in src[
            src.find("def window_parts"): src.find("def window_parts") + 400]:
        # fillers of the asking place in the key re-introduce the hidden token's neighbours
        # and often the token itself when the place bag is used as the key
        wp = src[src.find("def window_parts"): src.find("\ndef ", src.find("def window_parts") + 1)]
        if "places" in wp and "toks" in wp:
            f.append("1. hidden token is in the key")
    if "cand_places" not in src or "by_left" not in src:
        f.append("2. candidates are not address places")
    if "REACH_CANDS" in src or "walk_only_pick" in src:
        f.append("2. lab offer / walk_only pick leaked into stream train")
    if "most_common(1)" not in src or "def resolve" not in src:
        f.append("3. tape does not resolve by majority of the place")
    if "1.0 if freq == 1 else -1.0" not in src:
        f.append("4. refuse reward is not tied to freq==1")
    if "softmax(scores" not in src or 'sp["R"]' not in src:
        f.append("5. loss is not expected reward over place scores")
    if "CrossEntropy" in src or "n_vocab" in src:
        f.append("5. vocab CE present")
    if "gate_walk_only" in src:
        f.append("6. GATE-WO in the stream verdict")
    if "gate_mind_beats_rival" not in src or "gate_refuse_one_gt_ge2" not in src:
        f.append("6. declared gates missing")
    if "on_line" not in src:
        f.append("7. same-line drop missing")
    return f


MUTANTS = (
    ("token in the key",
     'def window_parts(T, pid) -> list[str]:\n'
     '    """The key is the address halves, never the hidden token."""\n'
     '    _w, left, right = T["addrs"][pid]\n'
     '    return [f"L:{x}" for x in left] + [f"R:{x}" for x in right]',
     'def window_parts(T, pid) -> list[str]:\n'
     '    """leaked"""\n'
     '    _w, left, right = T["addrs"][pid]\n'
     '    return [f"L:{x}" for x in left] + [f"R:{x}" for x in right] + '
     '[T["toks"][s] for s in T["places"][pid]]',
     "1."),
    ("lab offer leaked",
     '"""Scores (window_key, place) → scalar. Output alphabet = places (+ refuse), not words."""',
     '"""Scores words from REACH_CANDS offer."""\n    walk_only_pick = True',
     "2."),
    ("refuse ignores freq",
     "R.append(1.0 if freq == 1 else -1.0)",
     "R.append(1.0)",
     "4."),
)


def main() -> int:
    src = SRC.read_text(encoding="utf-8")
    bad = props(src)
    if bad:
        print("FAIL", *bad, sep="\n  ")
        return 1
    caught = 0
    for name, old, new, tag in MUTANTS:
        mut = src.replace(old, new, 1)
        if mut == src:
            print(f"mutant not applied: {name}")
            return 1
        hit = [x for x in props(mut) if x.startswith(tag)]
        if not hit:
            print(f"mutant not caught: {name}")
            return 1
        caught += 1
    print(f"all properties hold, and all {caught} re-introduced failures were caught")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
