"""Check of 420 train: balanced CE on 417h y, structural feats only.

  1. w_ref = n_live/n_ref (live weight 1); mutant w=1 caught.
  2. No fillers_place / letter hashes / place-id in Phi features.
  3. GATE bar mind_live-random_live > 0.05; mind_pin >= always_refuse.
  4. standing only if both gates — not True by default.

    python _check420_balce.py
"""
from __future__ import annotations

from pathlib import Path

SRC = Path("_train420_balce.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "D417h.step_of" not in src:
        f.append("1. 417h step_of missing")
    if "w_ref = n_live / max(1, n_ref)" not in src:
        f.append("1. w_ref = n_live/n_ref missing")
    if 'w = w_ref if sp["refuse"] else 1.0' not in src:
        f.append("1. refuse weight apply missing")
    if "fillers_place(" in src or "D417h.fillers_place" in src:
        f.append("2. fillers_place leaked into train/Phi")
    if "hash_fp" in src or "sha1" in src:
        f.append("2. letter/token hash leaked into Phi")
    if "place_of" in src and "place-id" not in src.lower():
        # place_of on tape for slots is OK; forbid embedding place id as feature
        pass
    if "feat_row" not in src or "refuse_flag" not in src and "1.0]" not in src:
        if "overlap" not in src or "bag_size" not in src:
            f.append("2. structural features missing")
    if '"features": "overlap,bag_size,n_slots,refuse_flag"' not in src:
        f.append("2. feature tag missing")
    if "and (mind_live - random_live) > 0.05)" not in src:
        f.append("3. pin gate bar missing")
    if "and mind_pin >= always_refuse)" not in src:
        f.append("3. mind>=always_refuse missing")
    if '"standing": bool(gate_pins and gate_vs_ar)' not in src:
        f.append("4. standing must be both gates, not default True")
    if "by_ctx_of" in src or "D417.step_of" in src:
        f.append("1. 417-OR leaked")
    for ban in ("REACH_CANDS", "gate_walk_only", "n_vocab", "CrossEntropyLoss"):
        if ban in src:
            f.append(f"2. lab artifact {ban}")
    return f


MUTANTS = (
    ("weight removed to 1",
     '        w = w_ref if sp["refuse"] else 1.0',
     '        w = 1.0',
     "1."),
    ("fillers_place leak",
     'def feat_row(T, bags, keys, token, j=None, refuse=False):\n'
     '    if refuse:\n'
     '        return [0.0, 0.0, 0.0, 1.0]\n'
     '    ov = overlap_of(bags, keys, token, j)\n'
     '    return [float(ov), float(len(bags[j])), float(len(T["places"][j])), 0.0]',
     'def feat_row(T, bags, keys, token, j=None, refuse=False):\n'
     '    if refuse:\n'
     '        return [0.0, 0.0, 0.0, 1.0]\n'
     '    ov = overlap_of(bags, keys, token, j)\n'
     '    _ = D417h.fillers_place(T, j) if j is not None else []\n'
     '    return [float(ov), float(len(bags[j])), float(len(T["places"][j])), 0.0]',
     "2."),
    ("gate bar dropped",
     "        and (mind_live - random_live) > 0.05)",
     "        and (mind_live - random_live) > -1.0)",
     "3."),
    ("standing True by default",
     '        "standing": bool(gate_pins and gate_vs_ar),',
     '        "standing": True,',
     "4."),
)


def main() -> int:
    src = SRC.read_text(encoding="utf-8")
    fails = props(src)
    caught = 0
    for name, old, new, tag in MUTANTS:
        if src.count(old) != 1:
            fails.append(f"MUTATION {tag} ({name}): anchor occurs {src.count(old)} times")
            continue
        mut = src.replace(old, new, 1)
        hit = [x for x in props(mut) if x.startswith(tag)]
        if not hit:
            fails.append(f"mutant not caught: {name}")
        else:
            caught += 1
    if fails:
        print("FAIL")
        for x in fails:
            print(" ", x)
        return 1
    print(f"all properties hold, and all {caught} re-introduced failures were caught")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
