"""Check of 567: two cand, first miss → second; saved not sole gate.
    python _check567_try2.py
"""
from __future__ import annotations
from pathlib import Path
SRC = Path("_audit567_try2.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "torch" in src:
        f.append("1. Phi")
    if "n == 0 or n >= 3" not in src:
        f.append("1. refuse not 0/3+")
    if "n == 1" not in src:
        f.append("1. one-cand arm missing")
    if "h1 = hop_hit(g, by, order[1]" not in src:
        f.append("1. second try missing")
    if "miss1=1" not in src:
        f.append("1. miss1 not marked")
    if 'if held in place:' not in src or '"read_hit"' not in src:
        f.append("1. direct READ not removed before retry")
    if "c != held" in src:
        f.append("1. teacher subtracted from candidate list")
    if "void = n2 < 20" not in src:
        f.append("2. VOID on n_two")
    if "sec >= 0.99" not in src or "h2 > 0.05" not in src:
        f.append("2. GATE second/hit")
    for line in src.splitlines():
        if "gate =" in line and "saved" in line:
            f.append("2. saved sole in GATE")
            break
    return f


MUTANTS = (
    ("no second try",
     "    h1 = hop_hit(g, by, order[1], env_m, held)\n"
     "    return dict(kind=\"two\", n=2, hit=int(h1), second=1, miss1=1), None",
     "    return dict(kind=\"two\", n=2, hit=0, second=0, miss1=1), None",
     "1."),
    ("void soft",
     "    void = n2 < 20",
     "    void = n2 < 0",
     "2."),
    ("gate on saved only",
     "    gate = (not void) and (nm == 0 or sec >= 0.99) and h2 > 0.05",
     "    gate = (not void) and saved > 0.05",
     "2."),
)


def main() -> int:
    src = SRC.read_text(encoding="utf-8")
    fails = props()
    for name, old, new, tag in MUTANTS:
        n = src.count(old)
        if n != 1:
            fails.append(f"MUTATION {tag} ({name}): anchor {n}")
            continue
        got = props(src=src.replace(old, new, 1))
        if not any(g.startswith(tag) for g in got):
            fails.append(f"MUTATION {tag} ({name}): not caught")
    for x in fails:
        print("FAIL " + x)
    print(f"{len(fails)} failures" if fails else
          f"all properties hold, and all {len(MUTANTS)} re-introduced failures were caught")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
