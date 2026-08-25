"""Check of 550: cluster mentions by env, not mixed 511 rec.

    python _check550_bundle.py
"""
from __future__ import annotations

from pathlib import Path

SRC = Path("_audit550_bundle.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "unique_next" in src or "narrow_next" in src:
        f.append("1. unique hunt exam")
    if "torch" in src:
        f.append("1. Φ in ceiling")
    if "env_m" not in src or "e & env_m" not in src:
        f.append("1. cluster must share env token")
    if "held, env = frame[0], set(frame[1:])" not in src:
        f.append("1. held is one companion; env is the rest")
    if "rec_from(g, by, v, rest" not in src:
        f.append("1. GL mixed star missing")
    if "rec_bundle(g, v, mates" not in src:
        f.append("1. CL bundle rec missing")
    if "d_gl > 0.05" not in src:
        f.append("2. GATE CL−GL")
    if "split < 0.20" not in src:
        f.append("2. VOID no-split")
    if "void_n" not in src:
        f.append("2. VOID must be corpus-scaled")
    if "oracle_reward" not in src or "p_extra" not in src:
        f.append("1. reward diagnostic missing")
    if "hit_cl − hit_GL" in src:
        f.append("2. reward must not be the gate")
    return f


MUTANTS = (
    ("no split, all rest",
     "        if e & env_m:\n            mates.append(t)",
     "        mates.append(t)",
     "1."),
    ("held leaked into env",
     "    held, env = frame[0], set(frame[1:])",
     "    held, env = frame[0], set(frame)",
     "1."),
    ("gate on reward",
     "    gate = (not void) and d_gl > 0.05",
     "    gate = (not void) and (stoph - allgo) > 0.05",
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
