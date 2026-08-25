"""Check of 551: env reranks 511 rec, does not filter.

    python _check551_rerank.py
"""
from __future__ import annotations

from pathlib import Path

SRC = Path("_audit551_rerank.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "torch" in src:
        f.append("1. Φ in ceiling")
    if "unique_next" in src:
        f.append("1. unique hunt")
    if "sorted(rec_gl, key=lambda c: (-cnt_m[c], -cnt_g[c]))" not in src:
        f.append("1. rerank by mate counts")
    if "rec_from(g, by, v, rest" not in src:
        f.append("1. GL universe missing")
    if "take_fl" not in src:
        f.append("1. 550 filter control missing")
    if "d_gl > 0.05" not in src:
        f.append("2. GATE RN-GL")
    if "held, env = frame[0], set(frame[1:])" not in src:
        f.append("1. held vs env")
    return f


MUTANTS = (
    ("filter leaked back",
     "    rec_rn = sorted(rec_gl, key=lambda c: (-cnt_m[c], -cnt_g[c]))",
     "    rec_rn = list(rec_fl) if rec_fl else rec_gl",
     "1."),
    ("held in env",
     "    held, env = frame[0], set(frame[1:])",
     "    held, env = frame[0], set(frame)",
     "1."),
    ("gate on filter",
     "    gate = (not void) and d_gl > 0.05",
     "    gate = (not void) and (hit_rn - hit_fl) > 0.05",
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
