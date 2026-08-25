"""Check 588: hop2 only after this-hand hop1 hit; depth in Q key."""
from __future__ import annotations

from pathlib import Path

SRC = Path("_audit588_chain.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    fails = []
    if "torch" in src:
        fails.append("1. torch leaked")
    if "shuffle(blocks)" in src:
        fails.append("1. windows shuffled")
    if 'ev["n2"] < 40' not in src:
        fails.append("1. VOID n2 missing")
    if "d_pmi > 0.05" not in src or "d_rnd > 0.05" not in src:
        fails.append("1. GATE not Q-PMI and Q-rnd")
    if "if a:" not in src or "if b:" not in src:
        fails.append("1. hop2 not gated on this-hand hop1")
    feat = src.split("def feat")[1].split("class QTab")[0]
    if "return (depth, pb, ob, bb)" not in feat:
        fails.append("1. depth not in feat")
    if "tok == held1" not in src or "tok == held2" not in src:
        fails.append("1. teacher not extra==held")
    if "adjust_frame_stats(co, df, qtoks, -1)" not in src:
        fails.append("2. query frame not subtracted from co+df")
    if "len(extra) != 1" not in src:
        fails.append("2. unique-extra filter missing")
    pmi = src.split("def mean_pmi")[1].split("def unique_extras")[0]
    if "held" in pmi:
        fails.append("2. pmi saw held")
    return fails


MUTANTS = (
    (
        "hop2 without hit",
        "        if a:\n            n2a += 1\n            f2a += fill_of(h2[\"rows\"], h2[\"held\"], \"pmi\", tab, rng)",
        "        if True:\n            n2a += 1\n            f2a += fill_of(h2[\"rows\"], h2[\"held\"], \"pmi\", tab, rng)",
        "1.",
    ),
    (
        "query in co",
        "        adjust_frame_stats(co, df, qtoks, -1)",
        "        adjust_frame_stats(co, df, qtoks, 0)",
        "2.",
    ),
    (
        "no depth",
        "    return (depth, pb, ob, bb)",
        "    return (pb, ob, bb)",
        "1.",
    ),
)


def main() -> int:
    src = SRC.read_text(encoding="utf-8")
    fails = props()
    for name, old, new, tag in MUTANTS:
        count = src.count(old)
        if count != 1:
            fails.append(f"MUTATION {tag} ({name}): anchor {count}")
            continue
        got = props(src.replace(old, new, 1))
        if not any(item.startswith(tag) for item in got):
            fails.append(f"MUTATION {tag} ({name}): not caught")
    for item in fails:
        print("FAIL " + item)
    print(
        f"{len(fails)} failures" if fails else
        f"all properties hold, and all {len(MUTANTS)} re-introduced failures were caught"
    )
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
