"""Check 587: hop1 only; miss refuse; teacher extra==held; no torch."""
from __future__ import annotations

from pathlib import Path

SRC = Path("_audit587_train.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    fails = []
    if "torch" in src:
        fails.append("1. torch leaked")
    if "shuffle(blocks)" in src:
        fails.append("1. windows shuffled")
    if "n_eval < 40" not in src and 'ev["n"] < 40' not in src:
        fails.append("1. VOID n missing")
    if "d_pmi > 0.05" not in src or "d_rnd > 0.05" not in src:
        fails.append("1. GATE not Q−PMI and Q−rnd")
    if "gate = (not void) and d_pmi > 0.05 and d_rnd > 0.05" not in src:
        fails.append("1. GATE line not Q-PMI and Q-rnd")
    if "tok == held" not in src:
        fails.append("1. teacher is not extra==held")
    if "adjust_frame_stats(co, df, qtoks, -1)" not in src:
        fails.append("2. query frame not subtracted from co+df")
    if "len(extra) != 1" not in src:
        fails.append("2. unique-extra filter missing")
    pmi = src.split("def mean_pmi")[1].split("def unique_extras")[0]
    if "held" in pmi:
        fails.append("2. pmi saw held")
    feat = src.split("def feat")[1].split("class QTab")[0]
    if "held" in feat:
        fails.append("2. feat saw held")
    if "windows[:cut]" not in src:
        fails.append("1. no train/eval split")
    return fails


MUTANTS = (
    (
        "query in co",
        "        adjust_frame_stats(co, df, qtoks, -1)",
        "        adjust_frame_stats(co, df, qtoks, 0)",
        "2.",
    ),
    (
        "teacher pmi",
        "            tab.touch(key, 1.0 if tok == held else 0.0)",
        "            tab.touch(key, 1.0 if pmi > 0 else 0.0)",
        "1.",
    ),
    (
        "no split",
        "    train_w, eval_w = windows[:cut], windows[cut:]",
        "    train_w, eval_w = windows, windows",
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
