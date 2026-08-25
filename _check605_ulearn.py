"""Check 605: unique offer; no PMI in features; crowd refuse; Q vs first/random."""
from __future__ import annotations

from pathlib import Path

SRC = Path("_audit605_ulearn.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    fails = []
    if "torch" in src:
        fails.append("1. torch leaked")
    feat = src.split("def feat", 1)[-1].split("class QTab", 1)[0]
    if "pmi" in feat.lower():
        fails.append("1. PMI in features")
    if 'len(ep["uniq"]) < 2' not in src:
        fails.append("1. train only n_uniq>=2 missing")
    if "held in set(uniq)" not in src:
        fails.append("1. crowd split missing")
    if "no unique" not in src or "ref += 1" not in src:
        fails.append("1. crowd refuse frozen missing")
    if "fq - fr > 0.05" not in src or "fq - ff > 0.05" not in src:
        fails.append("1. Q vs rnd/first missing")
    gate_bit = src.split("gate =", 1)[-1].split("\n", 1)[0]
    if "d_pmi" in gate_bit or "fp" in gate_bit:
        fails.append("1. PMI leaked into GATE")
    if "n_u2 < 40" not in src:
        fails.append("1. VOID missing")
    if "adjust_frame_stats(co, df, qtoks, -1)" not in src:
        fails.append("1. query frame not removed from co+df")
    if "train_pool, eval_pool = all_lines[:line_cut], all_lines[line_cut:]" not in src:
        fails.append("1. corpus not split before tapes")
    return fails


MUTANTS = (
    (
        "pmi in feat",
        "    return (ob, ib, bb)",
        "    return (ob, ib, bb, 1 if df_tok else 0)\n    # pmi bucket",
        "1.",
    ),
    (
        "no refuse frozen",
        "            ref += 1",
        "            ref += 0",
        "1.",
    ),
    (
        "same corpus",
        "    train_pool, eval_pool = all_lines[:line_cut], all_lines[line_cut:]",
        "    train_pool, eval_pool = all_lines, all_lines",
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
