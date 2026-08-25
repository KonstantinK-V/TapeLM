"""Check 578: rank unique-extra frames, no token in feat, beat 577 not only maj."""
from __future__ import annotations

from pathlib import Path

SRC = Path("_audit578_frrank.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    fails = []
    if "softplus" not in src:
        fails.append("1. rank loss missing")
    if "windows[:cut]" not in src:
        fails.append("1. prefix train/test missing")
    if "shuffle_y" not in src:
        fails.append("1. null arm missing")
    if "d_d > 0.05 and d_maj > 0.05" not in src:
        fails.append("1. GATE not B-D and B-maj")
    if "len(extra) != 1" not in src:
        fails.append("2. unique-extra offer missing")
    if "if len(extra) != 1:\n            continue" not in src:
        fails.append("2. unique-extra offer broken")
    if "if t == s_q" not in src:
        fails.append("2. query slot not excluded")
    feat = src.split("def feat_of")[1].split("class Ranker")[0]
    if "tok" in feat or "held" in feat:
        fails.append("2. token/held leaked into feat_of")
    if "tok not in env_m" not in src:
        fails.append("2. extra not dest\\env")
    return fails


MUTANTS = (
    (
        "all frames",
        "        if len(extra) != 1:\n            continue",
        "        if False and len(extra) != 1:\n            continue",
        "2.",
    ),
    (
        "held into feat",
        "def feat_of(jac, ov, width, df, mentions_n):\n    return [",
        "def feat_of(jac, ov, width, df, mentions_n):\n    held = 0\n    return [",
        "2.",
    ),
    (
        "gate vs maj only",
        "    gate = (not void) and d_d > 0.05 and d_maj > 0.05",
        "    gate = (not void) and d_maj > 0.05",
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
