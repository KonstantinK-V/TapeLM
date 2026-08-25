"""Check 576: U_any vs U_best vs maj, held only in identity of extra, no torch."""
from __future__ import annotations

from pathlib import Path

SRC = Path("_audit576_path.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    fails = []
    if "torch" in src:
        fails.append("1. torch leaked")
    if "shuffle(blocks)" in src:
        fails.append("1. windows shuffled")
    if "n < 40" not in src or "cover < 0.15" not in src:
        fails.append("1. VOID missing")
    if "d_best > 0.05 and d_maj > 0.05" not in src:
        fails.append("1. GATE not U_any-best and U_any-maj")
    if "extra == [held]" not in src:
        fails.append("2. U_any not unique-extra identity")
    if "if t == s_q" not in src:
        fails.append("2. query slot not excluded")
    body = src.split("def frame_paths")[1].split("def one_episode")[0]
    jac_line = [ln for ln in body.splitlines() if "jac =" in ln]
    if jac_line and "held" in jac_line[0]:
        fails.append("2. jac used held")
    if "tok not in env_m" not in src:
        fails.append("2. extra not dest\\env")
    return fails


MUTANTS = (
    (
        "any extra counts",
        "        if extra == [held]:",
        "        if held in extra:",
        "2.",
    ),
    (
        "query slot in",
        "        if t == s_q:\n            continue",
        "        if False and t == s_q:\n            continue",
        "2.",
    ),
    (
        "drop maj from gate",
        "    gate = (not void) and d_best > 0.05 and d_maj > 0.05",
        "    gate = (not void) and d_best > 0.05",
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
