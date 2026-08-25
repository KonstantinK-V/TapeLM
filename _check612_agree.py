"""Check 612: AGREE vs UNION; CONST split; held does not order."""
from __future__ import annotations

from pathlib import Path

SRC = Path("_audit612_agree.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    fails = []
    if "torch" in src:
        fails.append("1. torch leaked")
    if "unique_mode(vals)" not in src:
        fails.append("1. unique mode missing")
    if 'kind_b = "const"' not in src:
        fails.append("1. CONST bin missing")
    if "void = n < 40 or n_const < 20" not in src:
        fails.append("1. VOID CONST missing")
    if "d_c > 0.05" not in src:
        fails.append("1. GATE on CONST−MAJ missing")
    if "lottery = union - agree" not in src:
        fails.append("1. lottery DIAG missing")
    if "QTab" in src or "touch(" in src:
        fails.append("1. learner leaked")
    if "extract\"] == held), reverse" in src:
        fails.append("1. held orders the list")
    if "K = 6" in src.split("def ")[0]:
        fails.append("1. gated k is not 3")
    return fails


MUTANTS = (
    (
        "no const bin",
        '                kind_b = "const"',
        '                kind_b = "peak"',
        "1.",
    ),
    (
        "no const void",
        "    void = n < 40 or n_const < 20",
        "    void = n < 40",
        "1.",
    ),
    (
        "gate union",
        "    gate = (not void) and d_c > 0.05",
        "    gate = (not void) and lottery > 0.05",
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
