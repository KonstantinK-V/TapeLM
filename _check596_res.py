"""Check 596: residual learner; PMI not in key; copy-abort."""
from __future__ import annotations

from pathlib import Path

SRC = Path("_audit596_res.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    fails = []
    body = src.split("def key_of", 1)[-1].split("def pick_q", 1)[0]
    if "pmi" in body.lower() or "mean_pmi" in body:
        fails.append("1. PMI in key")
    if "not u_hit" not in src:
        fails.append("1. residual missing")
    if "n < 40 or n_keys < 3" not in src:
        fails.append("1. VOID missing")
    if "agr >= 0.8" not in src:
        fails.append("1. COPY abort missing")
    if "Unique frozen" not in src:
        fails.append("1. unique freeze missing")
    return fails


MUTANTS = (
    (
        "pmi in key",
        "    return (rep, w)",
        "    return (rep, w, mean_pmi)",
        "1.",
    ),
    (
        "no copy abort",
        "    copy = agr >= 0.8 and fl >= fb - 0.05",
        "    copy = False",
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
