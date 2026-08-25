"""Check 628: same-CURRENT context swap; held-blind exact-address resolver."""
from __future__ import annotations

from pathlib import Path

import _audit628_ctxswap as audit

SRC = Path("_audit628_ctxswap.py")


def function_source(src, name, next_name):
    start = src.index(f"def {name}(")
    end = src.index(f"\ndef {next_name}(", start)
    return src[start:end]


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    fails = []
    if "torch" in src:
        fails.append("1. torch leaked")
    if src.count("n_fr - 2") < 2 or "hide_two" not in src:
        fails.append("1. honest two-row hide missing")
    if 'groups[ep["current"]].append(ep)' not in src:
        fails.append("1. swap is not same-CURRENT")
    if 'other["support"] != ep["support"]' not in src:
        fails.append("1. swap history is not different")
    if 'ep["qi"] not in other["support"]' not in src:
        fails.append("1. donor W may contain target query")
    if "high[:1]" not in src:
        fails.append("1. high-df allow=1 missing")
    if "pins = sorted(mid_set)" not in src:
        fails.append("1. seeded run depends on set order")
    if "return winners[0] if len(winners) == 1 else None" not in src:
        fails.append("1. tie must REFUSE")
    gate = (
        "gate = (not void) and rates[\"changed\"] >= 0.10 "
        "and delta > 0.05"
    )
    if gate not in src:
        fails.append("1. context reward/change gate missing")
    if 'rates["swap"], rates["query"], rates["peak"], rates["count"]' not in src:
        fails.append("1. strong rivals missing")
    resolver = function_source(src, "resolve_context", "resolve_query")
    kernel = function_source(src, "address_kernel", "unique_best")
    if "held" in resolver:
        fails.append("1. held leaked into context resolver")
    if "held" in kernel:
        fails.append("1. held leaked into address geometry")
    return fails


def behavior():
    fails = []
    row1 = {"pi": 1}
    row2 = {"pi": 2}
    if audit.unique_best([(1.0, row1), (1.0, row2)]) is not None:
        fails.append("2. tie guessed")
    if audit.unique_best([(0.0, row1), (0.0, row2)]) is not None:
        fails.append("2. zero evidence guessed")
    if audit.unique_best([(2.0, row1), (1.0, row2)]) is not row1:
        fails.append("2. unique address not selected")
    place = {"addr": (2, ("low", "and"), ("the", "other"))}
    toks = audit.address_tokens(place, "other", {"and", "the"})
    if toks != ("low", "and"):
        fails.append("2. high cap is not one")
    return fails


MUTANTS = (
    (
        "cross-current swap",
        'groups[ep["current"]].append(ep)',
        'groups["all"].append(ep)',
        "1.",
    ),
    (
        "same history allowed",
        '            and other["support"] != ep["support"]\n',
        "",
        "1.",
    ),
    (
        "guess tie",
        "    return winners[0] if len(winners) == 1 else None",
        "    return winners[0]",
        "1.",
    ),
    (
        "held in resolver",
        "    if not via_pis:\n        return None",
        "    held = None\n    if not via_pis:\n        return None",
        "1.",
    ),
    (
        "n-1",
        "            n_use = max(n_fr - 2, 1)",
        "            n_use = max(n_fr - 1, 1)",
        "1.",
    ),
)


def main() -> int:
    src = SRC.read_text(encoding="utf-8")
    fails = props() + behavior()
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
        f"all properties hold, behavior holds, and all "
        f"{len(MUTANTS)} re-introduced failures were caught"
    )
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
