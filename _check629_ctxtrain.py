"""Check 629: policy emits constraints; full feedback; disjoint tapes."""
from __future__ import annotations

from pathlib import Path

import torch

import _audit629_ctxtrain as audit
import _context_contract_v1 as contract

AUDIT = Path("_audit629_ctxtrain.py")
CONTRACT = Path("_context_contract_v1.py")
COLLECTOR = Path("_audit628_ctxswap.py")
LAW = Path("_CONTEXT_CONTRACT_V1.txt")


def function_source(src, name, next_name):
    start = src.index(f"def {name}(")
    end = src.index(f"\ndef {next_name}(", start)
    return src[start:end]


def props(audit_src=None, contract_src=None, collector_src=None, law_src=None):
    a = AUDIT.read_text(encoding="utf-8") if audit_src is None else audit_src
    c = (
        CONTRACT.read_text(encoding="utf-8")
        if contract_src is None else contract_src
    )
    e = (
        COLLECTOR.read_text(encoding="utf-8")
        if collector_src is None else collector_src
    )
    law = LAW.read_text(encoding="utf-8") if law_src is None else law_src
    fails = []
    if (
        '"QUERY_VIA",\n    "QUERY_AND_VIA",'
        not in c
    ):
        fails.append("1. fixed constraint alphabet missing")
    if 'via_sum * (1.0 + query)' not in c:
        fails.append("1. 628 positive resolver missing")
    if "torch.nn.Linear(24, len(ACTIONS))" not in a:
        fails.append("1. policy output is not constraints")
    if 'y_rows = [list(ep["rewards"]) for ep in examples]' not in a:
        fails.append("1. full-feedback targets missing")
    if "rng.shuffle(y_rows)" not in a or "null=True" not in a:
        fails.append("1. shuffled full-feedback rival missing")
    if "train_pool, test_pool = all_lines[:cut], all_lines[cut:]" not in a:
        fails.append("1. split-before-tapes missing")
    if "n_use = max(n_fr - 2, 1)" not in a or "hide_two(" not in a:
        fails.append("1. honest two-row hide missing")
    if "if current == held_ask:" not in e:
        fails.append("1. DIRECT not removed from context loss")
    if "groups[ep[\"current\"]].append(ep)" not in e:
        fails.append("1. test swap not same-CURRENT")
    if "pins = sorted(mid_set)" not in e:
        fails.append("1. seeded tapes depend on set order")
    state = function_source(c, "state_features", "full_feedback")
    resolver = function_source(c, "constraint_scores", "resolve_constraints")
    if "held" in state or "held" in resolver:
        fails.append("1. held leaked into state/resolver")
    gate = (
        "and delta > 0.05\n"
        "        and delta_swap > 0.05\n"
        "        and rates[\"changed\"] >= 0.10\n"
        "        and rates[\"low95\"] > 0.0\n"
        "        and rates[\"learned_net\"] > rates[\"strongest_net\"]"
    )
    if gate not in a:
        fails.append("1. held-out context gate incomplete")
    if '"freq"' not in a or "majority" not in a:
        fails.append("1. frequency rival missing")
    if "Candidate count may grow with the tape" not in law:
        fails.append("1. scalable constraint contract missing")
    if "vocabulary CE" not in law:
        fails.append("1. non-vocab teacher contract missing")
    return fails


def behavior():
    fails = []
    row_hit = {"tok": "answer"}
    row_miss = {"tok": "other"}
    resolved = {
        "REFUSE": None,
        "QUERY": row_hit,
        "VIA_SUM": row_miss,
        "VIA_MAX": None,
        "QUERY_VIA": row_hit,
        "QUERY_AND_VIA": row_miss,
    }
    hits, reads, rewards = contract.full_feedback(resolved, "answer")
    if hits != [0, 1, 0, 0, 1, 0]:
        fails.append("2. full-feedback hits wrong")
    if reads != [0, 1, 1, 0, 1, 1]:
        fails.append("2. full-feedback reads wrong")
    if rewards != [
        0.0, 1.0, -contract.C_READ, 0.0, 1.0, -contract.C_READ,
    ]:
        fails.append("2. priced rewards wrong")
    model = audit.ConstraintPolicy(7)
    if tuple(model(torch.zeros(7)).shape) != (len(contract.ACTIONS),):
        fails.append("2. model does not emit one score per constraint")
    return fails


MUTANTS = (
    (
        "audit",
        "vocab output",
        "torch.nn.Linear(24, len(ACTIONS))",
        "torch.nn.Linear(24, 1000)",
        "1.",
    ),
    (
        "audit",
        "chosen-only target",
        '    y_rows = [list(ep["rewards"]) for ep in examples]',
        '    y_rows = [[ep["rewards"][0]] for ep in examples]',
        "1.",
    ),
    (
        "audit",
        "same tape split",
        "    train_pool, test_pool = all_lines[:cut], all_lines[cut:]",
        "    train_pool, test_pool = all_lines, all_lines",
        "1.",
    ),
    (
        "contract",
        "held state",
        '    """Fixed-width, name-free policy input plus tape-resolved actions."""',
        '    """Fixed-width state."""\n    held = None',
        "1.",
    ),
    (
        "collector",
        "DIRECT in loss",
        "                if current == held_ask:\n                    diag[\"direct\"] += 1\n                    continue",
        "                if False:\n                    diag[\"direct\"] += 1\n                    continue",
        "1.",
    ),
)


def main() -> int:
    sources = {
        "audit": AUDIT.read_text(encoding="utf-8"),
        "contract": CONTRACT.read_text(encoding="utf-8"),
        "collector": COLLECTOR.read_text(encoding="utf-8"),
    }
    fails = props() + behavior()
    for where, name, old, new, tag in MUTANTS:
        count = sources[where].count(old)
        if count != 1:
            fails.append(f"MUTATION {tag} ({name}): anchor {count}")
            continue
        changed = sources[where].replace(old, new, 1)
        got = props(
            audit_src=changed if where == "audit" else sources["audit"],
            contract_src=changed if where == "contract" else sources["contract"],
            collector_src=(
                changed if where == "collector" else sources["collector"]
            ),
        )
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
