"""Check 630: frozen 618; learned REFUSE branch has only three operations."""
from __future__ import annotations

import ast
from pathlib import Path

import torch

import _audit630_integrated as audit
import _integrated_contract_v1 as contract

AUDIT = Path("_audit630_integrated.py")
CONTRACT = Path("_integrated_contract_v1.py")
LAW = Path("_INTEGRATED_CONTRACT_V1.txt")


def function_has_held(src, name):
    tree = ast.parse(src)
    fn = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    names = [
        node.id for node in ast.walk(fn) if isinstance(node, ast.Name)
    ]
    names.extend(arg.arg for arg in fn.args.args)
    return any("held" in item.lower() for item in names)


def props(audit_src=None, contract_src=None, law_src=None):
    a = AUDIT.read_text(encoding="utf-8") if audit_src is None else audit_src
    c = (
        CONTRACT.read_text(encoding="utf-8")
        if contract_src is None else contract_src
    )
    law = LAW.read_text(encoding="utf-8") if law_src is None else law_src
    fails = []
    exact = 'ACTIONS = ("SEARCH_ONE", "COMMIT_RESOLVED", "REFUSE")'
    if exact not in c:
        fails.append("1. action alphabet is not exactly SEARCH/COMMIT/REFUSE")
    action_line = next(
        (line for line in c.splitlines() if line.startswith("ACTIONS = ")),
        "",
    )
    if "PEAK_STEP" in action_line:
        fails.append("1. frozen peak leaked into learned actions")
    if "torch.nn.Linear(24, len(ACTIONS))" not in a:
        fails.append("1. model output is not fixed operation alphabet")
    if "if hat is not None:" not in a or 'diag["peak_trials"] += 1' not in a:
        fails.append("1. frozen 618 branch missing")
    peak_block = a[a.index("                    if hat is not None:"):
                   a.index('                    diag["refuse_branch"] += 1')]
    if "continue" not in peak_block or "episodes.append" in peak_block:
        fails.append("1. peak branch entered training")
    for name in (
        "leftover_records", "open_record", "commit_resolved",
        "state_features", "valid_actions",
    ):
        if function_has_held(c, name):
            fails.append(f"1. held leaked into {name}")
    if "n_use = max(n_fr - 2, 1)" not in a or "hide_two(" not in a:
        fails.append("1. honest two-row hide missing")
    if "train_pool, test_pool = all_lines[:cut], all_lines[cut:]" not in a:
        fails.append("1. split-before-tapes missing")
    if "pins = sorted(mid_set)" not in a:
        fails.append("1. seeded tapes depend on set order")
    if "targets = full_feedback_q(commits, held_ask)" not in a:
        fails.append("1. all-operation teacher Q missing")
    if (
        "and rates[\"learned_net\"] > 0.0" not in a
        or "and rates[\"low95_refuse\"] > 0.0" not in a
        or "kill_switch = not enabled" not in a
    ):
        fails.append("1. always-REFUSE kill switch missing")
    if "majority" not in a or '"freq_net"' not in a:
        fails.append("1. same-place frequency rival missing")
    if "Candidate" in action_line or "door" in action_line.lower():
        fails.append("1. model names a candidate/door")
    if "It never emits a word, candidate index, door number" not in law:
        fails.append("1. non-vocab/non-door contract missing")
    if "618 -> REFUSE" not in law:
        fails.append("1. kill-switch fallback contract missing")
    return fails


def obs(tok, pi):
    return dict(
        tok=tok,
        hop_pi=pi,
        door_support_pi=100 + pi,
        majority=f"majority-{pi}",
        count_key=(0.5, -2, -2),
    )


def behavior():
    fails = []
    one = [{"door_support_pi": 9, "observations": [obs("a", 1)]}]
    got = contract.commit_resolved(one)
    if got is None or got["tok"] != "a":
        fails.append("2. one distinct result did not commit")
    tie = [{
        "door_support_pi": 9,
        "observations": [obs("a", 1), obs("b", 2)],
    }]
    if contract.commit_resolved(tie) is not None:
        fails.append("2. commit guessed on tie")
    peak = [{
        "door_support_pi": 9,
        "observations": [obs("a", 1), obs("a", 2), obs("b", 3)],
    }]
    got = contract.commit_resolved(peak)
    if got is None or got["tok"] != "a" or got["votes"] != 2:
        fails.append("2. strict peak did not resolve")
    commits = [None, None, dict(tok="answer")]
    q = contract.full_feedback_q(commits, "answer")
    if q[2] != [-1.0, 1.0, 0.0]:
        fails.append("2. final full-feedback Q wrong")
    if abs(q[1][0] - 0.95) > 1e-9 or abs(q[0][0] - 0.90) > 1e-9:
        fails.append("2. priced future SEARCH Q wrong")
    wrong = contract.full_feedback_q(
        [None, dict(tok="wrong")], "answer",
    )
    if max(wrong[0]) != 0.0:
        fails.append("2. teacher did not prefer REFUSE to priced miss")
    model = audit.RefusePolicy(8)
    if tuple(model(torch.zeros(8)).shape) != (3,):
        fails.append("2. model does not emit exactly three operations")
    return fails


MUTANTS = (
    (
        "contract",
        "learn peak",
        'ACTIONS = ("SEARCH_ONE", "COMMIT_RESOLVED", "REFUSE")',
        'ACTIONS = ("PEAK_STEP", "SEARCH_ONE", "COMMIT_RESOLVED", "REFUSE")',
        "1.",
    ),
    (
        "contract",
        "held commit",
        '    """Frozen 618-style peak over hop1 observations; held is unavailable."""',
        '    """Peak."""\n    held = None',
        "1.",
    ),
    (
        "audit",
        "train peak branch",
        '                        diag["peak_hit"] += int(hit)\n                        continue',
        '                        diag["peak_hit"] += int(hit)',
        "1.",
    ),
    (
        "audit",
        "n-1",
        "                n_use = max(n_fr - 2, 1)",
        "                n_use = max(n_fr - 1, 1)",
        "1.",
    ),
    (
        "audit",
        "disable kill switch",
        '        and rates["low95_refuse"] > 0.0',
        "        and True",
        "1.",
    ),
)


def main() -> int:
    sources = {
        "audit": AUDIT.read_text(encoding="utf-8"),
        "contract": CONTRACT.read_text(encoding="utf-8"),
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
