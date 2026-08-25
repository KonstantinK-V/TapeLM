"""Check 641: PMI-fail slice ceiling; reachability gate; no Phi."""
from __future__ import annotations

import ast
from pathlib import Path

AUDIT = Path("_audit641_pmifail.py")


def source_of(src, name):
    node = next(
        item for item in ast.parse(src).body
        if isinstance(item, (ast.FunctionDef, ast.ClassDef))
        and item.name == name
    )
    return ast.get_source_segment(src, node) or ""


def gate_expression(src):
    for node in ast.walk(ast.parse(src)):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "gate"
        ):
            return ast.get_source_segment(src, node.value) or ""
    return ""


def props(src=None):
    text = AUDIT.read_text(encoding="utf-8") if src is None else src
    fails = []
    if "import torch" in text or "torch.nn" in text:
        fails.append("1. Phi/torch leaked")
    if "from _audit640_swapceil import" not in text:
        fails.append("1. 640/638 offer helpers missing")
    if "contextual_offer" not in text:
        fails.append("1. exact leftover offer missing")
    collect = source_of(text, "collect_slice")
    if "unique_max(scores)" not in collect:
        fails.append("1. PMI unique-max slice filter missing")
    if "cands[pick][\"tok\"] == held_ask" not in collect:
        fails.append("1. PMI-hit trials are not excluded")
    if "continue" not in collect:
        fails.append("1. PMI hits may remain in the slice")
    # Held may label the slice outcome, never score hand rules.
    for name in ("count_scores", "row_q_scores", "share_via_scores"):
        body = source_of(text, name) if name == "count_scores" else ""
        if name == "count_scores":
            if "held" in body:
                fails.append("1. count rule sees held")
    if "hide_two(" not in text or "n_fr - 2" not in text:
        fails.append("1. honest two-row hide missing")
    if "pins = sorted(mid_set)" not in text:
        fails.append("1. seeded run depends on set order")
    if 'rates["n"] < 80' not in text:
        fails.append("1. VOID thin slice missing")
    gate = gate_expression(text)
    if not gate:
        fails.append("1. gate missing")
    else:
        if "room > BAR" not in gate:
            fails.append("1. oracle-random residual bar missing")
        if any(item in gate for item in ("d_share", "d_count", "d_row", "share")):
            fails.append("1. hand-rule diagnostic leaked into gate")
    if "fail_kind" not in text or "refuse" not in text or "wrong" not in text:
        fails.append("1. refuse/wrong sub-slice missing")
    if "Embedding" in text or "vocab_size" in text:
        fails.append("1. vocabulary leaked")
    return fails


def mutants():
    src = AUDIT.read_text(encoding="utf-8")
    cases = [
        src.replace("import json", "import json\nimport torch"),
        src.replace(
            "if pick is not None and cands[pick][\"tok\"] == held_ask:\n"
            "                    n_pmi_hit += 1\n"
            "                    continue",
            "if False:\n"
            "                    n_pmi_hit += 1\n"
            "                    continue",
        ),
        src.replace("rates[\"n\"] < 80", "rates[\"n\"] < 0"),
        src.replace(
            "gate = (not void) and room > BAR",
            "gate = (not void) and hand['share'] > BAR",
        ),
        src.replace("pins = sorted(mid_set)", "pins = list(mid_set)"),
        src.replace("n_use = max(n_fr - 2, 1)", "n_use = max(n_fr, 1)"),
        src.replace(
            "return [pg[\"places\"][cand[\"pi\"]][\"count_key\"] for cand in cands]",
            "return [1.0 if cand[\"tok\"] == held else 0.0 for cand in cands]",
        ),
    ]
    caught = sum(bool(props(mutant)) for mutant in cases)
    return caught, len(cases)


def main() -> int:
    fails = props()
    caught, total = mutants()
    if fails:
        print("FAIL")
        for item in fails:
            print(" ", item)
        return 1
    if caught != total:
        print(f"FAIL mutants {caught}/{total}")
        return 1
    print(
        f"all properties hold, and all {total} "
        "re-introduced failures were caught"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
