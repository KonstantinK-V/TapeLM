"""Check 641: PMI-silent slice; 618 peak; SHARE/wrong not in gate."""
from __future__ import annotations

import ast
from pathlib import Path

AUDIT = Path("_audit641_silent.py")


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
        fails.append("1. Φ/torch leaked")
    if "from _audit640_swapceil import" not in text:
        fails.append("1. 640 leftover offer missing")
    if "contextual_offer" not in text:
        fails.append("1. 638/640 offer missing")
    if "if pick is None:\n" not in text or "if pick is None or" in text:
        fails.append("1. silent slice is not PMI unique-max refuse")
    peak = source_of(text, "peak618_tok")
    if "n1 < 2" not in peak or "n1 <= n2" not in peak:
        fails.append("1. 618 peak law missing")
    if "held" in peak:
        fails.append("1. peak618 sees held")
    gate = gate_expression(text)
    if not gate:
        fails.append("1. gate missing")
    else:
        if "s_rates[\"d_peak\"] > BAR" not in gate:
            fails.append("1. silent peak−random bar missing")
        if "d_share" in gate or "wrong" in gate:
            fails.append("1. SHARE or wrong-slice leaked into gate")
        if "w_rates" in gate:
            fails.append("1. PMI-wrong recovery leaked into gate")
    if "s_rates[\"n\"] < 80" not in text:
        fails.append("1. VOID thin silent missing")
    if "s_rates[\"room\"] <= BAR" not in text:
        fails.append("1. VOID silent room missing")
    if "wrong_in_gate=False" not in text or "share_in_gate=False" not in text:
        fails.append("1. declared diagnostics missing")
    if "hide_two(" not in text or "n_fr - 2" not in text:
        fails.append("1. honest two-row hide missing")
    if "pins = sorted(mid_set)" not in text:
        fails.append("1. seeded run depends on set order")
    if "Embedding" in text or "vocab_size" in text:
        fails.append("1. vocabulary leaked")
    if "swap_via" in text:
        fails.append("1. 640 SWAP leaked; silent does not need it")
    return fails


def mutants():
    src = AUDIT.read_text(encoding="utf-8")
    cases = [
        src.replace("import json", "import json\nimport torch"),
        src.replace(
            "if pick is None:\n            silent.append(row)",
            "if pick is None or row[\"cands\"][pick][\"tok\"] != row[\"held\"]:\n"
            "            silent.append(row)",
        ),
        src.replace(
            "gate = (not void) and s_rates[\"d_peak\"] > BAR",
            "gate = (not void) and w_rates[\"d_peak\"] > BAR",
        ),
        src.replace(
            "gate = (not void) and s_rates[\"d_peak\"] > BAR",
            "gate = (not void) and s_rates[\"d_share\"] > BAR",
        ),
        src.replace("s_rates[\"n\"] < 80", "s_rates[\"n\"] < 0"),
        src.replace("or s_rates[\"room\"] <= BAR", ""),
        src.replace("if n1 < 2 or n1 <= n2:", "if False:"),
        src.replace("wrong_in_gate=False", "wrong_in_gate=True"),
        src.replace("pins = sorted(mid_set)", "pins = list(mid_set)"),
        src.replace("n_use = max(n_fr - 2, 1)", "n_use = max(n_fr, 1)"),
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
