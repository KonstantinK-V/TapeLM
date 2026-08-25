"""Check 640: torch-free TRUE/SWAP ceiling; PMI frozen; SHARE-VIA gate."""
from __future__ import annotations

import ast
from pathlib import Path

AUDIT = Path("_audit640_swapceil.py")


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
    if "contextual_offer" not in text or "leftover_records" not in text:
        fails.append("1. 638 leftover offer missing")
    swap = source_of(text, "swap_via")
    if 'cand["via_pi"]' not in swap or "shuffled" not in swap:
        fails.append("1. SWAP does not permute VIA pointers")
    if '["tok"]' in swap:
        fails.append("1. SWAP retargets extract — PMI no longer frozen")
    if 'out[index]["pi"]' in swap or "out[index]['pi']" in swap:
        fails.append("1. SWAP retargets place id — PMI no longer frozen")
    if '["via_pi"] is not None' not in swap:
        fails.append("1. SWAP may permute DIRECT (null VIA)")
    pmi = source_of(text, "pmi_scores")
    if "via_pi" in pmi or "keys" in pmi or "vals" in pmi:
        fails.append("1. PMI rival sees VIA/rows")
    unique = source_of(text, "unique_max")
    if "len(winners) == 1 else None" not in unique:
        fails.append("1. ties do not refuse")
    share = source_of(text, "share_via_scores")
    if '["keys"]' not in share or "via_pi" not in share:
        fails.append("1. SHARE-VIA is not place-key intersection")
    if "left & right" not in share:
        fails.append("1. SHARE-VIA is not key intersection")
    if "held" in share or "held_ask" in share:
        fails.append("1. SHARE-VIA sees the label")
    if "address_kernel" in text or "resolve_context" in text:
        fails.append("1. 628 kernel leaked into 640 W")
    if 'rates["n"] < 80' not in text:
        fails.append("1. VOID thin pairs missing")
    if "pmi_changed" not in text or "PMI_MOVE" not in text:
        fails.append("1. VOID PMI-moved missing")
    if 'rates["pmi_changed"] > PMI_MOVE' not in text:
        fails.append("1. VOID does not fail when PMI moves")
    if "room <= BAR" not in text:
        fails.append("1. VOID oracle-PMI room missing")
    gate = gate_expression(text)
    if not gate:
        fails.append("1. gate missing")
    else:
        if "d_share > BAR" not in gate:
            fails.append("1. SHARE-VIA TRUE−SWAP bar missing")
        if "d_row" in gate:
            fails.append("1. ROW-Q leaked into gate")
        if any(item in gate for item in ("oracle", "row_true", "rand")):
            fails.append("1. diagnostic leaked into gate")
    if "hide_two(" not in text or "n_fr - 2" not in text:
        fails.append("1. honest two-row hide missing")
    if "pins = sorted(mid_set)" not in text:
        fails.append("1. seeded run depends on set order")
    if "Embedding" in text or "vocab_size" in text:
        fails.append("1. vocabulary leaked")
    return fails


def mutants():
    src = AUDIT.read_text(encoding="utf-8")
    cases = [
        src.replace("import math", "import math\nimport torch"),
        src.replace(
            "out[index][\"via_pi\"] = via",
            "out[index][\"tok\"] = via",
        ),
        src.replace(
            "out[index][\"via_pi\"] = via",
            "out[index][\"pi\"] = via",
        ),
        src.replace(
            "return winners[0] if len(winners) == 1 else None",
            "return winners[0]",
        ),
        src.replace(
            "gate = (not void) and d_share > BAR",
            "gate = (not void) and d_row > BAR",
        ),
        src.replace("rates[\"n\"] < 80", "rates[\"n\"] < 0"),
        src.replace("rates[\"pmi_changed\"] > PMI_MOVE", "False"),
        src.replace("or room <= BAR", ""),
        src.replace("len(left & right)", "len(left)"),
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
