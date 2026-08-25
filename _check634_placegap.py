"""Check 634: D/H/P gate; 623 not in gate; same 633 read; no Φ."""
from __future__ import annotations

import ast
from pathlib import Path

AUDIT = Path("_audit634_placegap.py")


def function_has_held(src, name):
    fn = next(
        node for node in ast.parse(src).body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    names = [node.id for node in ast.walk(fn) if isinstance(node, ast.Name)]
    names.extend(arg.arg for arg in fn.args.args)
    return any("held" in item.lower() for item in names)


def gate_expression(src):
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "gate"
        ):
            return ast.get_source_segment(src, node.value) or ""
    return ""


def props(src=None):
    a = AUDIT.read_text(encoding="utf-8") if src is None else src
    fails = []
    if "import torch" in a or "torch.nn" in a:
        fails.append("1. Φ/torch leaked")
    if "from _audit633_gapcon import" not in a:
        fails.append("1. 633 read not reused")
    if "extracts_633" not in a or "unbundle" not in a:
        fails.append("1. 633 unbundle missing")
    if "search_623" not in a:
        fails.append("1. paired 623 DIAG missing")
    if "d_ora >= BAR_D" not in a:
        fails.append("1. DIRECT bar missing")
    if "h_ora >= BAR_H" not in a:
        fails.append("1. HOPONLY bar missing")
    if "(u_ora - peak_ora) >= BAR_PEAK" not in a:
        fails.append("1. peak-delta bar missing")
    if "n_live < 40" not in a:
        fails.append("1. VOID missing")
    if "n_only623" not in a or "n_onlyu" not in a:
        fails.append("1. only_623 / only_union missing")
    if "if hat is not None:" not in a:
        fails.append("1. 618 refuse-only missing")
    if "pins = sorted(mid_set)" not in a:
        fails.append("1. seeded run depends on set order")
    if "hide_two(" not in a or "n_fr - 2" not in a:
        fails.append("1. two-row hide missing")
    if "{pin, held_ask}" in a:
        fails.append("1. held filters leftover offer")
    if "leftover_records(" not in a or "{pin}," not in a:
        fails.append("1. leftover offer missing or not pin-only forbid")
    if "Linear(" in a or "GapPolicy" in a or "score(place)" in a:
        fails.append("1. learner leaked")
    gate_src = gate_expression(a)
    if not gate_src:
        fails.append("1. gate missing")
    else:
        if any(tok in gate_src for tok in ("KEEP_623", "kept", "s623", "BAR_U", "need")):
            fails.append("1. 623/kept leaked into the gate")
        if "d_ora >= BAR_D" not in gate_src:
            fails.append("1. DIRECT not in gate")
        if "h_ora >= BAR_H" not in gate_src:
            fails.append("1. HOPONLY not in gate")
        if "(u_ora - peak_ora) >= BAR_PEAK" not in gate_src:
            fails.append("1. peak-delta not in gate")
    # Offer builders live in 633; 634 must not invent a held-aware local unbundle.
    if "def unbundle(" in a and function_has_held(a, "unbundle"):
        fails.append("1. held leaked into local unbundle")
    return fails


def mutants():
    a = AUDIT.read_text(encoding="utf-8")
    cases = [
        a.replace(
            "and h_ora >= BAR_H\n",
            "",
        ),
        a.replace(
            "and (u_ora - peak_ora) >= BAR_PEAK",
            "",
        ),
        a.replace(
            "and d_ora >= BAR_D\n        and h_ora >= BAR_H\n        "
            "and (u_ora - peak_ora) >= BAR_PEAK",
            "and d_ora >= BAR_D\n        and h_ora >= BAR_H\n        "
            "and (u_ora - peak_ora) >= BAR_PEAK\n        and kept >= 0.9",
        ),
        a.replace(
            "and d_ora >= BAR_D\n        and h_ora >= BAR_H\n        "
            "and (u_ora - peak_ora) >= BAR_PEAK",
            "and d_ora >= BAR_D\n        and h_ora >= BAR_H\n        "
            "and (u_ora - peak_ora) >= BAR_PEAK\n        and u_ora >= 0.9 * s623",
        ),
        a.replace("n_live < 40", "n_live < 0"),
        a.replace("n_only623", "n_skip623"),
        a.replace(
            "pg, pin, qi, env_m, mid_set, high_set, {pin},",
            "pg, pin, qi, env_m, mid_set, high_set, {pin, held_ask},",
        ),
        a.replace("pins = sorted(mid_set)", "pins = list(mid_set)"),
    ]
    caught = 0
    for src in cases:
        if props(src):
            caught += 1
    return caught, len(cases)


def main() -> int:
    fails = props()
    caught, n = mutants()
    if fails:
        print("FAIL")
        for item in fails:
            print(" ", item)
        return 1
    if caught != n:
        print(f"FAIL mutants {caught}/{n}")
        return 1
    print(f"all properties hold, and all {n} re-introduced failures were caught")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
