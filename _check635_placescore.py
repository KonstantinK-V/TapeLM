"""Check 635: phi scores 634 places; held out of features; learning gate."""
from __future__ import annotations

import ast
from pathlib import Path

AUDIT = Path("_audit635_placescore.py")


def function_source(src, name):
    fn = next(
        node for node in ast.parse(src).body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    return ast.get_source_segment(src, fn) or ""


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


def function_has_held(src, name):
    fn = next(
        node for node in ast.parse(src).body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    names = [node.id for node in ast.walk(fn) if isinstance(node, ast.Name)]
    names.extend(arg.arg for arg in fn.args.args)
    return any("held" in item.lower() for item in names)


def props(src=None):
    a = AUDIT.read_text(encoding="utf-8") if src is None else src
    fails = []
    if "class PlaceScore" not in a or "cross_entropy" not in a:
        fails.append("1. no learner")
    if "opt.step()" not in a:
        fails.append("1. no training step")
    if "from _audit633_gapcon import" not in a or "unbundle" not in a:
        fails.append("1. 634/633 W missing")
    if "def feat_place(" not in a:
        fails.append("1. feat_place missing")
    feat = function_source(a, "feat_place")
    if not feat:
        fails.append("1. feat_place missing")
    else:
        if function_has_held(a, "feat_place"):
            fails.append("1. held leaked into feat_place")
        if "Embedding" in feat or "vocab" in feat.lower():
            fails.append("1. vocab/token id in features")
        if 'place["count_key"]' not in feat and "place['count_key']" not in feat:
            fails.append("1. count_key share missing (majority string is not a float)")
        if "float(place.get(\"majority\"" in feat or "float(place.get('majority'" in feat:
            fails.append("1. majority filler cast to float")
        if "float(place.get(\"count_key\"" in feat or "float(place.get('count_key'" in feat:
            fails.append("1. count_key tuple cast to float")
    if "hide_two(" not in a or "n_fr - 2" not in a:
        fails.append("1. two-row hide missing")
    if "pins = sorted(mid_set)" not in a:
        fails.append("1. seeded run depends on set order")
    if "{pin, held_ask}" in a:
        fails.append("1. held filters leftover offer")
    if "all_lines[:cut]" not in a or "all_lines[cut:]" not in a:
        fails.append("1. train/eval line split missing")
    if "pos_tr < 40" not in a or "n_ev < 40" not in a:
        fails.append("1. VOID missing")
    if "log.argmax()" not in a:
        fails.append("1. unique is not argmax phi")
    if "u_phi - u_rand" not in a or "u_phi - u_init" not in a:
        fails.append("1. learning bars missing")
    if "next-token" in a.lower() and "No next-token" not in a:
        fails.append("1. next-token CE")
    if "leftover_doors(" in a:
        fails.append("1. 623 doors in the offer")
    gate_src = gate_expression(a)
    if not gate_src:
        fails.append("1. gate missing")
    else:
        if any(tok in gate_src for tok in ("KEEP_623", "kept", "s623", "u_peak", "peak")):
            fails.append("1. 623/peak leaked into the gate")
        if "u_phi - u_rand" not in gate_src or "u_phi - u_init" not in gate_src:
            fails.append("1. learning bars not in gate")
    return fails


def mutants():
    a = AUDIT.read_text(encoding="utf-8")
    cases = [
        a.replace(
            "def feat_place(kind, tok, pi, pg, env_m, co, df, n_use):",
            "def feat_place(kind, tok, pi, pg, env_m, co, df, n_use, held_ask=None):",
        ),
        a.replace(
            "and (u_phi - u_rand) > BAR\n            and (u_phi - u_init) > BAR",
            "and (u_phi - u_rand) > BAR\n            and kept >= 0.9",
        ),
        a.replace("opt.step()", "pass"),
        a.replace("pick = int(log.argmax().item())", "pick = 0"),
        a.replace("pos_tr < 40", "pos_tr < 0"),
        a.replace(
            "maj_frac, neg_nvals, _neg_nkeys = place[\"count_key\"]",
            "maj_frac = float(place.get(\"majority\") or 0.0)\n"
            "    neg_nvals = float(place.get(\"count_key\") or 0.0)\n"
            "    _neg_nkeys = 0",
        ),
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
