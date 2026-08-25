"""Check 636: pos-only rank; no REFUSE class; same 635/634 feats and W."""
from __future__ import annotations

import ast
from pathlib import Path

AUDIT = Path("_audit636_posrank.py")


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
    if "class RankNet" not in a or "cross_entropy" not in a:
        fails.append("1. no learner")
    if "opt.step()" not in a:
        fails.append("1. no training step")
    if "self.refuse" in a:
        fails.append("1. REFUSE parameter")
    if "from _audit635_placescore import" not in a or "feat_place" not in a:
        fails.append("1. 635 feat_place not reused")
    if "_num(" in a or 'float(place.get("majority"' in a:
        fails.append("1. broken majority/count_key cast")
    if 'tr["y"] is not None' not in a or "pos = [" not in a:
        fails.append("1. train not filtered to pos")
    if "unbundle" not in a:
        fails.append("1. 634 W missing")
    if "hide_two(" not in a or "n_fr - 2" not in a:
        fails.append("1. two-row hide missing")
    if "pins = sorted(mid_set)" not in a:
        fails.append("1. seeded run depends on set order")
    if "{pin, held_ask}" in a:
        fails.append("1. held filters leftover offer")
    if "all_lines[:cut]" not in a or "all_lines[cut:]" not in a:
        fails.append("1. line split missing")
    if "n_ev < 40" not in a or "len(pos) < 40" not in a:
        fails.append("1. VOID missing")
    if "argmax" not in a:
        fails.append("1. unique not argmax")
    if "min(tr[\"y\"]" in a or "min(tr['y']" in a:
        fails.append("1. PAD clamp retargets teacher")
    gate_src = gate_expression(a)
    if not gate_src:
        fails.append("1. gate missing")
    else:
        if any(tok in gate_src for tok in ("KEEP_623", "kept", "s623", "peak")):
            fails.append("1. 623/peak leaked into the gate")
        if "u_phi - u_rand" not in gate_src or "u_phi - u_init" not in gate_src:
            fails.append("1. learning bars not in gate")
    # Local feat_place must not exist with held, if redefined.
    if "def feat_place(" in a and function_has_held(a, "feat_place"):
        fails.append("1. held leaked into feat_place")
    return fails


def mutants():
    a = AUDIT.read_text(encoding="utf-8")
    cases = [
        a.replace(
            "self.net = nn.Sequential(",
            "self.refuse = nn.Parameter(torch.zeros(1))\n        self.net = nn.Sequential(",
        ),
        a.replace(
            'tr["y"] is not None and tr["feats"] and tr["y"] < min(len(tr["feats"]), PAD)',
            'tr["feats"]',
        ),
        a.replace(
            "from _audit635_placescore import FDIM, PAD, feat_place, with_pmi_rank",
            "from _audit635_placescore import FDIM, PAD, with_pmi_rank\n"
            "def feat_place(kind, tok, pi, pg, env_m, co, df, n_use, held_ask=None):\n"
            "    return [0.0] * FDIM\n",
        ),
        a.replace("opt.step()", "pass"),
        a.replace(
            "pick = int(net.logits(x, mask)[0].argmax().item())",
            "pick = 0",
        ),
        a.replace(
            'y[i] = 0 if tr["y"] is None else int(tr["y"])',
            'y[i] = min(tr["y"], n - 1) if n and tr["y"] is not None else 0',
        ),
        a.replace("pins = sorted(mid_set)", "pins = list(mid_set)"),
        a.replace(
            "pg, pin, qi, env_m, mid_set, high_set, {pin},",
            "pg, pin, qi, env_m, mid_set, high_set, {pin, held_ask},",
        ),
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
