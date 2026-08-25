"""Check 633: unbundle DIRECT+HOPONLY; peak is control; no Φ; conserve 623."""
from __future__ import annotations

import ast
from pathlib import Path

AUDIT = Path("_audit633_gapcon.py")


def function_has_held(src, name):
    fn = next(
        node for node in ast.parse(src).body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    names = [node.id for node in ast.walk(fn) if isinstance(node, ast.Name)]
    names.extend(arg.arg for arg in fn.args.args)
    return any("held" in item.lower() for item in names)


def function_source(src, name):
    fn = next(
        node for node in ast.parse(src).body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    return ast.get_source_segment(src, fn)


def props(src=None):
    a = AUDIT.read_text(encoding="utf-8") if src is None else src
    fails = []
    if "import torch" in a or "torch.nn" in a:
        fails.append("1. Φ/torch leaked")
    if "pmi_rank" in a or "set(cands)" in a:
        fails.append("1. hash-order PMI leaked")
    if "commit_resolved" not in a or "peak_ora" not in a:
        fails.append("1. 630 peak control missing")
    if 'cands.append(("direct", door, door_pi))' not in a:
        fails.append("1. DIRECT place missing")
    if 'cands.append(("direct", rec["door"]' in a:
        fails.append("1. DIRECT ranks door words, not exact places")
    if 'cands.append(("hop1"' not in a:
        fails.append("1. HOPONLY place missing")
    if "search_623" not in a or "leftover_doors" not in a:
        fails.append("1. paired 623 missing")
    if (
        "BAR_U = 0.17" not in a
        or "BAR_H = 0.05" not in a
        or "KEEP_623 = 0.90" not in a
    ):
        fails.append("1. conservation bar missing")
    if (
        "u_ora >= need" not in a
        or "d_ora >= BAR_D" not in a
        or "h_ora >= BAR_H" not in a
    ):
        fails.append("1. UNION/DIRECT/HOPONLY gate missing")
    if "(u_ora - peak_ora) >= BAR_PEAK" not in a:
        fails.append("1. unbundle-beats-peak missing")
    if "n_live < 40" not in a:
        fails.append("1. VOID missing")
    if "hide_two(" not in a or "n_fr - 2" not in a:
        fails.append("1. two-row hide missing")
    if "if hat is not None:" not in a:
        fails.append("1. 618 refuse-only missing")
    if "if not cands:" in a:
        fails.append("1. empty exact offer removed from paired denominator")
    if function_has_held(a, "unbundle"):
        fails.append("1. held leaked into unbundle")
    if function_has_held(a, "open_hop1_623"):
        fails.append("1. held leaked into hop1 offer")
    if "{pin, held_ask}" in a:
        fails.append("1. held filters paired 623 offer")
    if "if door == held_ask:" not in a:
        fails.append("1. paired 623 DIRECT missing")
    hop_src = function_source(a, "open_hop1_623")
    if "if scanned >= K:" not in hop_src or "scanned += 1" not in hop_src:
        fails.append("1. unbundle does not match first-K 623 cards")
    elif hop_src.index("scanned += 1") > hop_src.index("extract_633("):
        fails.append("1. invalid hop1 cards buy replacements")
    unbundle_src = function_source(a, "unbundle")
    if (
        "seen_door_places" not in unbundle_src
        or "door, _bag, uniq = extract_633(" not in unbundle_src
    ):
        fails.append("1. DIRECT is not one resolved literal per exact address")
    extract_src = function_source(a, "extract_633")
    if (
        "list(dict.fromkeys(uniq))" not in extract_src
        or "for word in sorted(env_m):" not in extract_src
        or "max(uniq, key=score)" not in extract_src
    ):
        fails.append("1. exact read is not deterministic tape-order PMI")
    if "def hop1_623(" not in a or "if hop1_623(" not in a:
        fails.append("1. paired 623 does not share exact read")
    if "rows_p[:K]" not in a:
        fails.append("1. residual exclusion does not share 623 K")
    if "priced" in a and "Priced-from-623 is not a bar" not in a:
        fails.append("1. 623 priced leaked into the gate")
    if "GapPolicy" in a or "Linear(" in a:
        fails.append("1. learner leaked")
    return fails


def mutants():
    a = AUDIT.read_text(encoding="utf-8")
    cases = [
        a.replace(
            'cands.append(("hop1", obs["tok"], obs["hop_pi"]))',
            "pass",
        ),
        a.replace(
            'cands.append(("direct", door, door_pi))',
            "pass",
        ),
        a.replace("(u_ora - peak_ora) >= BAR_PEAK", "True"),
        a.replace("u_ora >= need", "True"),
        a.replace("h_ora >= BAR_H", "True"),
        a.replace("BAR_U = 0.17", "BAR_U = 0.0"),
        a.replace("n_live < 40", "n_live < 0"),
        a.replace(
            "pg, pin, skip, env_m, mid_set, high_set, {pin},",
            "pg, pin, skip, env_m, mid_set, high_set, {pin, held_ask},",
        ),
        a.replace(
            "        scanned += 1",
            "        scanned += int(bool(uniq))",
        ),
        a.replace(
            "                    cands, peak = unbundle(",
            "                    if not cands:\n"
            "                        continue\n"
            "                    cands, peak = unbundle(",
        ),
        a.replace(
            "uniq = list(dict.fromkeys(uniq))",
            "uniq = list(set(uniq))",
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
