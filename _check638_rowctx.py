"""Check 638: learned tape-row/QUERY/VIA encoder still chooses exact places."""
from __future__ import annotations

import ast
from pathlib import Path

AUDIT = Path("_audit638_rowctx.py")


def function_node(src, name):
    return next(
        node for node in ast.parse(src).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name
    )


def function_source(src, name):
    return ast.get_source_segment(src, function_node(src, name)) or ""


def names_in(src, name):
    node = function_node(src, name)
    names = [item.id for item in ast.walk(node) if isinstance(item, ast.Name)]
    if isinstance(node, ast.FunctionDef):
        names.extend(arg.arg for arg in node.args.args)
    return names


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
    text = AUDIT.read_text(encoding="utf-8") if src is None else src
    fails = []
    if "class RowContextRanker" not in text:
        fails.append("1. learned row/context encoder missing")
    if "self.atom_net" not in text or "self.base_net" not in text:
        fails.append("1. variable row atoms are not learned")
    if "Embedding" in text or "vocab_size" in text or "token_to_id" in text:
        fails.append("1. token/vocabulary identities entered weights")
    if "contextual_offer(" not in text or "open_hop1_623(" not in text:
        fails.append("1. exact 634 W with hop provenance missing")
    offer = function_source(text, "contextual_offer")
    if any("held" in name.lower() for name in names_in(text, "contextual_offer")):
        fails.append("1. held leaked into offer")
    if (
        'kind="direct"' not in offer
        or 'kind="hop1"' not in offer
        or "via_pi=door_pi" not in offer
    ):
        fails.append("1. DIRECT/HOPONLY/VIA exact pointers missing")
    relation = function_source(text, "relation_atom")
    if (
        "co.get((left, right)" not in relation
        or "df.get(left" not in relation
        or "left == right" not in relation
    ):
        fails.append("1. tape relation atom missing")
    if any("held" in name.lower() for name in names_in(text, "relation_atom")):
        fails.append("1. held leaked into relation atom")
    candidate = function_source(text, "candidate_tensors")
    for marker in (
        "output -> QUERY",
        "CURRENT -> QUERY",
        "output -> CURRENT",
        "output -> VIA",
        "address -> VIA",
        "address -> QUERY",
        "filler rows -> QUERY",
    ):
        if marker not in candidate:
            fails.append(f"1. context channel missing: {marker}")
    if any("held" in name.lower() for name in names_in(text, "candidate_tensors")):
        fails.append("1. held leaked into candidate context")
    if "correct[ci] = candidate[\"tok\"] == held_ask" not in text:
        fails.append("1. all-place record teacher missing")
    loss = function_source(text, "multi_positive_loss")
    if (
        "torch.logsumexp(logits" not in loss
        or "torch.logsumexp(pos_logits" not in loss
    ):
        fails.append("1. multi-positive loss missing")
    if "self.refuse" in text:
        fails.append("1. REFUSE class leaked")
    if "fit_normalizer(train)" not in text:
        fails.append("1. train-only normalization missing")
    if "attach_null_labels(train" not in text or '"null_correct"' not in text:
        fails.append("1. shuffled-label null missing")
    if "make_net(args.seed)" not in text or "torch.manual_seed(seed)" not in text:
        fails.append("1. identical deterministic init missing")
    if "hide_two(" not in text or "n_fr - 2" not in text:
        fails.append("1. honest two-row hide missing")
    if "pins = sorted(mid_set)" not in text:
        fails.append("1. seeded run depends on set order")
    if "{pin, held_ask}" in text:
        fails.append("1. held filters the candidate offer")
    if "all_lines[:cut]" not in text or "all_lines[cut:]" not in text:
        fails.append("1. disjoint 70/30 line split missing")
    if "unique_max(" not in text:
        fails.append("1. unique place or miss rule missing")
    unique = function_source(text, "unique_max")
    if "len(winners) == 1 else None" not in unique:
        fails.append("1. score tie does not refuse")
    for rival in ("rand", "init", "null", "pmi", "count", "majority"):
        if f'"{rival}"' not in function_source(text, "main"):
            fails.append(f"1. biting rival missing: {rival}")
    if (
        'rival_names = ("rand", "init", "null", "pmi", "count", "majority")'
        not in text
    ):
        fails.append("1. strongest set omits a biting rival")
    if 'output="exact_place"' not in text or "token_ids=False" not in text:
        fails.append("1. output alphabet is not auditable as exact places")
    gate = gate_expression(text)
    if not gate:
        fails.append("1. gate missing")
    else:
        if "d_strong > BAR" not in gate or "d_init > BAR" not in gate:
            fails.append("1. strongest/init learning bars missing")
        if any(word in gate for word in ("peak", "no_via", "context_lift")):
            fails.append("1. diagnostic leaked into gate")
    if "opt.step()" not in text:
        fails.append("1. weights never move")
    return fails


def mutants():
    src = AUDIT.read_text(encoding="utf-8")
    cases = [
        src.replace(
            "via_pi=door_pi,",
            "via_pi=None,",
        ),
        src.replace(
            "co.get((left, right), 0)",
            "0",
        ),
        src.replace(
            'correct[ci] = candidate["tok"] == held_ask',
            "correct[ci] = ci == 0",
        ),
        src.replace(
            "torch.logsumexp(pos_logits, dim=1)",
            "pos_logits.max(dim=1).values",
        ),
        src.replace("opt.step()", "pass"),
        src.replace("pins = sorted(mid_set)", "pins = list(mid_set)"),
        src.replace(
            "pg, pin, qi, env_m, mid_set, high_set, {pin},",
            "pg, pin, qi, env_m, mid_set, high_set, {pin, held_ask},",
        ),
        src.replace(
            "gate = (not void) and d_strong > BAR and d_init > BAR",
            "gate = rates['phi'] - rates['rand'] > BAR",
        ),
        src.replace(
            "return winners[0] if len(winners) == 1 else None",
            "return winners[0]",
        ),
        src.replace(
            'rival_names = ("rand", "init", "null", "pmi", "count", "majority")',
            'rival_names = ("rand", "init")',
        ),
        src.replace(
            "torch.manual_seed(seed)\n    return RowContextRanker()",
            "return RowContextRanker()",
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
