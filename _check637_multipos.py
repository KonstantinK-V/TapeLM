"""Check 637: multi-positive exact-place teacher and biting rivals."""
from __future__ import annotations

import ast
from pathlib import Path

AUDIT = Path("_audit637_multipos.py")


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


def props(src=None):
    text = AUDIT.read_text(encoding="utf-8") if src is None else src
    fails = []
    if "from _audit635_placescore import" not in text:
        fails.append("1. fixed 635 features not reused")
    if "opt.step()" not in text:
        fails.append("1. no training step")
    if "extracts_633" not in text or "unbundle" not in text:
        fails.append("1. exact-place W missing")
    if "self.refuse" in text:
        fails.append("1. REFUSE class leaked")
    if "Embedding" in text or "vocab_size" in text:
        fails.append("1. vocabulary output leaked")
    if (
        "correct = [\n"
        "                        bool(tok == held_ask) "
        "for _kind, tok, _pi in cands\n"
        "                    ]"
    ) not in text:
        fails.append("1. all-place teacher labels missing")
    if "next(" in function_source(text, "collect"):
        fails.append("1. first-positive teacher came back")
    loss = function_source(text, "multi_positive_loss")
    if (
        "logsumexp(logits" not in loss
        or "logsumexp(pos_logits" not in loss
    ):
        fails.append("1. multi-positive listwise loss missing")
    if "positive = torch.zeros" not in text:
        fails.append("1. positive mask missing")
    if "train_pos = [" not in text or 'any(trial["correct"][:PAD])' not in text:
        fails.append("1. train not filtered to positive trials")
    if "fit_normalizer(train)" not in text:
        fails.append("1. train-only normalization missing")
    make = function_source(text, "make_net")
    if "torch.manual_seed(seed)" not in make or "return RankNet()" not in make:
        fails.append("1. init is not reproducible")
    if text.count("make_net(args.seed)") < 1 or "make_net(seed)" not in text:
        fails.append("1. init/trained model do not share constructor")
    if "hide_two(" not in text or "n_fr - 2" not in text:
        fails.append("1. two-row hide missing")
    if "pins = sorted(mid_set)" not in text:
        fails.append("1. seeded run depends on set order")
    if "{pin, held_ask}" in text:
        fails.append("1. held filters the offer")
    if "all_lines[:cut]" not in text or "all_lines[cut:]" not in text:
        fails.append("1. 70/30 line split missing")
    if "majority_correct" not in text:
        fails.append("1. same-place majority rival missing")
    for rival in ("rand", "init", "pmi", "count", "majority"):
        if f'("{rival}", rates["{rival}"])' not in text:
            fails.append(f"1. {rival} rival missing from strongest")
    unique = function_source(text, "unique_max")
    if (
        "unique_max(" not in text
        or "return hits[0] if len(hits) == 1 else None" not in unique
    ):
        fails.append("1. unique-max/refuse missing")
    gate = gate_expression(text)
    if not gate:
        fails.append("1. gate missing")
    else:
        if "d_strong > BAR" not in gate or "d_init > BAR" not in gate:
            fails.append("1. strong/init bars missing")
        if any(word in gate for word in ("peak", "oracle", "kept", "623")):
            fails.append("1. diagnostic leaked into gate")
    return fails


def mutants():
    src = AUDIT.read_text(encoding="utf-8")
    cases = [
        src.replace(
            "correct = [\n"
            "                        bool(tok == held_ask) for _kind, tok, _pi in cands\n"
            "                    ]",
            "correct = [False for _kind, _tok, _pi in cands]\n"
            "                    for i, (_kind, tok, _pi) in enumerate(cands):\n"
            "                        if tok == held_ask:\n"
            "                            correct[i] = True\n"
            "                            break",
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
            '("majority", rates["majority"]),',
            "",
        ),
        src.replace(
            "gate = d_strong > BAR and d_init > BAR",
            "gate = rates['phi'] - rates['rand'] > BAR",
        ),
        src.replace(
            "torch.manual_seed(seed)\n    return RankNet()",
            "return RankNet()",
        ),
        src.replace(
            "hits[0] if len(hits) == 1 else None",
            "hits[0]",
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
