"""Check 639: frozen PMI base plus learned exact-place row/context residual."""
from __future__ import annotations

import ast
from pathlib import Path

AUDIT = Path("_audit639_residualctx.py")


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
    if "from _audit638_rowctx import" not in text or "collect" not in text:
        fails.append("1. 638 exact-place row/context W not reused")
    model = source_of(text, "ResidualContextRanker")
    if "pmi + residual" not in model:
        fails.append("1. frozen PMI skip connection missing")
    if (
        "zeros_(self.score[-1].weight)" not in model
        or "zeros_(self.score[-1].bias)" not in model
    ):
        fails.append("1. init is not exactly frozen PMI")
    drop = source_of(text, "drop_skip_atoms")
    if (
        'trial["atom_mask"] & (trial["atoms"][..., 0] > 0.5)' not in drop
        or 'trial["atom_mask"][kind0] = False' not in drop
        or 'trial["atoms"][kind0] = 0.0' not in drop
    ):
        fails.append("1. kind-0 extract/QUERY PMI remains in residual atoms")
    main = source_of(text, "main")
    if (
        "drop_skip_atoms(train)" not in main
        or "drop_skip_atoms(ev)" not in main
        or main.index("drop_skip_atoms(train)") > main.index("fit_normalizer(train)")
    ):
        fails.append("1. kind-0 atoms are not removed before train normalization")
    pairwise = source_of(text, "pairwise_full_feedback")
    if "positives" not in pairwise or "negatives" not in pairwise:
        fails.append("1. all positive/negative place pairs missing")
    if "F.softplus(diff)" not in pairwise:
        fails.append("1. pairwise logistic rank loss missing")
    multipos = source_of(text, "multi_positive_loss")
    if "torch.logsumexp(pos_logits" not in multipos:
        fails.append("1. multi-positive secondary loss missing")
    if "self.refuse" in text:
        fails.append("1. REFUSE class leaked")
    if "Embedding" in text or "vocab_size" in text:
        fails.append("1. vocabulary/token identity leaked")
    if "attach_null_labels(train" not in text or '"null_correct"' not in text:
        fails.append("1. shuffled-label null missing")
    if "make_net(args.seed)" not in text or "torch.manual_seed(seed)" not in text:
        fails.append("1. identical init control missing")
    if "unique_max(" not in text:
        fails.append("1. unique exact-place decision missing")
    unique = source_of(text, "unique_max")
    if "len(winners) == 1 else None" not in unique:
        fails.append("1. ties do not refuse")
    if (
        'rival_names = ("rand", "init", "null", "pmi", "count", "majority")'
        not in text
    ):
        fails.append("1. strongest rival set incomplete")
    gate = gate_expression(text)
    if not gate:
        fails.append("1. gate missing")
    else:
        if "d_strong > BAR" not in gate or "d_init > BAR" not in gate:
            fails.append("1. strongest/init bars missing")
        if any(item in gate for item in ("peak", "oracle")):
            fails.append("1. diagnostic leaked into gate")
    if "all_lines[:cut]" not in text or "all_lines[cut:]" not in text:
        fails.append("1. 70/30 split missing")
    if 'output="exact_place"' not in text or "token_ids=False" not in text:
        fails.append("1. exact-place output contract missing")
    if (
        '"residual_only"' not in source_of(text, "evaluate")
        or "residual_only=rates" not in main
    ):
        fails.append("1. residual-only diagnostic missing")
    if "n_commit=commits" not in main or "n_commit " not in main:
        fails.append("1. commit counts missing")
    if "init/pmi" in main:
        fails.append("1. init and PMI diagnostics are still merged")
    if "opt.step()" not in text:
        fails.append("1. learned residual weights never move")
    return fails


def mutants():
    src = AUDIT.read_text(encoding="utf-8")
    cases = [
        src.replace("score = pmi + residual", "score = residual"),
        src.replace(
            "torch.nn.init.zeros_(self.score[-1].weight)",
            "torch.nn.init.xavier_uniform_(self.score[-1].weight)",
        ),
        src.replace(
            'trial["atom_mask"] & (trial["atoms"][..., 0] > 0.5)',
            'trial["atom_mask"] & (trial["atoms"][..., 0] < -999.0)',
        ),
        src.replace("drop_skip_atoms(train)", "pass  # keep kind 0"),
        src.replace(
            "F.softplus(diff).mean()",
            "positives.mean() * 0.0",
        ),
        src.replace(
            "torch.logsumexp(pos_logits, dim=1)",
            "pos_logits.max(dim=1).values",
        ),
        src.replace("opt.step()", "pass"),
        src.replace(
            'rival_names = ("rand", "init", "null", "pmi", "count", "majority")',
            'rival_names = ("rand", "init")',
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
            "torch.manual_seed(seed)\n    return ResidualContextRanker()",
            "return ResidualContextRanker()",
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
