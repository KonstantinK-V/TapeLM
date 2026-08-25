"""Check of 545a: rank loss, balanced pairs, depth in the target, no lexicon."""
from __future__ import annotations

from pathlib import Path

import _audit545a_layer as M

SRC = Path("_audit545a_layer.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "from _audit544_phi import RuleScorer, features" not in src:
        f.append("1. 544 local features and rule arm reuse missing")
    if "from _audit542_curric import rand_unique" not in src:
        f.append("1. 542 baseline reuse missing")
    if "loss = F.softplus(-(self.net(Xp) - self.net(Xn))).mean()" not in src:
        f.append("1. rank loss missing - training must match the reading")
    if "pairs = [(p, neg[rng.randrange(len(neg))]) for p in pos]" not in src:
        f.append("1. one sampled negative per positive (the balance) missing")
    if "return features(P, g) + [d / 2.0]" not in src:
        f.append("1. the hop index must be part of the input")
    if "global_counts" in src or "gc.get(" in src or "ln.split()" in src:
        f.append("1. corpus frequency is back - the lexicon breaks separation")
    if "hash(" in src or "ord(" in src:
        f.append("1. a token identity enters the features")
    if "pos1 = [(P, 1) for P in ps if opens[P]]" not in src:
        f.append("1. the chain condition (hop1 counts only if it opens hop2) missing")
    if "pool2 = {Q for P in top for Q in hop1_of(P, g) if Q in g[\"slots_at\"]}" not in src:
        f.append("1. layer 2 must be reached THROUGH the layer-1 top, not sampled")
    if "if (Q in d1pos) != (Q in d2pos))" not in src:
        f.append("3. both_rate must count DISAGREEING targets, not membership")
    if "Q = pick_at(scorer, nxt, g, rng_pol, 2)" not in src:
        f.append("1. eval never walks the second hop at depth 2")
    if "e1 = evaluate(lines, args, sc, chain=False)" not in src:
        f.append("3. the hop1-only column (rank teacher alone) missing")
    if "return scorer.pick(pool, g, rng, 0.0, d)" not in src:
        f.append("1. depth must be honoured at eval, not just in training")
    if "if self.frozen or not pos or not neg:" not in src:
        f.append("1. the null must be the same net left untrained")
    if "C_null=Layered(args.seed, frozen=True)" not in src:
        f.append("1. C must be the frozen arm")
    if src.count("build_window(lines, rng_win") != 2:
        f.append("1. train and eval must both draw windows with rng_win")
    if 'rands = {a["eval_rand"] for a in arms.values()}' not in src:
        f.append("3. pairing not verified from the baseline itself")
    if "CrossEntropy" in src:
        f.append("4. CE leaked")
    gate = src[src.find("    gate = "):src.find("    rec = dict")]
    if "lb - ld > 0.05" not in gate:
        f.append("2. GATE must beat the single-count rule")
    if "lb - lc > 0.05" not in gate:
        f.append("2. GATE must beat the shuffled-pair null")
    if 'arms["B_layer"]["pairs_d2"] < 1000' not in src:
        f.append("2. VOID when layer 2 never trained missing")
    if 'arms["B_layer"]["n_second"] < 200' not in src:
        f.append("2. VOID when the second hop was never walked missing")
    if 'arms["B_layer"]["lift_chain"]' not in src:
        f.append("2. GATE must be read on the CHAIN lift, not the hop1 lift")
    return f


MUTANTS = (
    ("back to absolute values instead of ranking",
     "        loss = F.softplus(-(self.net(Xp) - self.net(Xn))).mean()",
     "        loss = ((self.net(Xp) - 1.0) ** 2).mean()",
     "1."),
    ("depth dropped from the input",
     "    return features(P, g) + [d / 2.0]",
     "    return features(P, g) + [0.0]",
     "1."),
    ("chain condition dropped: any hop1 counts",
     "        pos1 = [(P, 1) for P in ps if opens[P]]",
     "        pos1 = [(P, 1) for P in ps if hop1_of(P, g)]",
     "1."),
    ("the null starts training",
     "        if self.frozen or not pos or not neg:",
     "        if not pos or not neg:",
     "1."),
    ("eval stops walking the second hop",
     "                        Q = pick_at(scorer, nxt, g, rng_pol, 2)",
     "                        Q = None",
     "1."),
    ("both_rate back to a membership denominator",
     "            both_num += sum(1 for Q in pool2\n                            if (Q in d1pos) != (Q in d2pos))",
     "            both_num += len(pool2 & d1pos)",
     "3."),
    ("gate read on hop1 instead of the chain",
     '    lb, lc, ld = (arms["B_layer"]["lift_chain"], arms["C_null"]["lift_chain"],',
     '    lb, lc, ld = (arms["B_layer"]["lift_hop1"], arms["C_null"]["lift_hop1"],',
     "2."),
    ("layer-2 VOID removed",
     'arms["B_layer"]["pairs_d2"] < 1000',
     'False',
     "2."),
)


def main() -> int:
    src = SRC.read_text(encoding="utf-8")
    fails = props()
    for name, old, new, tag in MUTANTS:
        n = src.count(old)
        if n != 1:
            fails.append(f"MUTATION {tag} ({name}): its anchor occurs {n} times")
            continue
        saved = dict(M.__dict__)
        mutated = src.replace(old, new, 1)
        try:
            exec(compile(mutated, "<mutant>", "exec"), M.__dict__)
            got = props(src=mutated)
        except Exception as e:
            got = [f"{tag} the mutant raised {type(e).__name__}"]
        finally:
            M.__dict__.clear()
            M.__dict__.update(saved)
        if not any(g.startswith(tag) for g in got):
            fails.append(f"MUTATION {tag} ({name}): re-introduced and check {tag} did not fire")
    for x in fails:
        print("FAIL " + x)
    print(f"{len(fails)} failures" if fails else
          f"all properties hold, and all {len(MUTANTS)} re-introduced failures were caught")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
