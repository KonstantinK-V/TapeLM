"""Check of 546: the sweep is the only mechanism, and gamma=0 is a real ablation."""
from __future__ import annotations

from pathlib import Path

import _audit546_value as M

SRC = Path("_audit546_value.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    body = src.split('"""', 2)[-1]   # the module docstring names the old faults
    f = []
    if any(x in body for x in ("sharp(", "narrow_next(", "uniq_aug(", "hop1_of")):
        f.append("1. a target quantity leaked into r or into the edge set")
    if "def neighbours(g, ps):" not in src:
        f.append("1. neighbours must be plain adjacency, not the target relation")
    if '("C_flat", Value(net, args.gamma, args.sweeps, args.cost, True))' not in src:
        f.append("1. the constant-r control over the same edges missing")
    if "r = dict(zip(ps, self.net.score([(P, 1) for P in ps], g).tolist()))" not in src:
        f.append("1. r must come from Phi over the place's own counts")
    if "from _audit545a_layer import Layered, places_of, train" not in src:
        f.append("1. 545a Phi reuse missing")
    if "self.gamma * max((V[Q] - self.cost for Q in nbr[P])," not in src:
        f.append("1. the Bellman sweep with the per-hop cost missing")
    if "self.sweeps if self.gamma else 0" not in src:
        f.append("1. gamma=0 must skip the sweeps entirely (the ablation)")
    if '("A_now", Value(net, 0.0, 0, args.cost))' not in src:
        f.append("1. the no-propagation arm missing")
    if '("D_rule", RuleScorer(args.seed))' not in src:
        f.append("1. the single-count bar missing")
    if 'arms["A_now"]["eval_unique"] >= 0.999' not in src:
        f.append("2. VOID on a perfect score (the leak detector) missing")
    if 'paired = len({a["eval_rand"] for a in arms.values()}) == 1' not in src:
        f.append("3. pairing not verified from the baseline itself")
    gate = src[src.find("    gate = "):src.find("    rec = dict")]
    if "lb - la > 0.05" not in gate:
        f.append("2. GATE must beat the immediate payoff")
    if "lb - ld > 0.05" not in gate:
        f.append("2. GATE must beat the single count")
    if "lb - lc > 0.05" not in gate:
        f.append("2. GATE must beat the constant-r sweep (the graph-shape control)")
    return f


MUTANTS = (
    ("the ablation arm secretly propagates",
     '("A_now", Value(net, 0.0, 0, args.cost))',
     '("A_now", Value(net, 0.7, 4, args.cost))',
     "1."),
    ("the sweep loses the per-hop cost",
     "            V = {P: r[P] + self.gamma * max((V[Q] - self.cost for Q in nbr[P]),",
     "            V = {P: r[P] + self.gamma * max((V[Q] for Q in nbr[P]),",
     "1."),
    ("the graph-shape control drops out of the gate",
     "    gate = (not void) and lb - la > 0.05 and lb - ld > 0.05 and lb - lc > 0.05",
     "    gate = (not void) and lb - la > 0.05 and lb - ld > 0.05",
     "2."),
    ("edges come from the target again",
     "        nbr = neighbours(g, ps)",
     "        nbr = {P: hop1_of(P, g) for P in ps}",
     "1."),
    ("the leak detector removed",
     '            or arms["A_now"]["eval_unique"] >= 0.999\n',
     "",
     "2."),
    ("gate no longer needs to beat the immediate payoff",
     "    gate = (not void) and lb - la > 0.05 and lb - ld > 0.05",
     "    gate = (not void) and lb - ld > 0.05",
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
