"""The dozen numbers a two-hole run is read by - void conditions first, claim second.

Same discipline as _read299: the ceiling before the score, the counts before the rates, and
seeds pooled only within one arm. What is different here is WHICH void condition binds. On the
walk it was `reachable_rate` - can the tape answer at all. On pairs it is `both_offered`: no
world of a question contains the true pair unless both truths were on offer, so every rate
below is capped by it, and a run where it collapses has measured the offer, not the mind.

On COMP_ONLY / COMP_STRICT counting is zero BY CONSTRUCTION, so McNemar against it is a
tautology. Random 1/|offer_a|x|offer_b| is also the wrong floor: a mind that is independently
not-bad at each hole already clears it at right_a * right_b without composing. The bar is
`indep_expected` - the product of the mind's OWN per-hole hits inside the subset. At or below
that product nothing was composed. COMP_STRICT is the claim (all three rivals blind); COMP_ONLY
is printed beside it for continuity with 308.

    python _read309.py out/_stage289_decision_309b_s*.json --held
"""
from __future__ import annotations

import json
import math
import sys

VOID = ("both_offered", "offered_a", "offered_b", "joint_seen_rate", "bag_seen_rate",
        "in_own_both", "mean_pair_worlds", "world_rows", "first_hole_rate")
CLAIM = ("mind_exact", "mind_exact_of_offered", "holes_right_mean",
         "marginal_exact", "joint_exact", "bag_exact")


def z_of(b, c):
    return (b - c) / math.sqrt(b + c) if b + c else float("nan")


def one(path, arm):
    d = json.load(open(path, encoding="utf-8"))
    pc = d.get("pair")
    if not pc or not pc.get(arm):
        return None
    r = pc[arm]
    cfg = (pc.get("cands"), pc.get("max_rows"), pc.get("per_line"), pc.get("follow"),
           pc.get("blind", pc.get("independent")), pc.get("frame_max"),
           d.get("tape_sample"), d.get("import_k"))
    print(f"\n{path}  [{arm}]  {d['wall_s']:.0f}s  cands={pc.get('cands')} "
          f"follow={pc.get('follow')} blind={pc.get('blind', pc.get('independent'))} "
          f"frame_max={pc.get('frame_max')} sample={d.get('tape_sample')} "
          f"import_k={d.get('import_k')} seed={d['seed']}")
    print(f"  tape   n {r['n']}   resample_overlap {d['resample']['mean_overlap']:.3f}   "
          f"params {d['params']}")
    print("  void   " + "  ".join(f"{k.replace('_rate', '')} {r[k]:.4f}"
                                  for k in VOID if k in r and r[k] == r[k]))
    print("  claim  " + "  ".join(f"{k} {r[k]:.4f}" for k in CLAIM if k in r and r[k] == r[k]))
    vm = r.get("vs_marginal_offered") or {}
    if vm.get("n"):
        print(f"  vs MARGINAL (offered)  mind {vm['mind_only']} / rival {vm['rival_only']} "
              f"of {vm['n']}   z {vm['mcnemar_z']:+.2f}"
              f"{'   UNDERPOWERED' if vm.get('underpowered') else ''}")
    # THE CLAIM LIVES ON THE STRICT SUBSET when the report carries one - all three counting
    # routes blind - and the bar inside it is indep_expected, not the random floor: at or below
    # the product of the mind's own marginals nothing was composed. COMP_ONLY is printed too,
    # for continuity with 308.
    for nm in ("COMP_STRICT", "COMP_ONLY"):
        co = r.get(nm) or {}
        if co.get("n"):
            ind = co.get("indep_expected", float("nan"))
            print(f"  {nm}  {co['mind_right']} / {co['n']}   hit {co['hit_rate']:.4f}   "
                  f"floor {co['random_floor']:.4f}   indep {ind:.2f}   "
                  f"z {co['binomial_z']:+.2f}   (one hole {co['one_hole_mean']:.4f})")
    co = r.get("COMP_STRICT") or r.get("COMP_ONLY") or {}
    if co.get("n"):
        e = co.get("indep_expected", float("nan"))
        if e != e:
            e = co.get("random_floor", 0) * co["n"]
        return co["mind_right"], co["n"], e, cfg
    return 0, 0, 0.0, cfg


def main(argv) -> int:
    files = [a for a in argv if not a.startswith("--")]
    if not files:
        print(__doc__)
        return 1
    arms = ["held_out"] if "--held" in argv else ["held_out", "train_control"]
    tot = {a: [0, 0, 0.0] for a in arms}
    cfgs = {a: set() for a in arms}
    for f in files:
        for a in arms:
            got = one(f, a)
            if got:
                tot[a][0] += got[0]
                tot[a][1] += got[1]
                tot[a][2] += got[2]
                cfgs[a].add(got[3])
    if len(files) > 1:
        for a in arms:
            k, n, e = tot[a]
            if len(cfgs[a]) > 1:
                print(f"\nNOT POOLED ({a}): {len(cfgs[a])} different arms among these files. "
                      f"Pool seeds of ONE configuration; arms are compared, not summed.")
                continue
            if not n:
                continue
            z = (k - e) / math.sqrt(e * (1 - e / n)) if e > 0 else float("nan")
            print(f"\nPOOLED {a} over {len(files)} runs: claim {k} / {n}   "
                  f"indep expected {e:.1f}   z {z:+.2f}"
                  f"{'   UNDERPOWERED' if math.sqrt(max(e, 1e-9)) <= 1.645 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
