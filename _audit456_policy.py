"""456: return-table instead of pick_cost. Ablate soon."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from _audit449_adv import build, filter_cands
from _audit451_learn import options
from _audit453_depth import loop, starts
from _audit454_cost import soon, world_both, world_d1, world_d4, world_stop
from _audit455_dash import FAM

OUT = Path("results/_stage456_policy.json")


def teacher_pick(o, c, e, pk, v):
    for r2 in o:
        c2 = filter_cands(c, e | {v[r2[2]]}, pk)
        if len(c2) == 1:
            return r2
    for r2 in o:
        nh = sum(1 for p in c if v[r2[2]] in pk[p])
        if nh == 0:
            return r2
    return None


def train_return(rng, n_each, use_soon):
    win, tot = defaultdict(float), defaultdict(int)
    for fn in FAM.values():
        for _ in range(n_each):
            lines, names = fn(rng)
            g = build(lines)
            if g is None:
                continue
            for cands, visited, used_k, order in starts(g, rng, names["APPLES"]):
                opts = options(cands, order, g["by_key"], visited, used_k,
                               g["slots_at"], g["value"], rng)
                for rec in opts:
                    k, H, pinH, vis1, used1 = rec
                    extra = {g["value"][pinH]}
                    c1 = filter_cands(cands, extra, g["place_keys"])
                    n_hit = sum(1 for p in cands if g["value"][pinH] in g["place_keys"][p])
                    d1 = len(c1)
                    s = soon(g["value"][pinH], order, g["by_key"], vis1, cands, used1,
                             g["place_keys"]) if use_soon else 0
                    hops, ok, _ = loop(
                        g, c1, vis1, used1, list(order) + [g["value"][pinH]],
                        rng, teacher_pick, names["CRISP"])
                    if d1 == 1:
                        hops, ok = 0, True
                    R = (1.0 if ok else 0.0) - 0.05 * (hops + 1)
                    key = (n_hit, d1, s)
                    tot[key] += 1
                    win[key] += R
    return {k: win[k] / tot[k] for k in tot}


def make_pick(table, use_soon, order, g, visited, used_k):
    def pick(opts, cands, extra, pk, v):
        best, br = None, -10 ** 9
        for rec in opts:
            pinH = rec[2]
            vis1, used1 = rec[3], rec[4]
            n_hit = sum(1 for p in cands if v[pinH] in pk[p])
            d1 = len(filter_cands(cands, extra | {v[pinH]}, pk))
            s = soon(v[pinH], order, g["by_key"], vis1, cands, used1, pk) * int(use_soon)
            r = table.get((n_hit, d1, s), -1.0)
            if r > br:
                br, best = r, rec
        return best if br > -1.0 else None
    return pick


def eval_table(rng, fn, n, table, use_soon):
    n_ok = n_ref = n_ep = hops_sum = 0
    for _ in range(n):
        lines, names = fn(rng)
        g = build(lines)
        if g is None:
            continue
        for cands, visited, used_k, order in starts(g, rng, names["APPLES"]):
            n_ep += 1
            pick = make_pick(table, use_soon, order, g, visited, used_k)
            hops, ok, ref = loop(g, set(cands), set(visited), set(used_k), list(order),
                                 rng, pick, names["CRISP"])
            hops_sum += hops
            n_ok += int(ok)
            n_ref += int(ref)
    return dict(
        ep=n_ep,
        pin=n_ok / max(n_ep, 1),
        refuse=n_ref / max(n_ep, 1),
        mean_hops=hops_sum / max(n_ep, 1),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--train", type=int, default=8)
    ap.add_argument("--test", type=int, default=6)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    rng = random.Random(args.seed)
    tab_s = train_return(rng, args.train, True)
    tab_n = train_return(rng, args.train, False)
    yes = {k: eval_table(rng, fn, args.test, tab_s, True) for k, fn in FAM.items()}
    no = {k: eval_table(rng, fn, args.test, tab_n, False) for k, fn in FAM.items()}
    void = any(yes[k]["ep"] < 5 for k in FAM)
    gate = ((not void)
            and (yes["D1"]["pin"] == 1.0) and (yes["D1"]["mean_hops"] == 1.0)
            and (yes["D2"]["pin"] == 1.0) and (yes["D2"]["mean_hops"] == 2.0)
            and (yes["BOTH"]["pin"] == 1.0) and (yes["BOTH"]["mean_hops"] == 2.0)
            and (yes["D4"]["pin"] == 1.0) and (yes["D4"]["mean_hops"] == 4.0)
            and (yes["STOP"]["refuse"] == 1.0) and (yes["STOP"]["pin"] == 0.0)
            and (no["BOTH"]["mean_hops"] > 2.0))
    rec = dict(seed=args.seed, void=bool(void), gate=bool(gate),
               with_soon=yes, no_soon=no,
               n_keys_soon=len(tab_s), n_keys_nosoon=len(tab_n))
    print("with soon")
    for k in FAM:
        r = yes[k]
        print(f"  {k:4} pin {r['pin']:.2f} ref {r['refuse']:.2f} hops {r['mean_hops']:.1f}")
    print("no soon  BOTH hops", round(no["BOTH"]["mean_hops"], 2),
          "pin", round(no["BOTH"]["pin"], 2))
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("\nVOID: a family had <5 test eps.")
    elif gate:
        print("\nGO POLICY: return-table matches 455; without soon BOTH is not the short path.")
    else:
        print("\nSTOP: return-table missed a family, or soon was unused.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
