"""454: cost/utility. Don't go deeper if a shorter unique path exists.

soon(H) = opened place has a key that unique-filters C  (1-ply, no hole peek)
Pick: 451 table, then prefer soon=1.

BOTH  red→UNLOCK→TAG (2) and cat→U1→U2→U3→TAG (4). GATE hops==2
D4    long only. GATE hops==4
D1    one seek. GATE hops==1
STOP  no unique path. GATE refuse
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from _audit449_adv import build, filter_cands
from _audit450_follow import opened
from _audit451_learn import greedy_pick, options
from _audit453_depth import (
    loop, starts, train_mixed, world_d1, world_d4, world_stop, _chain,
)
from _audit452_xfer import _lex, _pad

OUT = Path("results/_stage454_cost.json")


def world_both(rng):
    n = _lex(rng)
    p, r, t, q, red = _chain(n, rng)
    cat, sat = n["cat"], n["sat"]
    mark = n["n4"]
    c1 = [f"{n['aa']} {n['bb']} {n['cc']} {n['CRISP']} {n['FRESH']} {mark} {n['TAG']}" + _pad(50 + i) for i in range(3)]
    c1.append(f"{n['aa']} {n['bb']} {n['cc']} {n['SOFT']} {n['FRESH']} {mark} {n['TAG']}" + _pad(53))
    c2 = [f"{n['dd']} {n['ee']} {n['ff']} {n['RIPE']} {n['FRESH']} {n['KEYA']} {n['u1']}" + _pad(60 + i) for i in range(3)]
    c2.append(f"{n['dd']} {n['ee']} {n['ff']} {n['RAW']} {n['FRESH']} {n['KEYA']} {n['u1']}" + _pad(63))
    c3 = [f"{n['gg']} {n['hh']} {n['ii']} {n['DRY']} {n['FRESH']} {n['KEYA']} {n['u2']}" + _pad(70 + i) for i in range(3)]
    c3.append(f"{n['gg']} {n['hh']} {n['ii']} {n['MUSH']} {n['FRESH']} {n['KEYA']} {n['u2']}" + _pad(73))
    c4 = [f"{n['jj']} {n['kk']} {n['ll']} {n['WET']} {n['FRESH']} {n['u3']} {n['u4']}" + _pad(80 + i) for i in range(3)]
    c4.append(f"{n['jj']} {n['kk']} {n['ll']} {n['DAMP']} {n['FRESH']} {n['u3']} {n['u4']}" + _pad(83))
    hA = [f"{n['clerk']} {n['wrote']} {n['KEYA']} {sat} {n['board']} {n['now']}" + _pad(90 + i) for i in range(3)]
    hA.append(f"{n['clerk']} {n['wrote']} {n['FOG']} {sat} {n['board']} {n['now']}" + _pad(93))
    hS = [f"{n['desk']} {n['shows']} {n['UNLOCK']} {red} {n['label']} {n['here']}" + _pad(100 + i) for i in range(3)]
    hS.append(f"{n['desk']} {n['shows']} {n['FOG']} {red} {n['label']} {n['here']}" + _pad(103))
    hT = [f"{n['vault']} {n['keeps']} {n['TAG']} {n['UNLOCK']} {mark} {n['here']}" + _pad(110 + i) for i in range(3)]
    hT.append(f"{n['vault']} {n['keeps']} {n['FOG']} {n['UNLOCK']} {mark} {n['here']}" + _pad(113))
    u1, u2, u3 = n["n1"], n["n2"], n["n3"]
    h1 = [f"{n['xx']} {n['yy']} {u1} {cat} {n['fruit']} {n['now']}" + _pad(120 + i) for i in range(3)]
    h1.append(f"{n['xx']} {n['yy']} {n['FOG']} {cat} {n['fruit']} {n['now']}" + _pad(123))
    h2 = [f"{n['blue']} {n['dog']} {u2} {u1} {n['store']} {n['yes']}" + _pad(130 + i) for i in range(3)]
    h2.append(f"{n['blue']} {n['dog']} {n['FOG']} {u1} {n['store']} {n['yes']}" + _pad(133))
    h3 = [f"{n['kids']} {n['like']} {u3} {u2} {n['today']} {n['here']}" + _pad(140 + i) for i in range(3)]
    h3.append(f"{n['kids']} {n['like']} {n['FOG']} {u2} {n['today']} {n['here']}" + _pad(143))
    h4 = [f"{n['barns']} {n['lay']} {n['TAG']} {u3} {n['sun']} {n['now']}" + _pad(150 + i) for i in range(3)]
    h4.append(f"{n['barns']} {n['lay']} {n['FOG']} {u3} {n['sun']} {n['now']}" + _pad(153))
    return p + r + t + c1 + c2 + c3 + c4 + hA + hS + hT + h1 + h2 + h3 + h4 + q, n


def soon(vH, order, by_key, visited, cands, used_k, place_keys):
    for _, H in opened(vH, order, by_key, visited, cands, used_k):
        for key in place_keys.get(H, ()):
            hit = sum(1 for p in cands if key in place_keys[p])
            if hit == 1:
                return 1
    return 0


def pick_cost(opts, cands, extra, place_keys, value, rate, order, by_key, visited, used_k):
    scored = []
    for rec in opts:
        pinH = rec[2]
        vis1, used1 = rec[3], rec[4]
        n_hit = sum(1 for p in cands if value[pinH] in place_keys[p])
        d1 = len(filter_cands(cands, extra | {value[pinH]}, place_keys))
        r = rate.get((n_hit, d1), 0.0)
        s = soon(value[pinH], order, by_key, vis1, cands, used1, place_keys)
        scored.append((r, s, -d1, rec))
    scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    if not scored or scored[0][0] <= 0:
        return None
    return scored[0][3]


FAM = {"D1": world_d1, "BOTH": world_both, "D4": world_d4, "STOP": world_stop}


def eval_fam(rng, fn, n, rate):
    n_ok = n_ref = n_ep = hops_sum = hops_g = 0
    for _ in range(n):
        lines, names = fn(rng)
        g = build(lines)
        if g is None:
            continue
        for cands, visited, used_k, order in starts(g, rng, names["APPLES"]):
            n_ep += 1

            def pick(o, c, e, pk, v):
                return pick_cost(o, c, e, pk, v, rate, order, g["by_key"], visited, used_k)

            hops, ok, ref = loop(g, set(cands), set(visited), set(used_k), list(order),
                                 rng, pick, names["CRISP"])
            hops_sum += hops
            n_ok += int(ok)
            n_ref += int(ref)
            gp = lambda o, c, e, pk, v: greedy_pick(o, c, e, pk, v)
            gh, gok, _ = loop(g, set(cands), set(visited), set(used_k), list(order),
                              rng, gp, names["CRISP"])
            hops_g += gh
    return dict(
        ep=n_ep,
        pin=n_ok / max(n_ep, 1),
        refuse=n_ref / max(n_ep, 1),
        mean_hops=hops_sum / max(n_ep, 1),
        greedy_hops=hops_g / max(n_ep, 1),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--train", type=int, default=8)
    ap.add_argument("--test", type=int, default=6)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    rng = random.Random(args.seed)
    rate = train_mixed(rng, args.train)
    reps = {k: eval_fam(rng, fn, args.test, rate) for k, fn in FAM.items()}
    void = any(reps[k]["ep"] < 5 for k in FAM)
    gate = ((not void)
            and (reps["D1"]["pin"] == 1.0) and (reps["D1"]["mean_hops"] == 1.0)
            and (reps["BOTH"]["pin"] == 1.0) and (reps["BOTH"]["mean_hops"] == 2.0)
            and (reps["D4"]["pin"] == 1.0) and (reps["D4"]["mean_hops"] == 4.0)
            and (reps["STOP"]["refuse"] == 1.0) and (reps["STOP"]["pin"] == 0.0)
            and (reps["BOTH"]["greedy_hops"] > 2.0))
    rec = dict(seed=args.seed, void=bool(void), gate=bool(gate),
               rate={f"{a},{b}": round(v, 3) for (a, b), v in rate.items()},
               **reps)
    for k in FAM:
        r = reps[k]
        print(f"{k:4} ep {r['ep']:2} pin {r['pin']:.2f} refuse {r['refuse']:.2f} "
              f"hops {r['mean_hops']:.1f} greedy_hops {r['greedy_hops']:.1f}")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("\nVOID: a family had <5 test eps.")
    elif gate:
        print("\nGO COST: BOTH takes the 2-hop unique, not the 4-hop. D4-only still pays 4. STOP refuses.")
    else:
        print("\nSTOP: took the long path when a short unique existed, or a family failed.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
