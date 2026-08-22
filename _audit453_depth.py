"""453: choose depth, don't hardcode hops==2.

Worlds (random lex, shuffled P keys):
  D1   one seek: unique-read is TAG → CRISP
  D2   449: UNLOCK then TAG
  D4   U1→U2→U3→TAG
  STOP 4 cands, KEYA shrinks, no TAG path → refuse

Agent: while |C|>1: pick by 451 table (n_hit,d1); no opts → refuse.
Budget 8 is a fuse, not the answer. GATE does not mention hops==2.

Train mixed D1/D2/D4/STOP. Test held-out names.
GATE  D1/D2/D4 pin==1 hops 1/2/4; STOP refuse; greedy uses extra trap hops
VOID  any family < 5 test eps

    python _check453_depth.py
    python _audit453_depth.py --seed 1337
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from _audit440_compose import think_place, think_slot
from _audit447_seek import hop, next_cands
from _audit449_adv import build, filter_cands
from _audit451_learn import greedy_pick, options
from _audit452_xfer import _lex, _pad, learned_pick, world as world452

OUT = Path("results/_stage453_depth.json")


def _chain(n, rng):
    ctx = [n["red"], n["cat"], n["sat"], n["on"], n["the"], n["mat"]]
    rng.shuffle(ctx)
    p = [f"{ctx[0]} {ctx[1]} {ctx[2]} {n['APPLES']} {ctx[3]} {ctx[4]} {ctx[5]}" + _pad(i) for i in range(3)]
    p.append(f"{ctx[0]} {ctx[1]} {ctx[2]} {n['ORANGES']} {ctx[3]} {ctx[4]} {ctx[5]}" + _pad(3))
    r = [f"{n['kids']} {n['like']} {n['SWEET']} {n['APPLES']} {n['today']} {n['yes']}" + _pad(20 + i) for i in range(3)]
    r.append(f"{n['kids']} {n['like']} {n['SOUR']} {n['APPLES']} {n['today']} {n['yes']}" + _pad(23))
    t = [f"{n['barns']} {n['store']} {n['FRESH']} {n['SWEET']} {n['fruit']} {n['now']}" + _pad(30 + i) for i in range(3)]
    t.append(f"{n['barns']} {n['store']} {n['STALE']} {n['SWEET']} {n['fruit']} {n['now']}" + _pad(33))
    q = []
    for i, f in enumerate((n["PEARS"], n["PLUMS"], n["PEARS"], n["PLUMS"])):
        q.append(f"{n['blue']} {n['dog']} {n['lay']} {f} {n['in']} {n['sun']}" + _pad(10 + i))
    return p, r, t, q, n["red"]


def world_d1(rng):
    n = _lex(rng)
    p, r, t, q, red = _chain(n, rng)
    c1 = [f"{n['aa']} {n['bb']} {n['cc']} {n['CRISP']} {n['FRESH']} {n['TAG']} {n['u1']}" + _pad(50 + i) for i in range(3)]
    c1.append(f"{n['aa']} {n['bb']} {n['cc']} {n['SOFT']} {n['FRESH']} {n['TAG']} {n['u1']}" + _pad(53))
    c2 = [f"{n['dd']} {n['ee']} {n['ff']} {n['RIPE']} {n['FRESH']} {n['u2']} {n['u3']}" + _pad(60 + i) for i in range(3)]
    c2.append(f"{n['dd']} {n['ee']} {n['ff']} {n['RAW']} {n['FRESH']} {n['u2']} {n['u3']}" + _pad(63))
    h = [f"{n['vault']} {n['keeps']} {n['TAG']} {red} {n['safe']} {n['here']}" + _pad(110 + i) for i in range(3)]
    h.append(f"{n['vault']} {n['keeps']} {n['FOG']} {red} {n['safe']} {n['here']}" + _pad(113))
    return p + r + t + c1 + c2 + h + q, n


def world_d2(rng):
    return world452(rng, neg=False)


def world_d4(rng):
    n = _lex(rng)
    p, r, t, q, red = _chain(n, rng)
    c1 = [f"{n['aa']} {n['bb']} {n['cc']} {n['CRISP']} {n['FRESH']} {n['KEYA']} {n['TAG']}" + _pad(50 + i) for i in range(3)]
    c1.append(f"{n['aa']} {n['bb']} {n['cc']} {n['SOFT']} {n['FRESH']} {n['KEYA']} {n['TAG']}" + _pad(53))
    c2 = [f"{n['dd']} {n['ee']} {n['ff']} {n['RIPE']} {n['FRESH']} {n['KEYA']} {n['u1']}" + _pad(60 + i) for i in range(3)]
    c2.append(f"{n['dd']} {n['ee']} {n['ff']} {n['RAW']} {n['FRESH']} {n['KEYA']} {n['u1']}" + _pad(63))
    c3 = [f"{n['gg']} {n['hh']} {n['ii']} {n['DRY']} {n['FRESH']} {n['KEYA']} {n['u2']}" + _pad(70 + i) for i in range(3)]
    c3.append(f"{n['gg']} {n['hh']} {n['ii']} {n['MUSH']} {n['FRESH']} {n['KEYA']} {n['u2']}" + _pad(73))
    c4 = [f"{n['jj']} {n['kk']} {n['ll']} {n['WET']} {n['FRESH']} {n['u3']} {n['u4']}" + _pad(80 + i) for i in range(3)]
    c4.append(f"{n['jj']} {n['kk']} {n['ll']} {n['DAMP']} {n['FRESH']} {n['u3']} {n['u4']}" + _pad(83))
    hA = [f"{n['clerk']} {n['wrote']} {n['KEYA']} {n['cat']} {n['board']} {n['now']}" + _pad(90 + i) for i in range(3)]
    hA.append(f"{n['clerk']} {n['wrote']} {n['FOG']} {n['cat']} {n['board']} {n['now']}" + _pad(93))
    u1, u2, u3 = n["UNLOCK"], n["n1"], n["n2"]
    h1 = [f"{n['desk']} {n['shows']} {u1} {red} {n['label']} {n['here']}" + _pad(100 + i) for i in range(3)]
    h1.append(f"{n['desk']} {n['shows']} {n['FOG']} {red} {n['label']} {n['here']}" + _pad(103))
    h2 = [f"{n['xx']} {n['yy']} {u2} {u1} {n['safe']} {n['here']}" + _pad(120 + i) for i in range(3)]
    h2.append(f"{n['xx']} {n['yy']} {n['FOG']} {u1} {n['safe']} {n['here']}" + _pad(123))
    h3 = [f"{n['n3']} {n['n4']} {u3} {u2} {n['store']} {n['now']}" + _pad(130 + i) for i in range(3)]
    h3.append(f"{n['n3']} {n['n4']} {n['FOG']} {u2} {n['store']} {n['now']}" + _pad(133))
    h4 = [f"{n['vault']} {n['keeps']} {n['TAG']} {u3} {n['fruit']} {n['here']}" + _pad(140 + i) for i in range(3)]
    h4.append(f"{n['vault']} {n['keeps']} {n['FOG']} {u3} {n['fruit']} {n['here']}" + _pad(143))
    return p + r + t + c1 + c2 + c3 + c4 + hA + h1 + h2 + h3 + h4 + q, n


def world_stop(rng):
    n = _lex(rng)
    p, r, t, q, red = _chain(n, rng)
    c1 = [f"{n['aa']} {n['bb']} {n['cc']} {n['CRISP']} {n['FRESH']} {n['KEYA']} {n['u1']}" + _pad(50 + i) for i in range(3)]
    c1.append(f"{n['aa']} {n['bb']} {n['cc']} {n['SOFT']} {n['FRESH']} {n['KEYA']} {n['u1']}" + _pad(53))
    c2 = [f"{n['dd']} {n['ee']} {n['ff']} {n['RIPE']} {n['FRESH']} {n['KEYA']} {n['u2']}" + _pad(60 + i) for i in range(3)]
    c2.append(f"{n['dd']} {n['ee']} {n['ff']} {n['RAW']} {n['FRESH']} {n['KEYA']} {n['u2']}" + _pad(63))
    c3 = [f"{n['gg']} {n['hh']} {n['ii']} {n['DRY']} {n['FRESH']} {n['KEYA']} {n['u3']}" + _pad(70 + i) for i in range(3)]
    c3.append(f"{n['gg']} {n['hh']} {n['ii']} {n['MUSH']} {n['FRESH']} {n['KEYA']} {n['u3']}" + _pad(73))
    c4 = [f"{n['jj']} {n['kk']} {n['ll']} {n['WET']} {n['FRESH']} {n['u4']} {n['n1']}" + _pad(80 + i) for i in range(3)]
    c4.append(f"{n['jj']} {n['kk']} {n['ll']} {n['DAMP']} {n['FRESH']} {n['u4']} {n['n1']}" + _pad(83))
    hA = [f"{n['clerk']} {n['wrote']} {n['KEYA']} {red} {n['board']} {n['now']}" + _pad(90 + i) for i in range(3)]
    hA.append(f"{n['clerk']} {n['wrote']} {n['FOG']} {red} {n['board']} {n['now']}" + _pad(93))
    return p + r + t + c1 + c2 + c3 + c4 + hA + q, n


FAM = {"D1": world_d1, "D2": world_d2, "D4": world_d4, "STOP": world_stop}


def starts(g, rng, tok):
    place, value, line = g["place"], g["value"], g["line"]
    out = []
    for s in range(g["n"]):
        if value[s] != tok:
            continue
        pin1 = think_slot(s, g["slots_at"], place, value, line, rng)
        if pin1 is None:
            continue
        pin2, _, R = hop(value[pin1], place[s], g["by_key"], g["slots_at"], value, rng)
        if pin2 is None:
            continue
        pin3, _, T = hop(value[pin2], R, g["by_key"], g["slots_at"], value, rng)
        if pin3 is None:
            continue
        cands = next_cands(value[pin3], T, g["by_key"])
        if len(cands) < 2:
            continue
        visited = {place[s], R, T}
        used_k = {value[pin1], value[pin2], value[pin3]}
        order = list(g["place_ord"][place[s]])
        out.append((cands, visited, used_k, order))
    return out


def loop(g, cands, visited, used_k, order, rng, pick, target):
    extra = set()
    hops = 0
    while len(cands) > 1:
        opts = options(cands, order, g["by_key"], visited, used_k,
                       g["slots_at"], g["value"], rng)
        rec = pick(opts, cands, extra, g["place_keys"], g["value"])
        if rec is None:
            return hops, False, True
        k, H, pinH, vis1, used1 = rec
        extra = extra | {g["value"][pinH]}
        cands = filter_cands(cands, extra, g["place_keys"])
        visited, used_k = vis1, used1
        order = list(order) + [g["value"][pinH]]
        hops += 1
        if hops > 8:
            return hops, False, True
    if len(cands) != 1:
        return hops, False, True
    pin = think_place(list(g["slots_at"][next(iter(cands))]), g["value"], rng)
    ok = pin is not None and g["value"][pin] == target
    return hops, ok, False


def train_mixed(rng, n_each):
    win, tot = defaultdict(int), defaultdict(int)
    for name, fn in FAM.items():
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

                    hops, ok, _ = loop(g, c1, vis1, used1, list(order) + [g["value"][pinH]],
                                       rng, teacher_pick, names["CRISP"])
                    if d1 == 1:
                        pin = think_place(list(g["slots_at"][next(iter(c1))]), g["value"], rng)
                        ok = pin is not None and g["value"][pin] == names["CRISP"]
                    tot[(n_hit, d1)] += 1
                    win[(n_hit, d1)] += int(ok)
    return {s: win[s] / tot[s] for s in tot}


def eval_fam(rng, fn, n, rate, kind):
    n_ok = n_ref = n_ep = hops_sum = hops_g = 0
    n_g = 0
    for _ in range(n):
        lines, names = fn(rng)
        g = build(lines)
        if g is None:
            continue
        for cands, visited, used_k, order in starts(g, rng, names["APPLES"]):
            n_ep += 1
            pick = lambda o, c, e, pk, v: learned_pick(o, c, e, pk, v, rate)
            hops, ok, ref = loop(g, cands, visited, used_k, order, rng, pick, names["CRISP"])
            hops_sum += hops
            n_ok += int(ok)
            n_ref += int(ref)
            gp = lambda o, c, e, pk, v: greedy_pick(o, c, e, pk, v)
            gh, gok, _ = loop(g, cands, set(visited), set(used_k), list(order), rng, gp, names["CRISP"])
            n_g += int(gok)
            hops_g += gh
    return dict(
        ep=n_ep,
        pin=n_ok / max(n_ep, 1),
        refuse=n_ref / max(n_ep, 1),
        greedy=n_g / max(n_ep, 1),
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
    reps = {k: eval_fam(rng, fn, args.test, rate, k) for k, fn in FAM.items()}
    void = any(reps[k]["ep"] < 5 for k in FAM)
    gate = ((not void)
            and (reps["D1"]["pin"] == 1.0)
            and (reps["D2"]["pin"] == 1.0)
            and (reps["D4"]["pin"] == 1.0)
            and (reps["STOP"]["refuse"] == 1.0) and (reps["STOP"]["pin"] == 0.0)
            and (reps["D1"]["mean_hops"] == 1.0)
            and (reps["D2"]["mean_hops"] == 2.0)
            and (reps["D4"]["mean_hops"] == 4.0)
            and (reps["D2"]["greedy_hops"] > 2.0)
            and (reps["D4"]["greedy_hops"] > 4.0))
    rec = dict(seed=args.seed, void=bool(void), gate=bool(gate),
               rate={f"{a},{b}": round(v, 3) for (a, b), v in rate.items()},
               **{k: reps[k] for k in FAM})
    for k in FAM:
        r = reps[k]
        print(f"{k:4} ep {r['ep']:2} pin {r['pin']:.2f} refuse {r['refuse']:.2f} "
              f"hops {r['mean_hops']:.1f} greedy_hops {r['greedy_hops']:.1f}")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("\nVOID: a family had <5 test eps.")
    elif gate:
        print("\nGO DEPTH: same loop does 1, 2, 4 and STOP. Greedy extra hops on 2 and 4.")
    else:
        print("\nSTOP: a depth family failed, or greedy did not take extra hops.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
