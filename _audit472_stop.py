"""472 standalone (no 470/471 files). Dead hop → refuse. Q[H] vs Q[trace]."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from _audit440_compose import think_place
from _audit449_adv import build
from _audit451_learn import options
from _audit452_xfer import _lex, _pad
from _audit456b_geo import starts_flat
from _audit458_keycut import filter_keys

OUT = Path("results/_stage472_stop.json")
DELTA = 0.15


def world(rng):
    n = _lex(rng)
    ctx = [n["red"], n["cat"], n["sat"], n["on"], n["the"], n["mat"]]
    rng.shuffle(ctx)
    q = [f"{ctx[0]} {ctx[1]} {ctx[2]} {n['FRESH']} {ctx[3]} {ctx[4]} {ctx[5]}" + _pad(i) for i in range(3)]
    q.append(f"{ctx[0]} {ctx[1]} {ctx[2]} {n['STALE']} {ctx[3]} {ctx[4]} {ctx[5]}" + _pad(3))
    mark, br = n["n4"], n["xx"]
    c1 = [f"{n['aa']} {n['bb']} {n['cc']} {n['CRISP']} {n['FRESH']} {mark} {n['u1']}" + _pad(50 + i) for i in range(3)]
    c1.append(f"{n['aa']} {n['bb']} {n['cc']} {n['SOFT']} {n['FRESH']} {mark} {n['u1']}" + _pad(53))
    c2 = [f"{n['dd']} {n['ee']} {n['ff']} {n['RIPE']} {n['FRESH']} {n['KEYA']} {n['u2']}" + _pad(60 + i) for i in range(3)]
    c2.append(f"{n['dd']} {n['ee']} {n['ff']} {n['RAW']} {n['FRESH']} {n['KEYA']} {n['u2']}" + _pad(63))
    c3 = [f"{n['gg']} {n['hh']} {n['ii']} {n['DRY']} {n['FRESH']} {n['u3']} {n['u4']}" + _pad(70 + i) for i in range(3)]
    c3.append(f"{n['gg']} {n['hh']} {n['ii']} {n['MUSH']} {n['FRESH']} {n['u3']} {n['u4']}" + _pad(73))
    hA = [f"{n['clerk']} {n['wrote']} {n['KEYA']} {n['sat']} {n['board']} {n['now']}" + _pad(90 + i) for i in range(3)]
    hA.append(f"{n['clerk']} {n['wrote']} {n['FOG']} {n['sat']} {n['board']} {n['now']}" + _pad(93))
    hD = [f"{n['kids']} {n['like']} {n['FOG']} {n['cat']} {n['today']} {n['yes']}" + _pad(100 + i) for i in range(3)]
    hD.append(f"{n['kids']} {n['like']} {n['STALE']} {n['cat']} {n['today']} {n['yes']}" + _pad(103))
    hS = [f"{n['desk']} {n['shows']} {n['FOG']} {n['red']} {br} {n['here']}" + _pad(110 + i) for i in range(3)]
    hS.append(f"{n['desk']} {n['shows']} {n['STALE']} {n['red']} {br} {n['here']}" + _pad(113))
    hT = [f"{n['vault']} {n['keeps']} {n['FOG']} {br} {mark} {n['here']}" + _pad(120 + i) for i in range(3)]
    hT.append(f"{n['vault']} {n['keeps']} {n['STALE']} {br} {mark} {n['here']}" + _pad(123))
    return q + c1 + c2 + c3 + hA + hD + hS + hT, n


def pack0(g, names):
    for p in starts_flat(g, random.Random(0), names["FRESH"]):
        return p
    return None


def n_after(cands, rec, g):
    H, pinH = rec[1], rec[2]
    return len(filter_keys(cands, H, g["place_keys"], g["value"][pinH]))


def pick_H(opts, table):
    if not opts:
        return None
    best, br = None, -1e9
    for r in opts:
        v = table.get(r[1], 0.0)
        if v > br:
            br, best = v, r
    return best if br > 0 else None


def pick_tr(opts, cands, g, S, table):
    if not opts:
        return None
    best, br = None, -1e9
    for r in opts:
        a = n_after(cands, r, g)
        v = table.get((S, a), 0.0)
        if v > br:
            br, best = v, r
    return best if br > 0 else None


def credit(traj, ok):
    G = 1.0 if ok else 0.0
    out = []
    for key in reversed(traj):
        G = G - 0.05
        out.append((key, G))
    out.reverse()
    return out


def opened(H, hole, g, visited, cands):
    vis = set(visited)
    n = 0
    for k in g["place_keys"].get(H, ()):
        if k == hole:
            continue
        for P in g["by_key"].get(k, ()):
            if P in vis or P in cands or P == H:
                continue
            n += 1
    return n


def rollout(g, names, rng, table, eps, kind):
    pack = pack0(g, names)
    if pack is None:
        return 0, False, True, [], []
    cands, visited, used_k, order = pack
    cands, visited, used_k, order = set(cands), set(visited), set(used_k), list(order)
    hops = last_valid = 0
    tH, tT = [], []
    while len(cands) > 1:
        opts = options(cands, order, g["by_key"], visited, used_k,
                       g["slots_at"], g["value"], rng)
        S = (len(cands), last_valid, hops)
        rec = None
        if opts and rng.random() < eps:
            rec = rng.choice(opts)
        elif kind == "H":
            rec = pick_H(opts, table)
        elif kind == "T":
            rec = pick_tr(opts, cands, g, S, table)
        elif kind == "R":
            rec = rng.choice(opts) if opts else None
        if rec is None:
            return hops, False, True, tH, tT
        a = n_after(cands, rec, g)
        H, pinH = rec[1], rec[2]
        hole = g["value"][pinH]
        nxt = filter_keys(cands, H, g["place_keys"], hole)
        shrunk = len(nxt) < len(cands)
        op = opened(H, hole, g, rec[3], cands)
        tH.append(H)
        tT.append((S, a))
        cands = nxt
        visited, used_k = rec[3], rec[4]
        order = list(order) + [x for x in g["place_keys"][H] if x != hole]
        hops += 1
        last_valid = int(shrunk or op > 0)
        if not shrunk and op == 0:
            return hops, False, True, tH, tT
        if hops > 8:
            return hops, False, True, tH, tT
    if len(cands) != 1:
        return hops, False, True, tH, tT
    pin = think_place(list(g["slots_at"][next(iter(cands))]), g["value"], rng)
    ok = pin is not None and g["value"][pin] == names["CRISP"]
    return hops, ok, False, tH, tT


def train(g, names, rng, n_ep, kind):
    table = {}
    tot, win = defaultdict(int), defaultdict(float)
    for i in range(n_ep):
        eps = max(0.08, 0.5 * (1.0 - i / n_ep))
        hops, ok, ref, tH, tT = rollout(g, names, rng, table, eps, kind)
        traj = tH if kind == "H" else tT
        for key, G in credit(traj, ok):
            tot[key] += 1
            win[key] += G
            table[key] = win[key] / tot[key]
    return table


def eval_kind(g, names, rng, table, kind, n):
    n_ok = n_ep = hops_sum = 0
    for i in range(n):
        hops, ok, ref, tH, tT = rollout(g, names, rng, table, 0.0, kind)
        n_ep += 1
        n_ok += int(ok)
        hops_sum += hops
    return dict(ep=n_ep, pin=n_ok / max(n_ep, 1),
                mean_hops=hops_sum / max(n_ep, 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--test", type=int, default=30)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    la, na = world(random.Random(args.seed))
    lb, nb = world(random.Random(args.seed + 99))
    ga, gb = build(la), build(lb)
    pack = pack0(ga, na)
    nopt = 0
    if pack:
        c, vis, u, o = pack
        nopt = len(options(c, o, ga["by_key"], vis, u,
                           ga["slots_at"], ga["value"], random.Random(0)))
    qh = train(ga, na, random.Random(args.seed), args.episodes, "H")
    qt = train(ga, na, random.Random(args.seed + 1), args.episodes, "T")
    hA = eval_kind(ga, na, random.Random(3), qh, "H", args.test)
    hB = eval_kind(gb, nb, random.Random(3), qh, "H", args.test)
    tA = eval_kind(ga, na, random.Random(4), qt, "T", args.test)
    tB = eval_kind(gb, nb, random.Random(4), qt, "T", args.test)
    rA = eval_kind(ga, na, random.Random(5), {}, "R", args.test)
    rB = eval_kind(gb, nb, random.Random(5), {}, "R", args.test)
    void = (tA["ep"] < 5) or (rB["pin"] >= 0.9) or (rB["pin"] <= 0.05)
    gate = ((not void)
            and (hB["pin"] == 0.0)
            and (tA["pin"] > rA["pin"])
            and (tB["pin"] > rB["pin"] + DELTA))
    rec = dict(seed=args.seed, void=bool(void), gate=bool(gate),
               qh_a=hA, qh_b=hB, tr_a=tA, tr_b=tB, rnd_a=rA, rnd_b=rB,
               n_opts=nopt, n_qh=len(qh), n_tr=len(qt),
               crisp_eq=na["CRISP"] == nb["CRISP"])
    print("opts", nopt)
    print("A  QH", round(hA["pin"], 2), "trace", round(tA["pin"], 2),
          "rnd", round(rA["pin"], 2))
    print("B  QH", round(hB["pin"], 2), "trace", round(tB["pin"], 2),
          "rnd", round(rB["pin"], 2))
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("\nVOID: random still not in (0.05, 0.90).")
    elif gate:
        print("\nGO STOP: dead hop refuses; trace beats random on B; Q[H] does not ride.")
    else:
        print("\nSTOP: env stop exists, but trace still not above random on B.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
