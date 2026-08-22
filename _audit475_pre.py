"""475: Q[S, a]. S = last (shrunk, opened). a = pre-action (n_keys, n_slots)."""
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

OUT = Path("results/_stage475_pre.json")
DELTA = 0.10
START = ("start",)


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


def observe(cands, rec, g):
    H, pinH = rec[1], rec[2]
    hole = g["value"][pinH]
    nxt = filter_keys(cands, H, g["place_keys"], hole)
    shrunk = int(len(nxt) < len(cands))
    op = int(opened(H, hole, g, rec[3], cands) > 0)
    return (shrunk, op), nxt, hole


def pre(rec, g):
    H, pinH = rec[1], rec[2]
    hole = g["value"][pinH]
    n_k = sum(1 for x in g["place_keys"].get(H, ()) if x != hole)
    n_s = len(g["slots_at"].get(H, ()))
    return (n_k, n_s)


def pick_H(opts, table):
    if not opts:
        return None
    best, br = None, -1e9
    for r in opts:
        v = table.get(r[1], 0.0)
        if v > br:
            br, best = v, r
    return best if br > 0 else None


def pick_pre(opts, S, table, g):
    if not opts:
        return None
    best, br = None, -1e9
    for r in opts:
        v = table.get((S, pre(r, g)), 0.0)
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


def rollout(g, names, rng, table, eps, kind):
    pack = pack0(g, names)
    if pack is None:
        return 0, False, True, [], []
    cands, visited, used_k, order = pack
    cands, visited, used_k, order = set(cands), set(visited), set(used_k), list(order)
    hops = 0
    S = START
    tH, tP = [], []
    while len(cands) > 1:
        opts = options(cands, order, g["by_key"], visited, used_k,
                       g["slots_at"], g["value"], rng)
        rec = None
        if opts and rng.random() < eps:
            rec = rng.choice(opts)
        elif kind == "H":
            rec = pick_H(opts, table)
        elif kind == "P":
            rec = pick_pre(opts, S, table, g)
        elif kind == "R":
            rec = rng.choice(opts) if opts else None
        if rec is None:
            return hops, False, True, tH, tP
        k, nxt, hole = observe(cands, rec, g)
        H = rec[1]
        tH.append(H)
        tP.append((S, pre(rec, g)))
        cands = nxt
        visited, used_k = rec[3], rec[4]
        order = list(order) + [x for x in g["place_keys"][H] if x != hole]
        hops += 1
        S = k
        shrunk, op = k
        if not shrunk and op == 0:
            return hops, False, True, tH, tP
        if hops > 8:
            return hops, False, True, tH, tP
    if len(cands) != 1:
        return hops, False, True, tH, tP
    pin = think_place(list(g["slots_at"][next(iter(cands))]), g["value"], rng)
    ok = pin is not None and g["value"][pin] == names["CRISP"]
    return hops, ok, False, tH, tP


def train(g, names, rng, n_ep, kind):
    table = {}
    tot, win = defaultdict(int), defaultdict(float)
    for i in range(n_ep):
        eps = max(0.08, 0.5 * (1.0 - i / n_ep))
        hops, ok, ref, tH, tP = rollout(g, names, rng, table, eps, kind)
        traj = tH if kind == "H" else tP
        for key, G in credit(traj, ok):
            tot[key] += 1
            win[key] += G
            table[key] = win[key] / tot[key]
    return table


def eval_kind(g, names, rng, table, kind, n):
    n_ok = n_ep = hops_sum = 0
    for i in range(n):
        hops, ok, ref, tH, tP = rollout(g, names, rng, table, 0.0, kind)
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
    qh = train(ga, na, random.Random(args.seed), args.episodes, "H")
    qp = train(ga, na, random.Random(args.seed + 1), args.episodes, "P")
    hA = eval_kind(ga, na, random.Random(3), qh, "H", args.test)
    hB = eval_kind(gb, nb, random.Random(3), qh, "H", args.test)
    pA = eval_kind(ga, na, random.Random(4), qp, "P", args.test)
    pB = eval_kind(gb, nb, random.Random(4), qp, "P", args.test)
    rA = eval_kind(ga, na, random.Random(5), {}, "R", args.test)
    rB = eval_kind(gb, nb, random.Random(5), {}, "R", args.test)
    void = ((pA["ep"] < 5) or (rB["pin"] >= 0.9) or (rB["pin"] <= 0.05)
            or (pB["pin"] >= 0.90))
    gate = ((not void)
            and (hB["pin"] == 0.0)
            and (pA["pin"] > rA["pin"])
            and (pB["pin"] > rB["pin"] + DELTA)
            and (pB["pin"] < 0.90))
    rec = dict(seed=args.seed, void=bool(void), gate=bool(gate),
               qh_a=hA, qh_b=hB, pre_a=pA, pre_b=pB, rnd_a=rA, rnd_b=rB,
               n_qh=len(qh), n_pre=len(qp),
               crisp_eq=na["CRISP"] == nb["CRISP"])
    print("A  QH", round(hA["pin"], 2), "pre", round(pA["pin"], 2),
          "rnd", round(rA["pin"], 2))
    print("B  QH", round(hB["pin"], 2), "pre", round(pB["pin"], 2),
          "rnd", round(rB["pin"], 2))
    print(f"VOID {void}  GATE {gate}  n_pre {len(qp)}")
    if void:
        print("\nVOID: random degenerate, or pin>=0.90 (cls sneak).")
    elif gate:
        print("\nGO PRE: Q[S, n_keys/n_slots] transfers; Q[H] does not.")
    else:
        print("\nSTOP: pre-action (n_keys, n_slots) did not beat random on B.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
