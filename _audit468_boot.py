"""468 TRACK D BOOT-LOOP Phase A. One frozen tape. Q[H] only."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from _audit440_compose import think_place
from _audit449_adv import build
from _audit451_learn import options
from _audit456b_geo import starts_flat
from _audit458_keycut import filter_keys
from _audit452_xfer import _lex, _pad

OUT = Path("results/_stage468_boot.json")


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


def start_pack(g, rng, names):
    for pack in starts_flat(g, rng, names["FRESH"]):
        return pack
    return None


def eps_pick(opts, table, rng, eps):
    if not opts:
        return None
    if rng.random() < eps:
        return rng.choice(opts)
    best, br = None, -10 ** 9
    for rec in opts:
        r = table.get(rec[1], 0.0)
        if r > br:
            br, best = r, rec
    if br <= 0:
        return None
    return best


def rollout(g, cands, visited, used_k, order, rng, table, eps, names):
    hops = 0
    traj = []
    h1_live = None
    h2_ok = None
    br, mark = names["xx"], names["n4"]
    target = names["CRISP"]
    while len(cands) > 1:
        opts = options(cands, order, g["by_key"], visited, used_k,
                       g["slots_at"], g["value"], rng)
        rec = eps_pick(opts, table, rng, eps)
        if rec is None:
            return hops, False, True, traj, h1_live, h2_ok
        k, H, pinH, vis1, used1 = rec
        hole = g["value"][pinH]
        live = br in g["place_keys"].get(H, ())
        if hops == 0:
            h1_live = live
        elif hops == 1 and h1_live:
            h2_ok = mark in g["place_keys"].get(H, ())
        nxt = filter_keys(cands, H, g["place_keys"], hole)
        cands = nxt
        visited, used_k = vis1, used1
        order = list(order) + [x for x in g["place_keys"][H] if x != hole]
        hops += 1
        traj.append(H)
        if hops > 8:
            return hops, False, True, traj, h1_live, h2_ok
    if len(cands) != 1:
        return hops, False, True, traj, h1_live, h2_ok
    pin = think_place(list(g["slots_at"][next(iter(cands))]), g["value"], rng)
    ok = pin is not None and g["value"][pin] == target
    return hops, ok, False, traj, h1_live, h2_ok


def returns(traj, ok, ref, hop_bonus=0.05, hop_cost=0.02):
    term = 1.0 if ok else (-1.0 if not ref else 0.0)
    out = []
    G = term
    for H in reversed(traj):
        G = G - hop_cost + hop_bonus
        out.append((H, G))
    out.reverse()
    return out


def eval_n(g, names, rng, table, n):
    n_ep = n_ok = n_ref = n_h1 = n_live = n_cond = n_h2 = hops_sum = 0
    pack0 = start_pack(g, random.Random(0), names)
    if pack0 is None:
        return dict(ep=0, pin=0, refuse=1, p_live=0, p_h2_g_h1=0, mean_hops=0)
    for i in range(n):
        cands, visited, used_k, order = pack0
        cands, visited, used_k, order = set(cands), set(visited), set(used_k), list(order)
        n_ep += 1
        hops, ok, ref, traj, h1_live, h2_ok = rollout(
            g, set(cands), set(visited), set(used_k), list(order),
            rng, table, 0.0, names)
        hops_sum += hops
        n_ok += int(ok)
        n_ref += int(ref)
        if h1_live is not None:
            n_h1 += 1
            n_live += int(h1_live)
            if h1_live:
                n_cond += 1
                n_h2 += int(bool(h2_ok))
    return dict(
        ep=n_ep,
        pin=n_ok / max(n_ep, 1),
        refuse=n_ref / max(n_ep, 1),
        mean_hops=hops_sum / max(n_ep, 1),
        p_live=n_live / max(n_h1, 1),
        p_h2_g_h1=n_h2 / max(n_cond, 1),
        n_h1=n_h1,
        n_cond=n_cond,
    )


def train_tape(g, names, rng, n_ep, update, shuffle):
    table = {}
    tot, win = defaultdict(int), defaultdict(float)
    pack0 = start_pack(g, random.Random(0), names)
    if pack0 is None:
        return table
    for i in range(n_ep):
        eps = max(0.05, 0.5 * (1.0 - i / n_ep)) if update else 0.0
        cands, visited, used_k, order = pack0
        cands, visited, used_k, order = set(cands), set(visited), set(used_k), list(order)
        hops, ok, ref, traj, _, _ = rollout(
            g, set(cands), set(visited), set(used_k), list(order),
            rng, table, eps, names)
        if shuffle:
            ok = rng.random() < 0.5
            ref = not ok
        if update:
            for H, G in returns(traj, ok, ref):
                tot[H] += 1
                win[H] += G
                table[H] = win[H] / tot[H]
    return table


def n_opts(g, names):
    pack = start_pack(g, random.Random(0), names)
    if pack is None:
        return 0
    cands, visited, used_k, order = pack
    opts = options(cands, order, g["by_key"], visited, used_k,
                   g["slots_at"], g["value"], random.Random(0))
    return len(opts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--test", type=int, default=40)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    lines, names = world(random.Random(args.seed))
    g = build(lines)
    n0 = n_opts(g, names)
    void = g is None or n0 < 2
    t0 = eval_n(g, names, random.Random(1), {}, args.test)
    learn = train_tape(g, names, random.Random(args.seed), args.episodes, True, False)
    frozen = train_tape(g, names, random.Random(args.seed + 1), args.episodes, False, False)
    shuf = train_tape(g, names, random.Random(args.seed + 2), args.episodes, True, True)
    t1 = eval_n(g, names, random.Random(9), learn, args.test)
    t1f = eval_n(g, names, random.Random(9), frozen, args.test)
    t1s = eval_n(g, names, random.Random(9), shuf, args.test)
    lines2, names2 = world(random.Random(args.seed + 99))
    g2 = build(lines2)
    held = eval_n(g2, names2, random.Random(9), learn, args.test)
    gate = ((not void)
            and (t0["pin"] == 0.0)
            and (t1["p_live"] > 0.8) and (t1["p_h2_g_h1"] > 0.8)
            and (t1f["p_live"] < 0.5)
            and (t1s["p_live"] < t1["p_live"] - 0.3))
    rec = dict(seed=args.seed, void=bool(void), gate=bool(gate), n_opts=n0,
               t0=t0, t1=t1, frozen=t1f, shuffle=t1s, held=held,
               n_keys=len(learn))
    print(f"tape opts {n0}  keys {len(learn)}")
    print(f"t0     pin {t0['pin']:.2f} p_live {t0['p_live']:.2f} p_h2|h1 {t0['p_h2_g_h1']:.2f}")
    print(f"learn  pin {t1['pin']:.2f} p_live {t1['p_live']:.2f} p_h2|h1 {t1['p_h2_g_h1']:.2f}")
    print(f"frozen p_live {t1f['p_live']:.2f}  shuffle {t1s['p_live']:.2f}  held {held['p_live']:.2f}")
    print(f"VOID {void}  GATE_D {gate}")
    if void:
        print("\nVOID: start has <2 actions.")
    elif gate:
        print("\nGO BOOT: Q[H] on one tape; P(h2|h1) grew; shuffle/frozen fail. Held-out not gated.")
    else:
        print("\nSTOP D: place-Q did not credit the path, or shuffle also grew.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
