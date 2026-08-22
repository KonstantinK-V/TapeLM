"""465 TRACK R: new world. Follow = non-hole key of H, not value-as-address.

loop: order += keys(H) minus hole. 464 soon_keys should work here.
FAM rebuilt: hop2 via BRIDGE token on H, hole is FOG.

GATE  soon_keys: 455 hops
      no-soon BOTH hops > 2
VOID  ep < 5

    python _check465_bridge.py
    python _audit465_bridge.py --seed 1337
"""
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
from _audit463_trap import trap_of, teacher_trap
from _audit464_soonkeys import soon_keys

OUT = Path("results/_stage465_bridge.json")


def _q(n, rng):
    ctx = [n["red"], n["cat"], n["sat"], n["on"], n["the"], n["mat"]]
    rng.shuffle(ctx)
    q = [f"{ctx[0]} {ctx[1]} {ctx[2]} {n['FRESH']} {ctx[3]} {ctx[4]} {ctx[5]}" + _pad(i) for i in range(3)]
    q.append(f"{ctx[0]} {ctx[1]} {ctx[2]} {n['STALE']} {ctx[3]} {ctx[4]} {ctx[5]}" + _pad(3))
    return q, n["red"], n["cat"], n["sat"]


def _cands4(n, mark):
    c1 = [f"{n['aa']} {n['bb']} {n['cc']} {n['CRISP']} {n['FRESH']} {mark} {n['u1']}" + _pad(50 + i) for i in range(3)]
    c1.append(f"{n['aa']} {n['bb']} {n['cc']} {n['SOFT']} {n['FRESH']} {mark} {n['u1']}" + _pad(53))
    c2 = [f"{n['dd']} {n['ee']} {n['ff']} {n['RIPE']} {n['FRESH']} {n['KEYA']} {n['u2']}" + _pad(60 + i) for i in range(3)]
    c2.append(f"{n['dd']} {n['ee']} {n['ff']} {n['RAW']} {n['FRESH']} {n['KEYA']} {n['u2']}" + _pad(63))
    c3 = [f"{n['gg']} {n['hh']} {n['ii']} {n['DRY']} {n['FRESH']} {n['KEYA']} {n['u3']}" + _pad(70 + i) for i in range(3)]
    c3.append(f"{n['gg']} {n['hh']} {n['ii']} {n['MUSH']} {n['FRESH']} {n['KEYA']} {n['u3']}" + _pad(73))
    c4 = [f"{n['jj']} {n['kk']} {n['ll']} {n['WET']} {n['FRESH']} {n['u4']} {n['n3']}" + _pad(80 + i) for i in range(3)]
    c4.append(f"{n['jj']} {n['kk']} {n['ll']} {n['DAMP']} {n['FRESH']} {n['u4']} {n['n3']}" + _pad(83))
    return c1, c2, c3, c4


def world_d1(rng):
    n = _lex(rng)
    q, red, cat, sat = _q(n, rng)
    mark = n["n4"]
    c1 = [f"{n['aa']} {n['bb']} {n['cc']} {n['CRISP']} {n['FRESH']} {mark} {n['u1']}" + _pad(50 + i) for i in range(3)]
    c1.append(f"{n['aa']} {n['bb']} {n['cc']} {n['SOFT']} {n['FRESH']} {mark} {n['u1']}" + _pad(53))
    c2 = [f"{n['dd']} {n['ee']} {n['ff']} {n['RIPE']} {n['FRESH']} {n['u2']} {n['u3']}" + _pad(60 + i) for i in range(3)]
    c2.append(f"{n['dd']} {n['ee']} {n['ff']} {n['RAW']} {n['FRESH']} {n['u2']} {n['u3']}" + _pad(63))
    h = [f"{n['vault']} {n['keeps']} {n['FOG']} {red} {mark} {n['here']}" + _pad(110 + i) for i in range(3)]
    h.append(f"{n['vault']} {n['keeps']} {n['STALE']} {red} {mark} {n['here']}" + _pad(113))
    return q + c1 + c2 + h, n


def world_d2(rng):
    n = _lex(rng)
    q, red, cat, sat = _q(n, rng)
    mark, br = n["n4"], n["xx"]
    c1, c2, c3, c4 = _cands4(n, mark)
    hA = [f"{n['clerk']} {n['wrote']} {n['KEYA']} {sat} {n['board']} {n['now']}" + _pad(90 + i) for i in range(3)]
    hA.append(f"{n['clerk']} {n['wrote']} {n['FOG']} {sat} {n['board']} {n['now']}" + _pad(93))
    hS = [f"{n['desk']} {n['shows']} {n['FOG']} {red} {br} {n['here']}" + _pad(100 + i) for i in range(3)]
    hS.append(f"{n['desk']} {n['shows']} {n['STALE']} {red} {br} {n['here']}" + _pad(103))
    hT = [f"{n['vault']} {n['keeps']} {n['FOG']} {br} {mark} {n['here']}" + _pad(110 + i) for i in range(3)]
    hT.append(f"{n['vault']} {n['keeps']} {n['STALE']} {br} {mark} {n['here']}" + _pad(113))
    return q + c1 + c2 + c3 + c4 + hA + hS + hT, n


def world_d4(rng):
    n = _lex(rng)
    q, red, cat, sat = _q(n, rng)
    mark = n["n4"]
    b1, b2, b3 = n["xx"], n["yy"], n["n1"]
    c1, c2, c3, c4 = _cands4(n, mark)
    hA = [f"{n['clerk']} {n['wrote']} {n['KEYA']} {sat} {n['board']} {n['now']}" + _pad(90 + i) for i in range(3)]
    hA.append(f"{n['clerk']} {n['wrote']} {n['FOG']} {sat} {n['board']} {n['now']}" + _pad(93))
    h1 = [f"{n['desk']} {n['shows']} {n['FOG']} {red} {b1} {n['here']}" + _pad(100 + i) for i in range(3)]
    h1.append(f"{n['desk']} {n['shows']} {n['STALE']} {red} {b1} {n['here']}" + _pad(103))
    h2 = [f"{n['kids']} {n['like']} {n['FOG']} {b1} {b2} {n['today']}" + _pad(120 + i) for i in range(3)]
    h2.append(f"{n['kids']} {n['like']} {n['STALE']} {b1} {b2} {n['today']}" + _pad(123))
    h3 = [f"{n['blue']} {n['dog']} {n['FOG']} {b2} {b3} {n['yes']}" + _pad(130 + i) for i in range(3)]
    h3.append(f"{n['blue']} {n['dog']} {n['STALE']} {b2} {b3} {n['yes']}" + _pad(133))
    h4 = [f"{n['vault']} {n['keeps']} {n['FOG']} {b3} {mark} {n['here']}" + _pad(140 + i) for i in range(3)]
    h4.append(f"{n['vault']} {n['keeps']} {n['STALE']} {b3} {mark} {n['here']}" + _pad(143))
    return q + c1 + c2 + c3 + c4 + hA + h1 + h2 + h3 + h4, n


def world_both(rng):
    n = _lex(rng)
    q, red, cat, sat = _q(n, rng)
    mark, br = n["n4"], n["xx"]
    b1, b2, b3 = n["yy"], n["n1"], n["n2"]
    c1, c2, c3, c4 = _cands4(n, mark)
    hA = [f"{n['clerk']} {n['wrote']} {n['KEYA']} {sat} {n['board']} {n['now']}" + _pad(90 + i) for i in range(3)]
    hA.append(f"{n['clerk']} {n['wrote']} {n['FOG']} {sat} {n['board']} {n['now']}" + _pad(93))
    hS = [f"{n['desk']} {n['shows']} {n['FOG']} {red} {br} {n['here']}" + _pad(100 + i) for i in range(3)]
    hS.append(f"{n['desk']} {n['shows']} {n['STALE']} {red} {br} {n['here']}" + _pad(103))
    hT = [f"{n['vault']} {n['keeps']} {n['FOG']} {br} {mark} {n['here']}" + _pad(110 + i) for i in range(3)]
    hT.append(f"{n['vault']} {n['keeps']} {n['STALE']} {br} {mark} {n['here']}" + _pad(113))
    h1 = [f"{n['kids']} {n['like']} {n['FOG']} {cat} {b1} {n['today']}" + _pad(120 + i) for i in range(3)]
    h1.append(f"{n['kids']} {n['like']} {n['STALE']} {cat} {b1} {n['today']}" + _pad(123))
    h2 = [f"{n['blue']} {n['dog']} {n['FOG']} {b1} {b2} {n['yes']}" + _pad(130 + i) for i in range(3)]
    h2.append(f"{n['blue']} {n['dog']} {n['STALE']} {b1} {b2} {n['yes']}" + _pad(133))
    h3 = [f"{n['barns']} {n['store']} {n['FOG']} {b2} {b3} {n['now']}" + _pad(140 + i) for i in range(3)]
    h3.append(f"{n['barns']} {n['store']} {n['STALE']} {b2} {b3} {n['now']}" + _pad(143))
    h4 = [f"{n['sun']} {n['lay']} {n['FOG']} {b3} {mark} {n['fruit']}" + _pad(150 + i) for i in range(3)]
    h4.append(f"{n['sun']} {n['lay']} {n['STALE']} {b3} {mark} {n['fruit']}" + _pad(153))
    return q + c1 + c2 + c3 + c4 + hA + hS + hT + h1 + h2 + h3 + h4, n


def world_stop(rng):
    n = _lex(rng)
    q, red, cat, sat = _q(n, rng)
    mark = n["n4"]
    c1, c2, c3, c4 = _cands4(n, mark)
    hA = [f"{n['clerk']} {n['wrote']} {n['KEYA']} {red} {n['board']} {n['now']}" + _pad(90 + i) for i in range(3)]
    hA.append(f"{n['clerk']} {n['wrote']} {n['FOG']} {red} {n['board']} {n['now']}" + _pad(93))
    return q + c1 + c2 + c3 + c4 + hA, n


FAM = {"D1": world_d1, "D2": world_d2, "BOTH": world_both, "D4": world_d4, "STOP": world_stop}


def loop_bridge(g, cands, visited, used_k, order, rng, pick, target):
    hops = 0
    while len(cands) > 1:
        opts = options(cands, order, g["by_key"], visited, used_k,
                       g["slots_at"], g["value"], rng)
        rec = pick(opts, cands, g)
        if rec is None:
            return hops, False, True
        k, H, pinH, vis1, used1 = rec
        hole = g["value"][pinH]
        cands = filter_keys(cands, H, g["place_keys"], hole)
        visited, used_k = vis1, used1
        order = list(order) + [x for x in g["place_keys"][H] if x != hole]
        hops += 1
        if hops > 8:
            return hops, False, True
    if len(cands) != 1:
        return hops, False, True
    pin = think_place(list(g["slots_at"][next(iter(cands))]), g["value"], rng)
    ok = pin is not None and g["value"][pin] == target
    return hops, ok, False


def sig(cands, H, pinH, g, visited, use_soon):
    hole = g["value"][pinH]
    d1 = len(filter_keys(cands, H, g["place_keys"], hole))
    tr = trap_of(cands, hole, g["place_keys"])
    s = soon_keys(H, hole, g["place_keys"], g["by_key"], visited, cands) if use_soon else 0
    return (d1, tr, s)


def train(rng, n_each, use_soon):
    win, tot = defaultdict(float), defaultdict(int)
    for fn in FAM.values():
        for _ in range(n_each):
            lines, names = fn(rng)
            g = build(lines)
            if g is None:
                continue
            for cands, visited, used_k, order in starts_flat(g, rng, names["FRESH"]):
                opts = options(cands, order, g["by_key"], visited, used_k,
                               g["slots_at"], g["value"], rng)
                for rec in opts:
                    k, H, pinH, vis1, used1 = rec
                    c1 = filter_keys(cands, H, g["place_keys"], g["value"][pinH])
                    key = sig(cands, H, pinH, g, vis1, use_soon)
                    hops, ok, _ = loop_bridge(
                        g, c1, vis1, used1,
                        list(order) + [x for x in g["place_keys"][H] if x != g["value"][pinH]],
                        rng, lambda o, c, gg: teacher_trap(o, c, gg), names["CRISP"])
                    if key[0] == 1:
                        hops, ok = 0, True
                    R = (1.0 if ok else 0.0) - 0.05 * (hops + 1)
                    tot[key] += 1
                    win[key] += R
    return {k: win[k] / tot[k] for k in tot}


def make_pick(table, use_soon, g):
    def pick(opts, cands, gg):
        best, br = None, -10 ** 9
        for rec in opts:
            H, pinH, vis1 = rec[1], rec[2], rec[3]
            key = sig(cands, H, pinH, gg, vis1, use_soon)
            r = table.get(key, -1.0)
            if r > br:
                br, best = r, rec
        return best if br > -1.0 else None
    return pick


def eval_tab(rng, fn, n, table, use_soon):
    n_ok = n_ref = n_ep = hops_sum = 0
    for _ in range(n):
        lines, names = fn(rng)
        g = build(lines)
        if g is None:
            continue
        for cands, visited, used_k, order in starts_flat(g, rng, names["FRESH"]):
            n_ep += 1
            pick = make_pick(table, use_soon, g)
            hops, ok, ref = loop_bridge(g, set(cands), set(visited), set(used_k),
                                        list(order), rng, pick, names["CRISP"])
            hops_sum += hops
            n_ok += int(ok)
            n_ref += int(ref)
    return dict(
        ep=n_ep,
        pin=n_ok / max(n_ep, 1),
        refuse=n_ref / max(n_ep, 1),
        mean_hops=hops_sum / max(n_ep, 1),
    )


def match_455(yes):
    return ((yes["D1"]["pin"] == 1.0) and (yes["D1"]["mean_hops"] == 1.0)
            and (yes["D2"]["pin"] == 1.0) and (yes["D2"]["mean_hops"] == 2.0)
            and (yes["BOTH"]["pin"] == 1.0) and (yes["BOTH"]["mean_hops"] == 2.0)
            and (yes["D4"]["pin"] == 1.0) and (yes["D4"]["mean_hops"] == 4.0)
            and (yes["STOP"]["refuse"] == 1.0) and (yes["STOP"]["pin"] == 0.0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--train", type=int, default=8)
    ap.add_argument("--test", type=int, default=6)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    rng = random.Random(args.seed)
    tab_s = train(rng, args.train, True)
    tab_n = train(rng, args.train, False)
    yes = {k: eval_tab(rng, fn, args.test, tab_s, True) for k, fn in FAM.items()}
    no = {k: eval_tab(rng, fn, args.test, tab_n, False) for k, fn in FAM.items()}
    void = any(yes[k]["ep"] < 5 for k in FAM)
    gate = (not void) and match_455(yes) and (no["BOTH"]["mean_hops"] > 2.0)
    rec = dict(seed=args.seed, void=bool(void), gate=bool(gate),
               with_soon=yes, no_soon=no)
    print("bridge world: order += keys(H) minus hole")
    print("with soon_keys")
    for k in FAM:
        r = yes[k]
        print(f"  {k:4} pin {r['pin']:.2f} ref {r['refuse']:.2f} hops {r['mean_hops']:.1f}")
    print("no soon  BOTH hops", round(no["BOTH"]["mean_hops"], 2))
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("\nVOID: empty.")
    elif gate:
        print("\nGO BRIDGE: follow is a key of H, not the hole. 464 was the old world.")
    else:
        print("\nSTOP R: even on a bridge world soon_keys did not select the short path.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
