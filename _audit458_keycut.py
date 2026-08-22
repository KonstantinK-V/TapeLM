"""458: hop cuts by KEYS of H, not by value[H]."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from _audit440_compose import think_place
from _audit449_adv import build, filter_cands
from _audit451_learn import options
from _audit452_xfer import _lex, _pad
from _audit454_cost import soon
from _audit456_policy import train_return, teacher_pick
from _audit456b_geo import starts_flat

OUT = Path("results/_stage458_keycut.json")


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
    mark = n["n4"]
    c1, c2, c3, c4 = _cands4(n, mark)
    hA = [f"{n['clerk']} {n['wrote']} {n['KEYA']} {sat} {n['board']} {n['now']}" + _pad(90 + i) for i in range(3)]
    hA.append(f"{n['clerk']} {n['wrote']} {n['FOG']} {sat} {n['board']} {n['now']}" + _pad(93))
    hS = [f"{n['desk']} {n['shows']} {n['UNLOCK']} {red} {n['label']} {n['here']}" + _pad(100 + i) for i in range(3)]
    hS.append(f"{n['desk']} {n['shows']} {n['FOG']} {red} {n['label']} {n['here']}" + _pad(103))
    hT = [f"{n['vault']} {n['keeps']} {n['FOG']} {n['UNLOCK']} {mark} {n['here']}" + _pad(110 + i) for i in range(3)]
    hT.append(f"{n['vault']} {n['keeps']} {n['STALE']} {n['UNLOCK']} {mark} {n['here']}" + _pad(113))
    return q + c1 + c2 + c3 + c4 + hA + hS + hT, n


def world_d4(rng):
    n = _lex(rng)
    q, red, cat, sat = _q(n, rng)
    mark = n["n4"]
    c1, c2, c3, c4 = _cands4(n, mark)
    u1, u2, u3 = n["UNLOCK"], n["n1"], n["n2"]
    hA = [f"{n['clerk']} {n['wrote']} {n['KEYA']} {sat} {n['board']} {n['now']}" + _pad(90 + i) for i in range(3)]
    hA.append(f"{n['clerk']} {n['wrote']} {n['FOG']} {sat} {n['board']} {n['now']}" + _pad(93))
    h1 = [f"{n['desk']} {n['shows']} {u1} {red} {n['label']} {n['here']}" + _pad(100 + i) for i in range(3)]
    h1.append(f"{n['desk']} {n['shows']} {n['FOG']} {red} {n['label']} {n['here']}" + _pad(103))
    h2 = [f"{n['xx']} {n['yy']} {u2} {u1} {n['safe']} {n['now']}" + _pad(120 + i) for i in range(3)]
    h2.append(f"{n['xx']} {n['yy']} {n['FOG']} {u1} {n['safe']} {n['now']}" + _pad(123))
    h3 = [f"{n['kids']} {n['like']} {u3} {u2} {n['today']} {n['yes']}" + _pad(130 + i) for i in range(3)]
    h3.append(f"{n['kids']} {n['like']} {n['FOG']} {u2} {n['today']} {n['yes']}" + _pad(133))
    h4 = [f"{n['vault']} {n['keeps']} {n['FOG']} {u3} {mark} {n['here']}" + _pad(140 + i) for i in range(3)]
    h4.append(f"{n['vault']} {n['keeps']} {n['STALE']} {u3} {mark} {n['here']}" + _pad(143))
    return q + c1 + c2 + c3 + c4 + hA + h1 + h2 + h3 + h4, n


def world_both(rng):
    n = _lex(rng)
    q, red, cat, sat = _q(n, rng)
    mark = n["n4"]
    c1, c2, c3, c4 = _cands4(n, mark)
    u1, u2, u3 = n["n1"], n["n2"], n["n3"]
    hA = [f"{n['clerk']} {n['wrote']} {n['KEYA']} {sat} {n['board']} {n['now']}" + _pad(90 + i) for i in range(3)]
    hA.append(f"{n['clerk']} {n['wrote']} {n['FOG']} {sat} {n['board']} {n['now']}" + _pad(93))
    hS = [f"{n['desk']} {n['shows']} {n['UNLOCK']} {red} {n['label']} {n['here']}" + _pad(100 + i) for i in range(3)]
    hS.append(f"{n['desk']} {n['shows']} {n['FOG']} {red} {n['label']} {n['here']}" + _pad(103))
    hT = [f"{n['vault']} {n['keeps']} {n['FOG']} {n['UNLOCK']} {mark} {n['here']}" + _pad(110 + i) for i in range(3)]
    hT.append(f"{n['vault']} {n['keeps']} {n['STALE']} {n['UNLOCK']} {mark} {n['here']}" + _pad(113))
    h1 = [f"{n['xx']} {n['yy']} {u1} {cat} {n['fruit']} {n['now']}" + _pad(120 + i) for i in range(3)]
    h1.append(f"{n['xx']} {n['yy']} {n['FOG']} {cat} {n['fruit']} {n['now']}" + _pad(123))
    h2 = [f"{n['blue']} {n['dog']} {u2} {u1} {n['store']} {n['yes']}" + _pad(130 + i) for i in range(3)]
    h2.append(f"{n['blue']} {n['dog']} {n['FOG']} {u1} {n['store']} {n['yes']}" + _pad(133))
    h3 = [f"{n['kids']} {n['like']} {u3} {u2} {n['today']} {n['here']}" + _pad(140 + i) for i in range(3)]
    h3.append(f"{n['kids']} {n['like']} {n['FOG']} {u2} {n['today']} {n['here']}" + _pad(143))
    h4 = [f"{n['barns']} {n['lay']} {n['FOG']} {u3} {mark} {n['sun']}" + _pad(150 + i) for i in range(3)]
    h4.append(f"{n['barns']} {n['lay']} {n['STALE']} {u3} {mark} {n['sun']}" + _pad(153))
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


def filter_keys(cands, H, place_keys, hole):
    keys = [k for k in place_keys.get(H, ()) if k != hole]
    keep = None
    for k in keys:
        hit = [p for p in cands if k in place_keys[p]]
        if len(hit) == 1:
            p = hit[0]
            if keep is None:
                keep = {p}
            else:
                keep.add(p)
    if keep is None:
        return set(cands)
    return keep


def loop_keys(g, cands, visited, used_k, order, rng, pick, target):
    hops = 0
    while len(cands) > 1:
        opts = options(cands, order, g["by_key"], visited, used_k,
                       g["slots_at"], g["value"], rng)
        rec = pick(opts, cands, g)
        if rec is None:
            return hops, False, True
        k, H, pinH, vis1, used1 = rec
        cands = filter_keys(cands, H, g["place_keys"], g["value"][pinH])
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


def teacher_keys(o, cands, g):
    for rec in o:
        H, pinH = rec[1], rec[2]
        c1 = filter_keys(cands, H, g["place_keys"], g["value"][pinH])
        if len(c1) == 1:
            return rec
    for rec in o:
        pinH, vis1 = rec[2], rec[3]
        nxt = g["by_key"].get(g["value"][pinH], ())
        if any(h not in vis1 for h in nxt):
            return rec
    return None


def sig(cands, H, pinH, g, order, visited, used_k, use_soon):
    hole = g["value"][pinH]
    d1 = len(filter_keys(cands, H, g["place_keys"], hole))
    n_hit = sum(1 for p in cands if hole in g["place_keys"][p])
    s = soon(hole, order, g["by_key"], visited, cands, used_k, g["place_keys"]) if use_soon else 0
    return n_hit, d1, s


def train_keys(rng, n_each, use_soon):
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
                    n_hit, d1, s = sig(cands, H, pinH, g, order, vis1, used1, use_soon)
                    hops, ok, _ = loop_keys(
                        g, c1, vis1, used1, list(order) + [g["value"][pinH]],
                        rng, lambda o, c, gg: teacher_keys(o, c, gg), names["CRISP"])
                    if d1 == 1:
                        hops, ok = 0, True
                    R = (1.0 if ok else 0.0) - 0.05 * (hops + 1)
                    tot[(n_hit, d1, s)] += 1
                    win[(n_hit, d1, s)] += R
    return {k: win[k] / tot[k] for k in tot}


def make_pick_keys(table, use_soon, order, g, visited, used_k):
    def pick(opts, cands, gg):
        best, br = None, -10 ** 9
        for rec in opts:
            H, pinH = rec[1], rec[2]
            vis1, used1 = rec[3], rec[4]
            n_hit, d1, s = sig(cands, H, pinH, gg, order, vis1, used1, use_soon)
            r = table.get((n_hit, d1, s), -1.0)
            if r > br:
                br, best = r, rec
        return best if br > -1.0 else None
    return pick


def eval_keys(rng, fn, n, table, use_soon):
    n_ok = n_ref = n_ep = hops_sum = 0
    for _ in range(n):
        lines, names = fn(rng)
        g = build(lines)
        if g is None:
            continue
        for cands, visited, used_k, order in starts_flat(g, rng, names["FRESH"]):
            n_ep += 1
            pick = make_pick_keys(table, use_soon, order, g, visited, used_k)
            hops, ok, ref = loop_keys(g, set(cands), set(visited), set(used_k),
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
    old = train_return(rng, args.train, True)
    xfer = {k: eval_keys(rng, fn, args.test, old, True) for k, fn in FAM.items()}
    tab = train_keys(rng, args.train, True)
    fit = {k: eval_keys(rng, fn, args.test, tab, True) for k, fn in FAM.items()}
    void = any(xfer[k]["ep"] < 5 or fit[k]["ep"] < 5 for k in FAM)
    xfer_ok = match_455(xfer)
    fit_ok = match_455(fit)
    new_prim = (not void) and (not xfer_ok)
    gate = new_prim and fit_ok
    rec = dict(seed=args.seed, void=bool(void), gate=bool(gate),
               new_prim=bool(new_prim), xfer_ok=xfer_ok, fit_ok=fit_ok,
               xfer=xfer, fit=fit)
    print("XFER value-table -> key-cut")
    for k in FAM:
        r = xfer[k]
        print(f"  {k:4} ep {r['ep']:2} pin {r['pin']:.2f} ref {r['refuse']:.2f} hops {r['mean_hops']:.1f}")
    print("FIT  retrain on key-cut")
    for k in FAM:
        r = fit[k]
        print(f"  {k:4} ep {r['ep']:2} pin {r['pin']:.2f} ref {r['refuse']:.2f} hops {r['mean_hops']:.1f}")
    print(f"VOID {void}  XFER {xfer_ok}  FIT {fit_ok}  new_prim {new_prim}  GATE {gate}")
    if void:
        print("\nVOID: empty AND/key options.")
    elif gate:
        print("\nGO KEYCUT: value-extra table dies; key-cut retrain recovers 455 hops. New primitive.")
    elif xfer_ok:
        print("\nSAME OBJECT: key-cut still looks like value-extra to the table.")
    else:
        print("\nSTOP: new cut broke transfer, but retrain did not recover the task.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
