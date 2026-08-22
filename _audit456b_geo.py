"""456-B: train on 455, test on flat start (no APPLES→SWEET→FRESH)."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from _audit440_compose import think_slot
from _audit447_seek import next_cands
from _audit449_adv import build
from _audit452_xfer import _lex, _pad
from _audit453_depth import loop
from _audit456_policy import make_pick, train_return

OUT = Path("results/_stage456b_geo.json")


def _q(n, rng):
    ctx = [n["red"], n["cat"], n["sat"], n["on"], n["the"], n["mat"]]
    rng.shuffle(ctx)
    q = [f"{ctx[0]} {ctx[1]} {ctx[2]} {n['FRESH']} {ctx[3]} {ctx[4]} {ctx[5]}" + _pad(i) for i in range(3)]
    q.append(f"{ctx[0]} {ctx[1]} {ctx[2]} {n['STALE']} {ctx[3]} {ctx[4]} {ctx[5]}" + _pad(3))
    noise = [f"{n['n1']} {n['n2']} {n['FOG']} {n['n3']} {n['n4']} {n['yes']}" + _pad(200 + i) for i in range(2)]
    return q, noise, n["red"], n["cat"], n["sat"]


def _cands4(n, mark=None):
    tag = n["TAG"] if mark is None else mark
    c1 = [f"{n['aa']} {n['bb']} {n['cc']} {n['CRISP']} {n['FRESH']} {tag} {n['TAG']}" + _pad(50 + i) for i in range(3)]
    c1.append(f"{n['aa']} {n['bb']} {n['cc']} {n['SOFT']} {n['FRESH']} {tag} {n['TAG']}" + _pad(53))
    c2 = [f"{n['dd']} {n['ee']} {n['ff']} {n['RIPE']} {n['FRESH']} {n['KEYA']} {n['u1']}" + _pad(60 + i) for i in range(3)]
    c2.append(f"{n['dd']} {n['ee']} {n['ff']} {n['RAW']} {n['FRESH']} {n['KEYA']} {n['u1']}" + _pad(63))
    c3 = [f"{n['gg']} {n['hh']} {n['ii']} {n['DRY']} {n['FRESH']} {n['KEYA']} {n['u2']}" + _pad(70 + i) for i in range(3)]
    c3.append(f"{n['gg']} {n['hh']} {n['ii']} {n['MUSH']} {n['FRESH']} {n['KEYA']} {n['u2']}" + _pad(73))
    c4 = [f"{n['jj']} {n['kk']} {n['ll']} {n['WET']} {n['FRESH']} {n['u3']} {n['u4']}" + _pad(80 + i) for i in range(3)]
    c4.append(f"{n['jj']} {n['kk']} {n['ll']} {n['DAMP']} {n['FRESH']} {n['u3']} {n['u4']}" + _pad(83))
    return c1, c2, c3, c4


def world_d1(rng):
    n = _lex(rng)
    q, noise, red, cat, sat = _q(n, rng)
    c1 = [f"{n['aa']} {n['bb']} {n['cc']} {n['CRISP']} {n['FRESH']} {n['TAG']} {n['u1']}" + _pad(50 + i) for i in range(3)]
    c1.append(f"{n['aa']} {n['bb']} {n['cc']} {n['SOFT']} {n['FRESH']} {n['TAG']} {n['u1']}" + _pad(53))
    c2 = [f"{n['dd']} {n['ee']} {n['ff']} {n['RIPE']} {n['FRESH']} {n['u2']} {n['u3']}" + _pad(60 + i) for i in range(3)]
    c2.append(f"{n['dd']} {n['ee']} {n['ff']} {n['RAW']} {n['FRESH']} {n['u2']} {n['u3']}" + _pad(63))
    h = [f"{n['vault']} {n['keeps']} {n['TAG']} {red} {n['safe']} {n['here']}" + _pad(110 + i) for i in range(3)]
    h.append(f"{n['vault']} {n['keeps']} {n['FOG']} {red} {n['safe']} {n['here']}" + _pad(113))
    return q + c1 + c2 + h + noise, n


def world_d2(rng):
    n = _lex(rng)
    q, noise, red, cat, sat = _q(n, rng)
    c1, c2, c3, c4 = _cands4(n)
    hA = [f"{n['clerk']} {n['wrote']} {n['KEYA']} {sat} {n['board']} {n['now']}" + _pad(90 + i) for i in range(3)]
    hA.append(f"{n['clerk']} {n['wrote']} {n['FOG']} {sat} {n['board']} {n['now']}" + _pad(93))
    hS = [f"{n['desk']} {n['shows']} {n['UNLOCK']} {red} {n['label']} {n['here']}" + _pad(100 + i) for i in range(3)]
    hS.append(f"{n['desk']} {n['shows']} {n['FOG']} {red} {n['label']} {n['here']}" + _pad(103))
    hT = [f"{n['vault']} {n['keeps']} {n['TAG']} {n['UNLOCK']} {n['safe']} {n['here']}" + _pad(110 + i) for i in range(3)]
    hT.append(f"{n['vault']} {n['keeps']} {n['FOG']} {n['UNLOCK']} {n['safe']} {n['here']}" + _pad(113))
    return q + c1 + c2 + c3 + c4 + hA + hS + hT + noise, n


def world_d4(rng):
    n = _lex(rng)
    q, noise, red, cat, sat = _q(n, rng)
    c1, c2, c3, c4 = _cands4(n)
    hA = [f"{n['clerk']} {n['wrote']} {n['KEYA']} {sat} {n['board']} {n['now']}" + _pad(90 + i) for i in range(3)]
    hA.append(f"{n['clerk']} {n['wrote']} {n['FOG']} {sat} {n['board']} {n['now']}" + _pad(93))
    u1, u2, u3 = n["UNLOCK"], n["n1"], n["n2"]
    h1 = [f"{n['desk']} {n['shows']} {u1} {red} {n['label']} {n['here']}" + _pad(100 + i) for i in range(3)]
    h1.append(f"{n['desk']} {n['shows']} {n['FOG']} {red} {n['label']} {n['here']}" + _pad(103))
    h2 = [f"{n['xx']} {n['yy']} {u2} {u1} {n['safe']} {n['here']}" + _pad(120 + i) for i in range(3)]
    h2.append(f"{n['xx']} {n['yy']} {n['FOG']} {u1} {n['safe']} {n['here']}" + _pad(123))
    h3 = [f"{n['kids']} {n['like']} {u3} {u2} {n['today']} {n['now']}" + _pad(130 + i) for i in range(3)]
    h3.append(f"{n['kids']} {n['like']} {n['FOG']} {u2} {n['today']} {n['now']}" + _pad(133))
    h4 = [f"{n['vault']} {n['keeps']} {n['TAG']} {u3} {n['fruit']} {n['here']}" + _pad(140 + i) for i in range(3)]
    h4.append(f"{n['vault']} {n['keeps']} {n['FOG']} {u3} {n['fruit']} {n['here']}" + _pad(143))
    return q + c1 + c2 + c3 + c4 + hA + h1 + h2 + h3 + h4 + noise, n


def world_both(rng):
    n = _lex(rng)
    q, noise, red, cat, sat = _q(n, rng)
    mark = n["n4"]
    c1, c2, c3, c4 = _cands4(n, mark=mark)
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
    return q + c1 + c2 + c3 + c4 + hA + hS + hT + h1 + h2 + h3 + h4 + noise, n


def world_stop(rng):
    n = _lex(rng)
    q, noise, red, cat, sat = _q(n, rng)
    c1, c2, c3, c4 = _cands4(n)
    hA = [f"{n['clerk']} {n['wrote']} {n['KEYA']} {red} {n['board']} {n['now']}" + _pad(90 + i) for i in range(3)]
    hA.append(f"{n['clerk']} {n['wrote']} {n['FOG']} {red} {n['board']} {n['now']}" + _pad(93))
    return q + c1 + c2 + c3 + c4 + hA + noise, n


FAM_B = {"D1": world_d1, "D2": world_d2, "BOTH": world_both, "D4": world_d4, "STOP": world_stop}


def starts_flat(g, rng, tok):
    place, value, line = g["place"], g["value"], g["line"]
    out = []
    for s in range(g["n"]):
        if value[s] != tok:
            continue
        pin = think_slot(s, g["slots_at"], place, value, line, rng)
        if pin is None:
            continue
        cands = next_cands(value[pin], place[s], g["by_key"])
        if len(cands) < 2:
            continue
        visited = {place[s]}
        used_k = {value[pin]}
        order = list(g["place_ord"][place[s]])
        out.append((cands, visited, used_k, order))
    return out


def eval_flat(rng, fn, n, table, use_soon):
    n_ok = n_ref = n_ep = hops_sum = 0
    for _ in range(n):
        lines, names = fn(rng)
        g = build(lines)
        if g is None:
            continue
        for cands, visited, used_k, order in starts_flat(g, rng, names["FRESH"]):
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
    yes = {k: eval_flat(rng, fn, args.test, tab_s, True) for k, fn in FAM_B.items()}
    no = {k: eval_flat(rng, fn, args.test, tab_n, False) for k, fn in FAM_B.items()}
    void = any(yes[k]["ep"] < 5 for k in FAM_B)
    gate = ((not void)
            and (yes["D1"]["pin"] == 1.0) and (yes["D1"]["mean_hops"] == 1.0)
            and (yes["D2"]["pin"] == 1.0) and (yes["D2"]["mean_hops"] == 2.0)
            and (yes["BOTH"]["pin"] == 1.0) and (yes["BOTH"]["mean_hops"] == 2.0)
            and (yes["D4"]["pin"] == 1.0) and (yes["D4"]["mean_hops"] == 4.0)
            and (yes["STOP"]["refuse"] == 1.0) and (yes["STOP"]["pin"] == 0.0)
            and (no["BOTH"]["mean_hops"] > 2.0))
    rec = dict(seed=args.seed, void=bool(void), gate=bool(gate),
               with_soon=yes, no_soon=no)
    print("flat-start, table trained on chain-start 455")
    print("with soon")
    for k in FAM_B:
        r = yes[k]
        print(f"  {k:4} ep {r['ep']:2} pin {r['pin']:.2f} ref {r['refuse']:.2f} hops {r['mean_hops']:.1f}")
    print("no soon  BOTH hops", round(no["BOTH"]["mean_hops"], 2),
          "pin", round(no["BOTH"]["pin"], 2))
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("\nVOID: a family had <5 test eps.")
    elif gate:
        print("\nGO FLAT: frozen return-policy chain-start -> flat-start, same downstream geometry. soon still required on BOTH.")
    else:
        print("\nSTOP: frozen table does not run flat-start.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
