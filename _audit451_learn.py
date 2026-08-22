"""451: learn first-seek from outcomes. No n_follow in the chooser."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from _audit440_compose import think_place, think_slot
from _audit447_seek import hop, next_cands
from _audit448_pick import all_seeks
from _audit449_adv import build, filter_cands
from _audit450_follow import opened

OUT = Path("results/_stage451_learn.json")


def _pad(k):
    return " " + " ".join(f"z{k}x{j}" for j in range(20))


def _tok(rng, used):
    while True:
        w = "w" + "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5))
        if w not in used:
            used.add(w)
            return w


def iso_world(rng):
    u = set()
    n = {k: _tok(rng, u) for k in (
        "APPLES", "ORANGES", "SWEET", "SOUR", "FRESH", "STALE",
        "CRISP", "SOFT", "RIPE", "RAW", "DRY", "MUSH", "WET", "DAMP",
        "KEYA", "UNLOCK", "TAG", "FOG", "PEARS", "PLUMS",
        "u1", "u2", "u3", "u4")}
    p = [f"red cat sat {n['APPLES']} on the mat" + _pad(i) for i in range(3)]
    p.append(f"red cat sat {n['ORANGES']} on the mat" + _pad(3))
    r = [f"kids like {n['SWEET']} {n['APPLES']} today yes" + _pad(20 + i) for i in range(3)]
    r.append(f"kids like {n['SOUR']} {n['APPLES']} today yes" + _pad(23))
    t = [f"barns store {n['FRESH']} {n['SWEET']} fruit now" + _pad(30 + i) for i in range(3)]
    t.append(f"barns store {n['STALE']} {n['SWEET']} fruit now" + _pad(33))
    c1 = [f"aa bb cc {n['CRISP']} {n['FRESH']} {n['KEYA']} {n['TAG']}" + _pad(50 + i) for i in range(3)]
    c1.append(f"aa bb cc {n['SOFT']} {n['FRESH']} {n['KEYA']} {n['TAG']}" + _pad(53))
    c2 = [f"dd ee ff {n['RIPE']} {n['FRESH']} {n['KEYA']} {n['u1']}" + _pad(60 + i) for i in range(3)]
    c2.append(f"dd ee ff {n['RAW']} {n['FRESH']} {n['KEYA']} {n['u1']}" + _pad(63))
    c3 = [f"gg hh ii {n['DRY']} {n['FRESH']} {n['KEYA']} {n['u2']}" + _pad(70 + i) for i in range(3)]
    c3.append(f"gg hh ii {n['MUSH']} {n['FRESH']} {n['KEYA']} {n['u2']}" + _pad(73))
    c4 = [f"jj kk ll {n['WET']} {n['FRESH']} {n['u3']} {n['u4']}" + _pad(80 + i) for i in range(3)]
    c4.append(f"jj kk ll {n['DAMP']} {n['FRESH']} {n['u3']} {n['u4']}" + _pad(83))
    hA = [f"clerk wrote {n['KEYA']} red board now" + _pad(90 + i) for i in range(3)]
    hA.append(f"clerk wrote {n['FOG']} red board now" + _pad(93))
    hB = [f"desk shows {n['UNLOCK']} cat label here" + _pad(100 + i) for i in range(3)]
    hB.append(f"desk shows {n['FOG']} cat label here" + _pad(103))
    hT = [f"vault keeps {n['TAG']} {n['UNLOCK']} safe here" + _pad(110 + i) for i in range(3)]
    hT.append(f"vault keeps {n['FOG']} {n['UNLOCK']} safe here" + _pad(113))
    q = []
    for i, f in enumerate((n["PEARS"], n["PLUMS"], n["PEARS"], n["PLUMS"])):
        q.append(f"blue dog lay {f} in the sun" + _pad(10 + i))
    return p + r + t + c1 + c2 + c3 + c4 + hA + hB + hT + q, n


def episode_start(g, rng, start_tok):
    place, value, line = g["place"], g["value"], g["line"]
    out = []
    for s in range(g["n"]):
        if value[s] != start_tok:
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
        if len(cands) != 4:
            continue
        visited = {place[s], R, T}
        used_k = {value[pin1], value[pin2], value[pin3]}
        order = list(g["place_ord"][place[s]])
        out.append((cands, visited, used_k, order, s))
    return out


def finish(H, pinH, cands, extra, order, by_key, vis1, used1, slots_at, value, place_keys, rng, target):
    extra1 = extra | {value[pinH]}
    c1 = filter_cands(cands, extra1, place_keys)
    d1 = len(c1)
    n_hit = sum(1 for p in cands if value[pinH] in place_keys[p])
    opts2 = opened(value[pinH], order, by_key, vis1, c1, used1)
    if len(opts2) != 1:
        return n_hit, d1, False
    _, H2 = opts2[0]
    pin2b = think_place(list(slots_at[H2]), value, rng)
    if pin2b is None:
        return n_hit, d1, False
    c2 = filter_cands(c1, extra1 | {value[pin2b]}, place_keys)
    if len(c2) != 1:
        return n_hit, d1, False
    pin4 = think_place(list(slots_at[next(iter(c2))]), value, rng)
    ok = pin4 is not None and value[pin4] == target
    return n_hit, d1, ok


def options(cands, order, by_key, visited, used_k, slots_at, value, rng):
    opts = []
    for k, H in all_seeks(order, by_key, visited, cands, used_k):
        pinH = think_place(list(slots_at[H]), value, rng)
        if pinH is None:
            continue
        vis1 = set(visited) | {H}
        used1 = set(used_k) | {k}
        opts.append((k, H, pinH, vis1, used1))
    return opts


def greedy_pick(opts, cands, extra, place_keys, value):
    best, bd = None, 10 ** 9
    for rec in opts:
        pinH = rec[2]
        c1 = filter_cands(cands, extra | {value[pinH]}, place_keys)
        if len(c1) < bd:
            bd = len(c1)
            best = rec
    return best


def run_split(rng, n_train, n_test):
    win, tot = defaultdict(int), defaultdict(int)
    for i in range(n_train):
        lines, names = iso_world(rng)
        g = build(lines)
        if g is None:
            continue
        for cands, visited, used_k, order, s in episode_start(g, rng, names["APPLES"]):
            opts = options(cands, order, g["by_key"], visited, used_k,
                           g["slots_at"], g["value"], rng)
            for rec in opts:
                k, H, pinH, vis1, used1 = rec
                n_hit, d1, ok = finish(
                    H, pinH, cands, set(), order, g["by_key"], vis1, used1,
                    g["slots_at"], g["value"], g["place_keys"], rng, names["CRISP"])
                tot[(n_hit, d1)] += 1
                win[(n_hit, d1)] += int(ok)
    rate = {sig: win[sig] / tot[sig] for sig in tot}

    def learned_pick(opts, cands, extra, place_keys, value):
        best, br = None, -1.0
        for rec in opts:
            pinH = rec[2]
            n_hit = sum(1 for p in cands if value[pinH] in place_keys[p])
            d1 = len(filter_cands(cands, extra | {value[pinH]}, place_keys))
            r = rate.get((n_hit, d1), 0.0)
            if r > br:
                br = r
                best = rec
        return best if br > 0 else None

    n_l = n_g = n_r = n_ep = 0
    for i in range(n_test):
        lines, names = iso_world(rng)
        g = build(lines)
        if g is None:
            continue
        for cands, visited, used_k, order, s in episode_start(g, rng, names["APPLES"]):
            opts = options(cands, order, g["by_key"], visited, used_k,
                           g["slots_at"], g["value"], rng)
            if len(opts) < 2:
                continue
            n_ep += 1
            for picker, bucket in (
                (learned_pick, "L"),
                (greedy_pick, "G"),
                (lambda o, c, e, pk, v: rng.choice(o), "R"),
            ):
                rec = picker(opts, cands, set(), g["place_keys"], g["value"])
                if rec is None:
                    continue
                k, H, pinH, vis1, used1 = rec
                _, _, ok = finish(
                    H, pinH, cands, set(), order, g["by_key"], vis1, used1,
                    g["slots_at"], g["value"], g["place_keys"], rng, names["CRISP"])
                if bucket == "L":
                    n_l += int(ok)
                elif bucket == "G":
                    n_g += int(ok)
                else:
                    n_r += int(ok)
    return dict(
        n_train=n_train, n_test_ep=n_ep, n_sig=len(tot),
        learned=n_l / max(n_ep, 1),
        greedy=n_g / max(n_ep, 1),
        random=n_r / max(n_ep, 1),
        signatures={f"{a},{b}": round(rate[(a, b)], 3) for a, b in rate},
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--train", type=int, default=20)
    ap.add_argument("--test", type=int, default=10)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    rng = random.Random(args.seed)
    rep = run_split(rng, args.train, args.test)
    void = (rep["n_test_ep"] == 0) or (rep["n_sig"] == 0)
    gate = ((not void)
            and (rep["learned"] == 1.0)
            and (rep["greedy"] == 0.0)
            and (rep["learned"] - rep["random"] >= 0.25))
    rec = dict(seed=args.seed, void=bool(void), gate=bool(gate), **rep)
    print(f"train {rep['n_train']} test_ep {rep['n_test_ep']} sigs {rep['n_sig']}")
    print(f"learned {rep['learned']:.2f}  greedy {rep['greedy']:.2f}  random {rep['random']:.2f}")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("\nVOID: no test episodes or empty train table.")
    elif gate:
        print("\nGO LEARN: win-rate over (n_hit,d1) transfers to new names.")
    else:
        print("\nSTOP: learned policy did not beat greedy/random on held-out isos.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
