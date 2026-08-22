"""478 D/B: online credit mid-episode on 441-chain + 1 decoy hop2."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import _tape_frames as tframes
from _audit440_compose import think_place, think_slot

OUT = Path("results/_stage478_online.json")
DELTA = 0.20


def _pad(k):
    return " " + " ".join(f"z{k}x{j}" for j in range(20))


def _tok(rng, used):
    while True:
        w = "w" + "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5))
        if w not in used:
            used.add(w)
            return w


def world(rng):
    u = set()
    n = {k: _tok(rng, u) for k in (
        "APPLES", "ORANGES", "SWEET", "SOUR", "FRESH", "STALE",
        "PEARS", "PLUMS", "ROTTEN")}
    p = [f"red cat sat {n['APPLES']} on the mat" + _pad(i) for i in range(3)]
    p.append(f"red cat sat {n['ORANGES']} on the mat" + _pad(3))
    r = [f"kids like {n['SWEET']} {n['APPLES']} today yes" + _pad(20 + i) for i in range(3)]
    r.append(f"kids like {n['SOUR']} {n['APPLES']} today yes" + _pad(23))
    d = [f"goats eat {n['ROTTEN']} {n['APPLES']} tonight no" + _pad(40 + i) for i in range(3)]
    d.append(f"goats eat {n['SOUR']} {n['APPLES']} tonight no" + _pad(43))
    t = [f"barns store {n['FRESH']} {n['SWEET']} fruit now" + _pad(30 + i) for i in range(3)]
    t.append(f"barns store {n['STALE']} {n['SWEET']} fruit now" + _pad(33))
    q = []
    for i, f in enumerate((n["PEARS"], n["PLUMS"], n["PEARS"], n["PLUMS"])):
        q.append(f"blue dog lay {f} in the sun" + _pad(10 + i))
    return p + r + d + t + q, n


def graph(lines):
    keep, toks, owner = tframes.frame_keep(lines, 3, 2)
    if not keep:
        return None
    place, value, line, keys = [], [], [], []
    for (w, left, right), ps in keep:
        pname = f"{' '.join(left)}|{' '.join(right)}"
        ks = {x for x in list(left) + list(right) if x}
        for i in ps:
            place.append(pname)
            value.append(toks[i])
            line.append(owner[i])
            keys.append(ks)
    n = len(place)
    slots_at = defaultdict(list)
    for s in range(n):
        slots_at[place[s]].append(s)
    by_key = defaultdict(set)
    for s in range(n):
        for k in keys[s]:
            by_key[k].add(place[s])
    return dict(n=n, place=place, value=value, line=line, keys=keys,
                slots_at=slots_at, by_key=by_key)


def starts(g, names, rng):
    out = []
    A = names["APPLES"]
    for s in range(g["n"]):
        if g["value"][s] != A:
            continue
        pin = think_slot(s, g["slots_at"], g["place"], g["value"], g["line"], rng)
        if pin is None:
            continue
        if g["value"][pin] != A:
            continue
        cands = g["by_key"].get(A, set()) - {g["place"][s]}
        if len(cands) != 2:
            continue
        out.append((s, pin, sorted(cands)))
    return out


def pick(cands, table, rng, eps):
    opts = list(cands)
    if rng.random() < eps:
        return rng.choice(opts)
    best, br = None, -1e9
    for H in opts:
        v = table.get(H, 0.0)
        if v > br:
            br, best = v, H
    if best is None or br <= 0:
        return rng.choice(opts)
    return best


def touch(table, tot, win, H, r):
    tot[H] += 1
    win[H] += r
    table[H] = win[H] / tot[H]


def episode(g, names, rng, table, tot, win, eps, online):
    st = starts(g, names, rng)
    if not st:
        return None
    s, pin1, cands = rng.choice(st)
    H = pick(cands, table, rng, eps)
    pin2 = think_place(list(g["slots_at"][H]), g["value"], rng)
    cost = -0.05
    if pin2 is None:
        if online:
            touch(table, tot, win, H, cost - 0.3)
        return dict(h1=1, h2=0, h3=0)
    v2 = g["value"][pin2]
    if v2 != names["SWEET"]:
        if online:
            touch(table, tot, win, H, cost - 0.3)
        return dict(h1=1, h2=0, h3=0)
    if online:
        touch(table, tot, win, H, cost + 0.2)
    nxt = g["by_key"].get(v2, set()) - {H}
    if len(nxt) != 1:
        return dict(h1=1, h2=1, h3=0)
    T = next(iter(nxt))
    pin3 = think_place(list(g["slots_at"][T]), g["value"], rng)
    ok3 = pin3 is not None and g["value"][pin3] == names["FRESH"]
    if online:
        touch(table, tot, win, H, 1.0 if ok3 else 0.0)
    return dict(h1=1, h2=1, h3=int(ok3))


def run_n(g, names, rng, table, tot, win, n, eps, online):
    n2 = n1 = n3 = 0
    for _ in range(n):
        rec = episode(g, names, rng, table, tot, win, eps, online)
        if rec is None:
            continue
        n1 += rec["h1"]
        n2 += rec["h2"]
        n3 += rec["h3"]
    return dict(n1=n1, p_h2=n2 / max(n1, 1), p_h3=n3 / max(n1, 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--train", type=int, default=200)
    ap.add_argument("--test", type=int, default=40)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    lines, names = world(random.Random(args.seed))
    g = graph(lines)
    if g is None:
        print("no tape")
        return 1
    st = starts(g, names, random.Random(0))
    nc = len(st[0][2]) if st else 0
    t0 = run_n(g, names, random.Random(1), {}, defaultdict(int), defaultdict(float),
               args.test, 0.0, False)
    tab, tot, win = {}, defaultdict(int), defaultdict(float)
    rng_tr = random.Random(args.seed)
    for i in range(args.train):
        eps = max(0.08, 0.5 * (1.0 - i / args.train))
        episode(g, names, rng_tr, tab, tot, win, eps, True)
    t1 = run_n(g, names, random.Random(2), tab, tot, win, args.test, 0.0, False)
    fr, ft, fw = {}, defaultdict(int), defaultdict(float)
    rng_f = random.Random(args.seed + 7)
    for i in range(args.train):
        episode(g, names, rng_f, fr, ft, fw, 0.5, False)
    fz = run_n(g, names, random.Random(3), fr, ft, fw, args.test, 0.0, False)
    void = (nc != 2) or (t0["p_h2"] > 0.85) or (t0["n1"] < 5)
    gate = ((not void)
            and (t1["p_h2"] - t0["p_h2"] > DELTA)
            and (t1["p_h2"] >= 0.80)
            and (abs(fz["p_h2"] - t0["p_h2"]) < 0.10))
    rec = dict(seed=args.seed, void=bool(void), gate=bool(gate),
               n_cands=nc, n_starts=len(st), t0=t0, t1=t1, frozen=fz,
               n_q=len(tab))
    print("hop2 cands", nc, "starts", len(st))
    print("t0", round(t0["p_h2"], 2), "t1", round(t1["p_h2"], 2),
          "frozen", round(fz["p_h2"], 2), "h3", round(t1["p_h3"], 2))
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("\nVOID: hop2 not a 2-way choice, or t0 already solved.")
    elif gate:
        print("\nGO ONLINE: hop2 after pin APPLES learns mid-episode; frozen stays t0.")
    else:
        print("\nSTOP: online credit did not lift P(hop2|hop1).")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
