"""479: (1) hop3 fork SAME episode after +0.2; (2) iso, not Q[H]."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import _tape_frames as tframes
from _audit440_compose import think_place, think_slot

OUT = Path("results/_stage479_intra.json")
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
        "PEARS", "PLUMS", "ROTTEN", "MOLD")}
    p = [f"red cat sat {n['APPLES']} on the mat" + _pad(i) for i in range(3)]
    p.append(f"red cat sat {n['ORANGES']} on the mat" + _pad(3))
    r = [f"kids like {n['SWEET']} {n['APPLES']} today yes" + _pad(20 + i) for i in range(3)]
    r.append(f"kids like {n['SOUR']} {n['APPLES']} today yes" + _pad(23))
    d = [f"goats eat {n['ROTTEN']} {n['APPLES']} tonight no" + _pad(40 + i) for i in range(3)]
    d.append(f"goats eat {n['SOUR']} {n['APPLES']} tonight no" + _pad(43))
    t = [f"barns store {n['FRESH']} {n['SWEET']} fruit now" + _pad(30 + i) for i in range(3)]
    t.append(f"barns store {n['STALE']} {n['SWEET']} fruit now" + _pad(33))
    m = [f"owls hide {n['MOLD']} {n['SWEET']} cave dark" + _pad(50 + i) for i in range(3)]
    m.append(f"owls hide {n['STALE']} {n['SWEET']} cave dark" + _pad(53))
    q = []
    for i, f in enumerate((n["PEARS"], n["PLUMS"], n["PEARS"], n["PLUMS"])):
        q.append(f"blue dog lay {f} in the sun" + _pad(10 + i))
    return p + r + d + t + m + q, n


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
    pk = {P: set() for P in slots_at}
    for s in range(n):
        pk[place[s]] |= set(keys[s])
    return dict(n=n, place=place, value=value, line=line, keys=keys,
                slots_at=slots_at, by_key=by_key, place_keys=pk)


def pre(H, g):
    n_k = len(g["place_keys"].get(H, ()))
    n_s = len(g["slots_at"].get(H, ()))
    return (n_k, n_s)


def starts(g, names, rng):
    out = []
    A = names["APPLES"]
    for s in range(g["n"]):
        if g["value"][s] != A:
            continue
        pin = think_slot(s, g["slots_at"], g["place"], g["value"], g["line"], rng)
        if pin is None or g["value"][pin] != A:
            continue
        c2 = g["by_key"].get(A, set()) - {g["place"][s]}
        if len(c2) != 2:
            continue
        out.append((s, sorted(c2)))
    return out


def pick(cands, table, keyfn, rng, eps):
    opts = list(cands)
    if rng.random() < eps:
        return rng.choice(opts)
    best, br = None, -1e9
    for H in opts:
        v = table.get(keyfn(H), 0.0)
        if v > br:
            br, best = v, H
    if best is None or br <= 0:
        return rng.choice(opts)
    return best


def touch(table, tot, win, key, r):
    tot[key] += 1
    win[key] += r
    table[key] = win[key] / tot[key]


def episode(g, names, rng, table, tot, win, eps, online, kind):
    st = starts(g, names, rng)
    if not st:
        return None
    s, c2 = rng.choice(st)
    keyfn = (lambda H: H) if kind == "H" else (lambda H: pre(H, g))
    H2 = pick(c2, table, keyfn, rng, eps)
    pin2 = think_place(list(g["slots_at"][H2]), g["value"], rng)
    k2 = keyfn(H2)
    if pin2 is None or g["value"][pin2] != names["SWEET"]:
        if online:
            touch(table, tot, win, k2, -0.35)
        return dict(h1=1, h2=0, h3=0, n3=0)
    if online:
        touch(table, tot, win, k2, 0.15)
    c3 = g["by_key"].get(names["SWEET"], set()) - {H2}
    n3 = len(c3)
    if n3 != 2:
        return dict(h1=1, h2=1, h3=0, n3=n3)
    H3 = pick(c3, table, keyfn, rng, eps)
    pin3 = think_place(list(g["slots_at"][H3]), g["value"], rng)
    ok = pin3 is not None and g["value"][pin3] == names["FRESH"]
    if online:
        touch(table, tot, win, keyfn(H3), 1.0 if ok else -0.3)
    return dict(h1=1, h2=1, h3=int(ok), n3=n3)


def run_n(g, names, rng, table, tot, win, n, eps, online, kind):
    n1 = n2 = n3 = n3c = 0
    for _ in range(n):
        rec = episode(g, names, rng, table, tot, win, eps, online, kind)
        if rec is None:
            continue
        n1 += rec["h1"]
        n2 += rec["h2"]
        n3 += rec["h3"]
        n3c = rec["n3"]
    return dict(n1=n1, p_h2=n2 / max(n1, 1), p_h3=n3 / max(n1, 1), n3=n3c)


def train(g, names, rng, n, kind):
    table, tot, win = {}, defaultdict(int), defaultdict(float)
    for i in range(n):
        eps = max(0.08, 0.5 * (1.0 - i / n))
        episode(g, names, rng, table, tot, win, eps, True, kind)
    return table, tot, win


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--train", type=int, default=250)
    ap.add_argument("--test", type=int, default=40)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    la, na = world(random.Random(args.seed))
    lb, nb = world(random.Random(args.seed + 99))
    ga, gb = graph(la), graph(lb)
    if ga is None or gb is None:
        print("no tape")
        return 1
    st = starts(ga, na, random.Random(0))
    nc2 = len(st[0][1]) if st else 0
    t0 = run_n(ga, na, random.Random(1), {}, defaultdict(int), defaultdict(float),
               args.test, 0.0, False, "H")
    qh, _, _ = train(ga, na, random.Random(args.seed), args.train, "H")
    qp, _, _ = train(ga, na, random.Random(args.seed + 1), args.train, "P")
    hA = run_n(ga, na, random.Random(2), qh, defaultdict(int), defaultdict(float),
               args.test, 0.0, False, "H")
    pA = run_n(ga, na, random.Random(4), qp, defaultdict(int), defaultdict(float),
               args.test, 0.0, False, "P")
    hB = run_n(gb, nb, random.Random(3), qh, defaultdict(int), defaultdict(float),
               args.test, 0.0, False, "H")
    pB = run_n(gb, nb, random.Random(5), qp, defaultdict(int), defaultdict(float),
               args.test, 0.0, False, "P")
    void = (nc2 != 2) or (t0["n3"] != 2) or (t0["p_h3"] > 0.6) or (t0["n1"] < 5)
    gate1 = ((not void)
             and (hA["p_h3"] - t0["p_h3"] > DELTA)
             and (hA["p_h3"] >= 0.80))
    gate2 = ((not void)
             and (hB["p_h3"] <= t0["p_h3"] + 0.10)
             and (pB["p_h3"] - t0["p_h3"] > DELTA))
    rec = dict(seed=args.seed, void=bool(void), gate=bool(gate1 and gate2),
               gate1=bool(gate1), gate2=bool(gate2),
               n2=nc2, n3=t0["n3"], t0=t0, qh_a=hA, qh_b=hB, qp_a=pA, qp_b=pB,
               n_qh=len(qh), n_qp=len(qp),
               names_diff=na["FRESH"] != nb["FRESH"])
    print("cands hop2/hop3", nc2, t0["n3"], "pre keys", len(qp))
    print("t0 h3", round(t0["p_h3"], 2))
    print("A QH", round(hA["p_h3"], 2), "QP", round(pA["p_h3"], 2))
    print("B QH", round(hB["p_h3"], 2), "QP", round(pB["p_h3"], 2))
    print(f"VOID {void}  GATE1 {gate1}  GATE2 {gate2}")
    if void:
        print("\nVOID: forks missing, or t0 already high.")
    elif gate1 and gate2:
        print("\nGO 1+2: intra hop3 learns on A; pre transfers, H does not.")
    elif gate1:
        print("\nSTOP at 2: intra on A yes; iso pre did not beat H-blind.")
    else:
        print("\nSTOP at 1: hop3 in-episode did not learn.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
