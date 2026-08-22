"""482: hop3 reads tape mark. LIVE → continue; DEAD → not that slot."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import _tape_frames as tframes
from _audit440_compose import think_place, think_slot

OUT = Path("results/_stage482_nexthop.json")


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
        if pin is None or g["value"][pin] != A:
            continue
        c2 = g["by_key"].get(A, set()) - {g["place"][s]}
        if len(c2) != 2:
            continue
        out.append((s, sorted(c2)))
    return out


def touch(table, tot, win, key, r):
    tot[key] += 1
    win[key] += r
    table[key] = win[key] / tot[key]


def pick_local(cands, qloc, rng, eps):
    opts = list(cands)
    if rng.random() < eps:
        return rng.choice(opts)
    best, br = None, -1e9
    for H in opts:
        v = qloc.get(H, 0.0)
        if v > br:
            br, best = v, H
    if best is None or br <= 0:
        return rng.choice(opts)
    return best


def open_slots(cands, tape):
    return [H for H in cands if tape.get(H) != "DEAD"]


def hop3_from_mark(g, names, tape, qg, eps, rng, kind):
    live = [H for H, m in tape.items() if m == "LIVE"]
    if not live:
        return 0
    if kind == "C":
        go = (rng.random() < eps) or (qg.get("LIVE", 0.0) > 0)
        if not go:
            return 0
    nxt = [P for P in g["by_key"].get(names["SWEET"], set()) if P not in tape]
    if len(nxt) != 1:
        return 0
    pin3 = think_place(list(g["slots_at"][nxt[0]]), g["value"], rng)
    return int(pin3 is not None and g["value"][pin3] == names["FRESH"])


def episode(g, names, rng, qg, tg, wg, qloc, tl, wl, eps, online, kind):
    st = starts(g, names, rng)
    if not st:
        return None
    _, c2 = rng.choice(st)
    pool = list(c2)
    tape = {}
    first_dead = False
    for _att in range(2):
        if kind == "L":
            H = pick_local(pool, qloc, rng, eps)
        else:
            opts = open_slots(pool, tape)
            if not opts:
                break
            H = rng.choice(sorted(opts))
        pin = think_place(list(g["slots_at"][H]), g["value"], rng)
        live = pin is not None and g["value"][pin] == names["SWEET"]
        tape[H] = "LIVE" if live else "DEAD"
        if live:
            if online and kind == "L":
                touch(qloc, tl, wl, H, 1.0)
            if online and kind == "C" and first_dead:
                touch(qg, tg, wg, "DEAD", 0.8)
            break
        first_dead = True
        if online and kind == "L":
            touch(qloc, tl, wl, H, -0.4)
        if kind == "L":
            break
        retry = True if rng.random() < eps else qg.get("DEAD", 0.0) > 0
        if not retry:
            if online and kind == "C":
                touch(qg, tg, wg, "DEAD", -0.3)
            break
    h3 = hop3_from_mark(g, names, tape, qg, eps, rng, kind)
    if online and kind == "C" and any(m == "LIVE" for m in tape.values()):
        touch(qg, tg, wg, "LIVE", 0.8 if h3 else -0.3)
    h2 = int(any(m == "LIVE" for m in tape.values()))
    return dict(h1=1, h2=h2, h3=h3)


def run_n(g, names, rng, qg, qloc, n, eps, online, kind):
    tg, wg = defaultdict(int), defaultdict(float)
    tl, wl = defaultdict(int), defaultdict(float)
    n1 = n2 = n3 = 0
    for _ in range(n):
        rec = episode(g, names, rng, qg, tg, wg, qloc, tl, wl, eps, online, kind)
        if rec is None:
            continue
        n1 += rec["h1"]
        n2 += rec["h2"]
        n3 += rec["h3"]
    return dict(n1=n1, p_h2=n2 / max(n1, 1), p_h3=n3 / max(n1, 1))


def train(g, names, rng, n, kind):
    qg, qloc = {}, {}
    tg, wg = defaultdict(int), defaultdict(float)
    tl, wl = defaultdict(int), defaultdict(float)
    for i in range(n):
        eps = max(0.08, 0.5 * (1.0 - i / n))
        episode(g, names, rng, qg, tg, wg, qloc, tl, wl, eps, True, kind)
    return qg, qloc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--train", type=int, default=220)
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
    nc = len(st[0][1]) if st else 0
    t0 = run_n(ga, na, random.Random(1), {}, {}, args.test, 0.0, False, "C")
    qgC, _ = train(ga, na, random.Random(args.seed), args.train, "C")
    _, qL = train(ga, na, random.Random(args.seed + 5), args.train, "L")
    cA = run_n(ga, na, random.Random(2), qgC, {}, args.test, 0.0, False, "C")
    lA = run_n(ga, na, random.Random(2), {}, qL, args.test, 0.0, False, "L")
    cB = run_n(gb, nb, random.Random(3), qgC, {}, args.test, 0.0, False, "C")
    lB = run_n(gb, nb, random.Random(3), {}, qL, args.test, 0.0, False, "L")
    void = (nc != 2) or (t0["p_h2"] > 0.70) or (t0["n1"] < 5)
    gate = ((not void)
            and (t0["p_h3"] < 0.20)
            and (cB["p_h3"] >= 0.85)
            and (cB["p_h3"] - t0["p_h3"] > 0.50)
            and (lB["p_h2"] <= t0["p_h2"] + 0.10))
    rec = dict(seed=args.seed, void=bool(void), gate=bool(gate),
               n_cands=nc, t0=t0, qg=dict(qgC),
               c_a=cA, l_a=lA, c_b=cB, l_b=lB)
    print("qg", qgC)
    print("t0 h2/h3", round(t0["p_h2"], 2), round(t0["p_h3"], 2))
    print("A  C", round(cA["p_h3"], 2), "L h2", round(lA["p_h2"], 2))
    print("B  C", round(cB["p_h3"], 2), "L h2", round(lB["p_h2"], 2))
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("\nVOID: not a 2-way hop2, or t0 already high.")
    elif gate:
        print("\nGO NEXT: hop3 reads LIVE on tape; Q[LIVE] transfers; H dies on B.")
    else:
        print("\nSTOP: mark did not become hop3 context.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
