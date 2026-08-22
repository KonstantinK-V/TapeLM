"""511: second ring. Same 1/df budget as 510. And spent the hop; физика continues."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import _tape_frames as tframes

OUT = Path("results/_stage511_ring.json")
WIKI = Path("data/_wikitext103_train.txt")
FALLBACK = Path("data/external_tinystories_mini.txt")
HIGH_DF = 80


def pick_corpus(explicit):
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise SystemExit(f"no corpus {p}")
        kind = "wiki" if "wiki" in p.name.lower() else "given"
        return p, kind, 80 if kind == "wiki" else 20
    if WIKI.exists():
        return WIKI, "wiki", 80
    if FALLBACK.exists():
        return FALLBACK, "tinystories-fallback", 20
    raise SystemExit("no corpus")


def graph(lines, frame_max, min_fillers):
    keep, toks, owner = tframes.frame_keep(lines, frame_max, min_fillers)
    if not keep:
        return None
    value, keys = [], []
    for (w, left, right), ps in keep:
        ks = tuple(x for x in list(left) + list(right) if x)
        for i in ps:
            value.append(toks[i])
            keys.append(ks)
    n = len(value)
    df = Counter()
    for s in range(n):
        df[value[s]] += 1
        for k in keys[s]:
            df[k] += 1
    return dict(n=n, value=value, keys=keys, df=df)


def mentions(g):
    by = defaultdict(list)
    for s in range(g["n"]):
        v = g["value"][s]
        by[v].append(s)
        for k in g["keys"][s]:
            if k != v:
                by[k].append(s)
    return {v: list(dict.fromkeys(sl)) for v, sl in by.items()}


def comps(g, s, v):
    out = [x for x in g["keys"][s] if x != v]
    fv = g["value"][s]
    if fv != v:
        out.append(fv)
    return out


def rec_all(g, by, v, cap=30):
    sl = list(by.get(v, ()))[:cap]
    if len(sl) < 8:
        return []
    cnt = Counter()
    for s in sl:
        cnt.update(set(comps(g, s, v)))
    return [(c, n) for c, n in cnt.items() if n >= 2 and c != v]


def cheap_rec(g, by, v, cache):
    if v not in cache:
        rec = rec_all(g, by, v)
        rec.sort(key=lambda cn: g["df"][cn[0]])
        cache[v] = [c for c, _ in rec if g["df"][c] <= HIGH_DF]
    return cache[v]


def meets_of(g, by, nodes, v, cache):
    if len(nodes) < 2:
        return 0
    sets = [set(cheap_rec(g, by, c, cache)) for c in nodes]
    seen = set()
    n = 0
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            meet = sets[i] & sets[j]
            meet.discard(v)
            meet -= set(nodes)
            for m in meet:
                if m not in seen:
                    seen.add(m)
                    n += 1
    return n


def walk(g, by, v, cache):
    allow = max(1, int(200 / max(g["df"][v], 1)))
    r1, seen = [], {v}
    for c in cheap_rec(g, by, v, cache):
        if len(r1) >= allow:
            break
        if c in seen:
            continue
        seen.add(c)
        r1.append(c)
    remain = allow - len(r1)
    r2 = []
    frontier = list(r1)
    while remain > 0 and frontier:
        nxt = []
        for a in frontier:
            if remain <= 0:
                break
            for c in cheap_rec(g, by, a, cache):
                if remain <= 0:
                    break
                if c in seen:
                    continue
                seen.add(c)
                r2.append(c)
                nxt.append(c)
                remain -= 1
        frontier = nxt
        if not nxt:
            break
    m1 = meets_of(g, by, r1, v, cache)
    m12 = meets_of(g, by, r1 + r2, v, cache)
    return dict(d1=len(r1), d2=len(r2), m1=m1, m2=max(m12 - m1, 0))


def eval_bin(g, by, vs):
    cache = {}
    rows = [walk(g, by, v, cache) for v in vs]
    n = len(rows)
    if not n:
        return dict(n=0, d1=0.0, d2=0.0, m1=0.0, m2=0.0)
    return dict(
        n=n,
        d1=sum(r["d1"] for r in rows) / n,
        d2=sum(r["d2"] for r in rows) / n,
        m1=sum(r["m1"] for r in rows) / n,
        m2=sum(r["m2"] for r in rows) / n,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=int, default=30_000_000)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=2)
    ap.add_argument("--window-lines", type=int, default=400)
    ap.add_argument("--lines", type=int, default=25000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--corpus", default="")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    path, kind, min_line = pick_corpus(args.corpus or None)
    text = path.open("r", encoding="utf-8", errors="ignore").read(args.bytes)
    all_lines = [l.strip() for l in text.split("\n") if len(l.strip()) >= min_line]
    pool = all_lines[: int(0.7 * len(all_lines))][: args.lines]
    rng = random.Random(args.seed)
    L = args.window_lines
    if L < len(pool):
        s0 = rng.randrange(len(pool) - L + 1)
        lines = pool[s0:s0 + L]
    else:
        lines = pool
    g = graph(lines, args.frame_max, args.min_fillers)
    if g is None:
        print("no tape")
        return 1
    by = mentions(g)
    dfn = {v: len(sl) for v, sl in by.items()}
    mid = [v for v, d in dfn.items() if 8 <= d <= 30]
    high = [v for v, d in dfn.items() if d > 80]
    mid_rep = eval_bin(g, by, mid)
    high_rep = eval_bin(g, by, high)
    void = mid_rep["n"] < 20 or high_rep["n"] < 5
    dd = mid_rep["d2"] - high_rep["d2"]
    dm = mid_rep["m2"] - high_rep["m2"]
    gate = (not void) and (dd > 0) and (dm > 0.05)
    rec = dict(seed=args.seed, corpus=kind, n_lines=len(lines),
               mid=mid_rep, high=high_rep, d_d2=dd, d_m2=dm,
               void=bool(void), gate=bool(gate))
    print(f"corpus {kind}  window {len(lines)}")
    print(f"MID  n {mid_rep['n']}  d1 {mid_rep['d1']:.2f} d2 {mid_rep['d2']:.2f}  "
          f"m1 {mid_rep['m1']:.3f} m2 {mid_rep['m2']:.3f}")
    print(f"HIGH n {high_rep['n']}  d1 {high_rep['d1']:.2f} d2 {high_rep['d2']:.2f}  "
          f"m1 {high_rep['m1']:.3f} m2 {high_rep['m2']:.3f}")
    print(f"Δd2 {dd:+.2f}  Δm2 {dm:+.3f}  VOID {void}  GATE {gate}")
    if void:
        print("\nVOID: not enough words.")
    elif gate:
        print("\nGO RING: second ring grows for mid words, not for and.")
    else:
        print("\nSTOP: ring2 does not belong to физика more than to and.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
