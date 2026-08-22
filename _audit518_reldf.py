"""518: relative df. Was 516 the 200/df knob or long-wiki geometry?"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from _audit511_ring import cheap_rec, graph, mentions, pick_corpus

OUT = Path("results/_stage518_reldf.json")
SIZES = (100, 400, 1200, 2400)


def pct_band(g, by):
    dfn = {v: g["df"][v] for v in by if len(by[v]) >= 8 and v in g["df"]}
    if len(dfn) < 8:
        return [], [], 0, 0
    vals = sorted(dfn.values())
    p25 = vals[int(0.25 * (len(vals) - 1))]
    p75 = vals[int(0.75 * (len(vals) - 1))]
    mid = [v for v, d in dfn.items() if p25 <= d <= p75]
    high = [v for v, d in dfn.items() if d > p75]
    return mid, high, p25, p75


def walk_rel(g, by, v, cache, k):
    allow = max(1, int(k * g["n"] / max(g["df"][v], 1)))
    r1, seen = [], {v}
    for c in cheap_rec(g, by, v, cache):
        if len(r1) >= allow:
            break
        if c in seen:
            continue
        seen.add(c)
        r1.append(c)
    remain = allow - len(r1)
    r2, frontier = [], list(r1)
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
    return dict(d1=len(r1), d2=len(r2), allow=allow)


def eval_bin(g, by, vs, k):
    cache = {}
    rows = [walk_rel(g, by, v, cache, k) for v in vs]
    n = len(rows)
    if not n:
        return dict(n=0, d1=0.0, d2=0.0, allow=0.0)
    return dict(
        n=n,
        d1=sum(r["d1"] for r in rows) / n,
        d2=sum(r["d2"] for r in rows) / n,
        allow=sum(r["allow"] for r in rows) / n,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=int, default=30_000_000)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=2)
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
    L = max(SIZES)
    if L < len(pool):
        s0 = rng.randrange(len(pool) - L + 1)
        base = pool[s0:s0 + L]
    else:
        base = pool
    g400 = graph(base[: min(400, len(base))], args.frame_max, args.min_fillers)
    if g400 is None:
        print("no tape at 400")
        return 1
    k = 200.0 / max(g400["n"], 1)
    print(f"k {k:.6f}  n400 {g400['n']}  (N=400 allow == 200/df)")
    curve = {}
    for n in SIZES:
        lines = base[: min(n, len(base))]
        g = graph(lines, args.frame_max, args.min_fillers)
        if g is None:
            curve[str(n)] = dict(n_lines=len(lines), void=True)
            continue
        by = mentions(g)
        mid, high, p25, p75 = pct_band(g, by)
        mid_rep = eval_bin(g, by, mid, k)
        high_rep = eval_bin(g, by, high, k)
        curve[str(n)] = dict(n_lines=len(lines), p25=p25, p75=p75,
                             n_slots=g["n"], mid=mid_rep, high=high_rep)
        print(f"N {len(lines):4d}  p25–75 {p25}-{p75}  slots {g['n']}  "
              f"MID n {mid_rep['n']:4d} d1 {mid_rep['d1']:.2f} d2 {mid_rep['d2']:.2f} "
              f"allow {mid_rep['allow']:.1f}  "
              f"HIGH n {high_rep['n']:3d} d1 {high_rep['d1']:.2f} d2 {high_rep['d2']:.2f}")
    m100 = curve.get("100", {}).get("mid") or {}
    m2400 = curve.get("2400", {}).get("mid") or {}
    h2400 = curve.get("2400", {}).get("high") or {}
    void = m100.get("n", 0) < 10 or m2400.get("n", 0) < 20
    gate = (not void) and (m2400.get("d2", 0) > m100.get("d2", 0) + 1) and (
        h2400.get("d2", 99) < 1.0)
    rec = dict(seed=args.seed, corpus=kind, k=k, n400=g400["n"],
               sizes=list(SIZES), curve=curve, void=bool(void), gate=bool(gate))
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("\nVOID: not enough mid words on the curve.")
    elif gate:
        print("\nGO REL: mid ring2 grows with tape under n/df; high stays short. 516 was the absolute knob.")
    else:
        print("\nSTOP: relative df does not save depth. Geometry of long wiki + 1/df, not the constant 200.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
