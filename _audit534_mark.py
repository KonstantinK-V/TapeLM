"""534: end-cycle mark moves the offer. Not Φ, not 533 pool."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from _audit511_ring import cheap_rec, mentions, pick_corpus
from _audit518_reldf import pct_band
from _audit527_learn import allow_of, majority, v1_nodes
from _audit528_step import cover, trials
from _audit532_pool import slice_graph

OUT = Path("results/_stage534_mark.json")


def offer(g, by, v, cache, k, high_set, marked):
    rec = [c for c in cheap_rec(g, by, v, cache) if c != v]
    rec_set = set(rec)
    first = [c for c in marked if c in rec_set]
    rec = first + [c for c in rec if c not in first]
    allow = allow_of(g, v, k, high_set)
    r1, seen = [], {v}
    for c in rec:
        if len(r1) >= allow:
            break
        if c in seen:
            continue
        seen.add(c)
        r1.append(c)
    if v in high_set:
        return r1
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
    return r1 + r2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=int, default=80_000_000)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=2)
    ap.add_argument("--window-lines", type=int, default=250)
    ap.add_argument("--windows", type=int, default=16)
    ap.add_argument("--lines", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--corpus", default="")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    path, kind, min_line = pick_corpus(args.corpus or None)
    text = path.open("r", encoding="utf-8", errors="ignore").read(args.bytes)
    all_lines = [l.strip() for l in text.split("\n") if len(l.strip()) >= min_line]
    pool = all_lines[: args.lines]
    rng = random.Random(args.seed)
    L = args.window_lines
    graphs = []
    for _ in range(args.windows):
        g, nL = slice_graph(pool, L, rng, args.frame_max, args.min_fillers)
        if g is None:
            continue
        graphs.append(g)
    if len(graphs) < 4:
        print("too few windows")
        return 1
    n_tr = max(2, int(0.7 * args.windows))
    train_g, test_g = graphs[:n_tr], graphs[n_tr:]
    marks = defaultdict(list)
    seen_m = defaultdict(set)
    n_mark = 0
    for g in train_g:
        by = mentions(g)
        mid, high, _, _ = pct_band(g, by)
        k = 200.0 / max(g["n"], 1)
        high_set = set(high)
        cache = {}
        vs = list(mid) + list(high)
        for v, rest, held, maj in trials(g, by, vs, rng):
            saved = by[v]
            by[v] = rest
            cache.pop(v, None)
            nodes = v1_nodes(g, by, v, cache, k, high_set)
            by[v] = saved
            if not nodes:
                continue
            hop1 = nodes[0]
            seen = {hop1} if hop1 in held else set()
            for c in nodes[1:]:
                if c in held and c not in seen and c != maj:
                    if c not in seen_m[v]:
                        seen_m[v].add(c)
                        marks[v].append(c)
                        n_mark += 1
                    seen.add(c)

    def collect(gs):
        rows = []
        rr = random.Random(args.seed + 17)
        for g in gs:
            by = mentions(g)
            mid, high, _, _ = pct_band(g, by)
            k = 200.0 / max(g["n"], 1)
            high_set = set(high)
            cache511, cache_m = {}, {}
            for v, rest, held, maj in trials(g, by, mid, rr):
                saved = by[v]
                by[v] = rest
                cache511.pop(v, None)
                cache_m.pop(v, None)
                n511 = v1_nodes(g, by, v, cache511, k, high_set)
                nm = offer(g, by, v, cache_m, k, high_set, marks.get(v, ()))
                by[v] = saved
                rows.append(dict(
                    hop1_511=cover(n511[:1], held) if n511 else 0.0,
                    hop1_m=cover(nm[:1], held) if nm else 0.0,
                    all_511=cover(n511, held),
                    all_m=cover(nm, held),
                    hops_511=len(n511),
                    hops_m=len(nm),
                    extra=int(any(c not in n511 for c in nm)),
                ))
        n = max(len(rows), 1)

        def avg(key):
            return sum(r[key] for r in rows) / n

        return dict(n=len(rows), hop1_511=avg("hop1_511"), hop1_m=avg("hop1_m"),
                    all_511=avg("all_511"), all_m=avg("all_m"),
                    hops_511=avg("hops_511"), hops_m=avg("hops_m"),
                    extra=avg("extra"))

    te = collect(test_g)
    void = te["n"] < 20 or n_mark < 20
    d1 = te["hop1_m"] - te["hop1_511"]
    dA = te["all_m"] - te["all_511"]
    gate = (not void) and ((d1 > 0.05) or (dA > 0.05))
    rec = dict(seed=args.seed, corpus=kind, windows=len(graphs),
               n_train=len(train_g), n_test=len(test_g), n_mark=n_mark,
               n_v=len(marks), test=te, d_hop1=d1, d_allgo=dA,
               void=bool(void), gate=bool(gate))
    print(f"corpus {kind}  W {L}  windows {len(graphs)}  marks {n_mark} on {len(marks)} v")
    print(f"TEST n {te['n']}  extra-in-offer {te['extra']:.3f}")
    print(f"HOP1  511 {te['hop1_511']:.3f}  mark {te['hop1_m']:.3f}  Δ {d1:+.3f}")
    print(f"ALLGO 511 {te['all_511']:.3f}  mark {te['all_m']:.3f}  Δ {dA:+.3f}")
    print(f"hops  511 {te['hops_511']:.2f}  mark {te['hops_m']:.2f}")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("\nVOID: few marks or tiny test.")
    elif gate:
        print("\nGO MARK: offer moved. Iteration can open/rerank paths. 529 kept.")
    else:
        print("\nSTOP: marks did not beat frozen 511 offer. Keep 529; chance not used yet.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
