"""535: recurrent marks only, hop1 frozen 511.

Keep (v,c) if residual on >=2 train windows (506).
Test offer: hop1 = 511 hop1; remaining allow = recurrent marks
that sit on this rec, then 511 tail + ring2.
Held never enters the offer.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from _audit511_ring import cheap_rec, mentions, pick_corpus
from _audit518_reldf import pct_band
from _audit527_learn import allow_of, majority, v1_nodes
from _audit528_step import cover, trials
from _audit532_pool import slice_graph

OUT = Path("results/_stage535_recur.json")


def offer_h1frozen(g, by, v, cache, k, high_set, marked):
    rec = [c for c in cheap_rec(g, by, v, cache) if c != v]
    rec_set = set(rec)
    h1 = rec[0] if rec else None
    extra = [c for c in marked if c in rec_set and c != h1]
    tail = [c for c in rec if c != h1 and c not in extra]
    rec2 = ([h1] if h1 else []) + extra + tail
    allow = allow_of(g, v, k, high_set)
    r1, seen = [], {v}
    for c in rec2:
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
    ap.add_argument("--min-recur", type=int, default=2)
    ap.add_argument("--corpus", default="")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    path, kind, min_line = pick_corpus(args.corpus or None)
    text = path.open("r", encoding="utf-8", errors="ignore").read(args.bytes)
    all_lines = [l.strip() for l in text.split("\n") if len(l.strip()) >= min_line]
    pool = all_lines[: args.lines]
    rng = random.Random(args.seed)
    graphs = []
    for _ in range(args.windows):
        g, _nL = slice_graph(pool, args.window_lines, rng, args.frame_max, args.min_fillers)
        if g is not None:
            graphs.append(g)
    if len(graphs) < 4:
        print("too few windows")
        return 1
    n_tr = max(2, int(0.7 * args.windows))
    train_g, test_g = graphs[:n_tr], graphs[n_tr:]
    pair = Counter()
    for g in train_g:
        by = mentions(g)
        mid, high, _, _ = pct_band(g, by)
        k = 200.0 / max(g["n"], 1)
        high_set = set(high)
        cache = {}
        vs = list(mid) + list(high)
        got = set()
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
                    got.add((v, c))
                    seen.add(c)
        for vc in got:
            pair[vc] += 1
    recur = defaultdict(list)
    n_r = 0
    for (v, c), n in pair.items():
        if n >= args.min_recur:
            recur[v].append(c)
            n_r += 1

    def collect(gs):
        rows = []
        rr = random.Random(args.seed + 19)
        for g in gs:
            by = mentions(g)
            mid, high, _, _ = pct_band(g, by)
            k = 200.0 / max(g["n"], 1)
            high_set = set(high)
            c511, cm = {}, {}
            for v, rest, held, maj in trials(g, by, mid, rr):
                saved = by[v]
                by[v] = rest
                c511.pop(v, None)
                cm.pop(v, None)
                n511 = v1_nodes(g, by, v, c511, k, high_set)
                nm = offer_h1frozen(g, by, v, cm, k, high_set, recur.get(v, ()))
                by[v] = saved
                h1_same = int(bool(n511) and bool(nm) and n511[0] == nm[0])
                rows.append(dict(
                    hop1_511=cover(n511[:1], held) if n511 else 0.0,
                    hop1_m=cover(nm[:1], held) if nm else 0.0,
                    all_511=cover(n511, held),
                    all_m=cover(nm, held),
                    hops_511=len(n511), hops_m=len(nm),
                    extra=int(any(c not in n511 for c in nm)),
                    h1_same=h1_same,
                ))
        n = max(len(rows), 1)

        def avg(key):
            return sum(r[key] for r in rows) / n

        return dict(n=len(rows), hop1_511=avg("hop1_511"), hop1_m=avg("hop1_m"),
                    all_511=avg("all_511"), all_m=avg("all_m"),
                    hops_511=avg("hops_511"), hops_m=avg("hops_m"),
                    extra=avg("extra"), h1_same=avg("h1_same"))

    te = collect(test_g)
    void = te["n"] < 20 or n_r < 10
    d1 = te["hop1_m"] - te["hop1_511"]
    dA = te["all_m"] - te["all_511"]
    gate = (not void) and (te["extra"] > 0.05) and (dA > 0.05)
    rec = dict(seed=args.seed, corpus=kind, n_recur=n_r, n_pair=len(pair),
               test=te, d_hop1=d1, d_allgo=dA, void=bool(void), gate=bool(gate))
    print(f"corpus {kind}  recur marks {n_r} / pairs {len(pair)}")
    print(f"TEST n {te['n']}  extra {te['extra']:.3f}  hop1 frozen {te['h1_same']:.3f}")
    print(f"HOP1  511 {te['hop1_511']:.3f}  mark {te['hop1_m']:.3f}  Δ {d1:+.3f}")
    print(f"ALLGO 511 {te['all_511']:.3f}  mark {te['all_m']:.3f}  Δ {dA:+.3f}")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("\nVOID: too few recurrent marks.")
    elif gate:
        print("\nGO RECUR: stable marks opened allow and raised cover. 529 kept.")
    else:
        print("\nSTOP: recurrent marks did not lift allgo cover. 529 kept.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
