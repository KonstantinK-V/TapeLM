"""523: disagreement on a star. Depth only if halves agree."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from _audit511_ring import cheap_rec, graph, mentions, pick_corpus
from _audit517_window import comps
from _audit518_reldf import pct_band

OUT = Path("results/_stage523_disagree.json")


def rec_of(g, by, v, slots, cache):
    saved = by[v]
    by[v] = list(slots)
    cache.pop(v, None)
    out = [c for c in cheap_rec(g, by, v, cache) if c != v]
    by[v] = saved
    return out


def walk_nodes(g, by, v, cache, k):
    rec = rec_of(g, by, v, by[v], cache)
    allow = max(1, int(k * g["n"] / max(g["df"][v], 1)))
    r1, seen = [], {v}
    for c in rec:
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
    return r1 + r2


def mean_hit(rows):
    n = max(len(rows), 1)
    h1 = sum(r["h1"] for r in rows) / n
    w = sum(r["w"] for r in rows) / n
    return dict(n=len(rows), hop1=h1, walk=w, extra=w - h1,
                jacc=sum(r["j"] for r in rows) / n)


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
    mid, _high, p25, p75 = pct_band(g, by)
    k = 200.0 / max(g["n"], 1)
    cache = {}
    mix, con = [], []
    for v in mid:
        sl = list(by[v])
        if len(sl) < 9:
            continue
        rng.shuffle(sl)
        held_s, rest = sl[0], sl[1:]
        held = set(comps(g, held_s, v))
        if not held:
            continue
        half = len(rest) // 2
        A, B = rest[:half], rest[half:]
        if not A or not B:
            continue
        ra = set(rec_of(g, by, v, A, cache))
        rb = set(rec_of(g, by, v, B, cache))
        uni = ra | rb
        j = (len(ra & rb) / len(uni)) if uni else 0.0
        saved = by[v]
        by[v] = rest
        cache.pop(v, None)
        nodes = walk_nodes(g, by, v, cache, k)
        rec = rec_of(g, by, v, rest, cache)
        by[v] = saved
        if not rec or not nodes:
            continue
        row = dict(j=j, h1=int(rec[0] in held),
                   w=int(any(x in held for x in nodes)))
        if j < 0.15:
            mix.append(row)
        elif j >= 0.40:
            con.append(row)
    mx, cn = mean_hit(mix), mean_hit(con)
    void = mx["n"] < 20 or cn["n"] < 20
    gate = (not void) and (mx["extra"] <= 0.05) and (cn["extra"] > 0.05)
    rec = dict(seed=args.seed, corpus=kind, n_lines=len(lines), k=k,
               p25=p25, p75=p75, mix=mx, con=cn,
               void=bool(void), gate=bool(gate))
    print(f"corpus {kind}  window {len(lines)}  p25–75 {p25}-{p75}")
    print(f"MIX n {mx['n']} jacc {mx['jacc']:.3f}  hop1 {mx['hop1']:.3f}  "
          f"walk {mx['walk']:.3f}  extra {mx['extra']:+.3f}")
    print(f"CON n {cn['n']} jacc {cn['jacc']:.3f}  hop1 {cn['hop1']:.3f}  "
          f"walk {cn['walk']:.3f}  extra {cn['extra']:+.3f}")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("\nVOID: not enough mix/con stars.")
    elif gate:
        print("\nGO DISAGREE: depth only when halves agree; mixed extra is noise. Refuse ring2 on mix.")
    else:
        print("\nSTOP: mix/con extra do not split this way. Do not special-case disagreement yet.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
