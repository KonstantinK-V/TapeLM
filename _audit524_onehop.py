"""524: uncertainty inside one hop, not A∩B (523 STOP)."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from _audit511_ring import cheap_rec, graph, mentions, pick_corpus
from _audit517_window import comps
from _audit518_reldf import pct_band

OUT = Path("results/_stage524_onehop.json")


def rec_counts(g, by, v, slots, cache):
    saved = by[v]
    by[v] = list(slots)
    cache.pop(v, None)
    rec = [c for c in cheap_rec(g, by, v, cache) if c != v]
    cnt = Counter()
    for s in slots:
        cnt.update(set(comps(g, s, v)))
    by[v] = saved
    return rec, cnt


def mean_hit(rows):
    n = max(len(rows), 1)
    return dict(n=len(rows), hit=sum(rows) / n)


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
    cache = {}
    cert, unc = [], []
    for v in mid:
        sl = list(by[v])
        if len(sl) < 8:
            continue
        rng.shuffle(sl)
        held_s, rest = sl[0], sl[1:]
        held = set(comps(g, held_s, v))
        if not held or not rest:
            continue
        rec, cnt = rec_counts(g, by, v, rest, cache)
        if not rec:
            continue
        n0 = cnt.get(rec[0], 0)
        n1 = cnt.get(rec[1], 0) if len(rec) > 1 else 0
        peaked = (len(rec) == 1) or (n0 > 0 and n1 < 0.5 * n0)  # top-2 gap
        hit = int(rec[0] in held)
        if peaked:
            cert.append(hit)
        else:
            unc.append(hit)
    c, u = mean_hit(cert), mean_hit(unc)
    void = c["n"] < 20 or u["n"] < 20
    gate = (not void) and (c["hit"] - u["hit"] > 0.05)
    rec = dict(seed=args.seed, corpus=kind, n_lines=len(lines),
               p25=p25, p75=p75, cert=c, unc=u,
               void=bool(void), gate=bool(gate))
    print(f"corpus {kind}  window {len(lines)}  p25–75 {p25}-{p75}")
    print(f"CERT n {c['n']} hit {c['hit']:.3f}")
    print(f"UNC  n {u['n']} hit {u['hit']:.3f}  Δ {c['hit']-u['hit']:+.3f}")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("\nVOID: not enough peaked/tied hops.")
    elif gate:
        print("\nGO ONEHOP: peaked hop pins; tied top-2 do not. Enumerate, no pin.")
    else:
        print("\nSTOP: within-hop peak does not predict pin. Leave 511 order; no refuse-on-tie.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
