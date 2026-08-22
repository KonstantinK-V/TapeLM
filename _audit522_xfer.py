"""522: 519 walk on a foreign long tape. No new levers."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from _audit511_ring import graph, mentions, pick_corpus
from _audit518_reldf import pct_band
from _audit519_highcap import eval_cap

OUT = Path("results/_stage522_xfer.json")
SIZES = (100, 400, 1200, 2400)


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
    cut = int(0.7 * len(all_lines))
    home_pool = all_lines[:cut][: args.lines]
    fore_pool = all_lines[cut:][: args.lines]
    rng = random.Random(args.seed)
    if len(home_pool) < 400:
        print("home too short")
        return 1
    s_h = rng.randrange(len(home_pool) - 400 + 1)
    g400 = graph(home_pool[s_h:s_h + 400], args.frame_max, args.min_fillers)
    if g400 is None:
        print("no home tape")
        return 1
    k = 200.0 / max(g400["n"], 1)
    L = max(SIZES)
    if L < len(fore_pool):
        s0 = rng.randrange(len(fore_pool) - L + 1)
        base = fore_pool[s0:s0 + L]
    else:
        base = fore_pool
    print(f"HOME {kind} k {k:.6f} n400 {g400['n']}  FORE lines {len(base)}")
    curve = {}
    for n in SIZES:
        lines = base[: min(n, len(base))]
        g = graph(lines, args.frame_max, args.min_fillers)
        if g is None:
            curve[str(n)] = dict(n_lines=len(lines), void=True)
            continue
        by = mentions(g)
        mid, high, p25, p75 = pct_band(g, by)
        high_set = set(high)
        mid_rep = eval_cap(g, by, mid, k, high_set)
        high_rep = eval_cap(g, by, high, k, high_set)
        curve[str(n)] = dict(n_lines=len(lines), p25=p25, p75=p75,
                             n_slots=g["n"], mid=mid_rep, high=high_rep)
        print(f"FORE N {len(lines):4d}  p25–75 {p25}-{p75}  "
              f"MID n {mid_rep['n']:4d} d1 {mid_rep['d1']:.2f} d2 {mid_rep['d2']:.2f} "
              f"allow {mid_rep['allow']:.1f}  "
              f"HIGH n {high_rep['n']:3d} d1 {high_rep['d1']:.2f} d2 {high_rep['d2']:.2f}")
    m100 = curve.get("100", {}).get("mid") or {}
    m2400 = curve.get("2400", {}).get("mid") or {}
    h2400 = curve.get("2400", {}).get("high") or {}
    void = m100.get("n", 0) < 10 or m2400.get("n", 0) < 20 or len(base) < 400
    gate = (not void) and (m2400.get("d2", 0) > m100.get("d2", 0) + 1) and (
        h2400.get("d2", 99) < 1.0)
    rec = dict(seed=args.seed, corpus=kind, k=k, n400=g400["n"],
               n_fore=len(base), sizes=list(SIZES), curve=curve,
               void=bool(void), gate=bool(gate))
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("\nVOID: foreign tape too thin.")
    elif gate:
        print("\nGO XFER: 519 on unseen lines still grows mid and caps and.")
    else:
        print("\nSTOP: 519 does not replicate on the foreign slice.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
