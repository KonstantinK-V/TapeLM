"""516: 511 star as the tape grows. Nested prefixes, not a new world."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from _audit511_ring import eval_bin, graph, mentions, pick_corpus

OUT = Path("results/_stage516_len.json")
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
    pool = all_lines[: int(0.7 * len(all_lines))][: args.lines]
    rng = random.Random(args.seed)
    L = max(SIZES)
    if L < len(pool):
        s0 = rng.randrange(len(pool) - L + 1)
        base = pool[s0:s0 + L]
    else:
        base = pool
    curve = {}
    for n in SIZES:
        lines = base[: min(n, len(base))]
        g = graph(lines, args.frame_max, args.min_fillers)
        if g is None:
            curve[str(n)] = dict(n_lines=len(lines), void=True)
            continue
        by = mentions(g)
        dfn = {v: len(sl) for v, sl in by.items()}
        mid = [v for v, d in dfn.items() if 8 <= d <= 30]
        high = [v for v, d in dfn.items() if d > 80]
        mid_rep = eval_bin(g, by, mid)
        high_rep = eval_bin(g, by, high)
        curve[str(n)] = dict(n_lines=len(lines), mid=mid_rep, high=high_rep)
        print(f"N {len(lines):4d}  MID n {mid_rep['n']:4d} d1 {mid_rep['d1']:.2f} "
              f"d2 {mid_rep['d2']:.2f} m2 {mid_rep['m2']:.3f}  "
              f"HIGH n {high_rep['n']:3d} d1 {high_rep['d1']:.2f} "
              f"d2 {high_rep['d2']:.2f}")
    m100 = curve.get("100", {}).get("mid") or {}
    m2400 = curve.get("2400", {}).get("mid") or {}
    h2400 = curve.get("2400", {}).get("high") or {}
    void = m100.get("n", 0) < 10 or m2400.get("n", 0) < 20
    gate = (not void) and (m2400.get("d2", 0) > m100.get("d2", 0) + 1) and (
        h2400.get("d2", 99) < 1.0)
    rec = dict(seed=args.seed, corpus=kind, sizes=list(SIZES), curve=curve,
               void=bool(void), gate=bool(gate))
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("\nVOID: not enough mid words on the curve.")
    elif gate:
        print("\nGO LEN: mid ring2 grows with tape; and stays short.")
    else:
        print("\nSTOP: longer tape does not grow the mid walk, or and starts to flood.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
