"""526: v1 walk on foreign file, wiki home k, curve through 4800+."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from _audit511_ring import graph, mentions
from _audit518_reldf import pct_band
from _audit519_highcap import eval_cap
from _contract_v1 import calibrate_k, corpus_min_line, k_from_home

OUT = Path("results/_stage526_openxfer.json")
SIZES = (100, 400, 1200, 2400, 4800, 9600)
WIKI = Path("data/_wikitext103_train.txt")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=int, default=30_000_000)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=2)
    ap.add_argument("--lines", type=int, default=25000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--home-corpus", default=str(WIKI))
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    home_path = Path(args.home_corpus)
    fore_path = Path(args.corpus)
    if not home_path.exists() or not fore_path.exists():
        print("missing home or foreign corpus")
        return 1
    home_min = corpus_min_line(home_path)
    fore_min = corpus_min_line(fore_path)
    k, n400 = calibrate_k(home_path, home_min, args.frame_max, args.min_fillers,
                            args.lines, args.seed, args.bytes)
    if k is None:
        print("no home tape at 400")
        return 1
    text = fore_path.open("r", encoding="utf-8", errors="ignore").read(args.bytes)
    all_lines = [l.strip() for l in text.split("\n") if len(l.strip()) >= fore_min]
    pool = all_lines[: int(0.7 * len(all_lines))][: args.lines]
    rng = random.Random(args.seed)
    L = max(SIZES)
    if L < len(pool):
        s0 = rng.randrange(len(pool) - L + 1)
        base = pool[s0:s0 + L]
    else:
        base = pool
    sizes = tuple(n for n in SIZES if n <= len(base))
    if len(base) > max(SIZES):
        sizes = sizes + (len(base),)
    home_tag = "wiki" if "wiki" in home_path.name.lower() else home_path.name
    fore_tag = "wiki" if "wiki" in fore_path.name.lower() else fore_path.name
    print(f"HOME {home_tag} k {k:.6f} n400 {n400}  FORE {fore_tag} lines {len(base)}  sizes {sizes}")
    curve = {}
    for n in sizes:
        lines = base[:n]
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
        print(f"FORE N {len(lines):5d}  p25–75 {p25}-{p75}  "
              f"MID n {mid_rep['n']:4d} d2 {mid_rep['d2']:.2f} allow {mid_rep['allow']:.1f}  "
              f"HIGH n {high_rep['n']:3d} d2 {high_rep['d2']:.2f}")
    m100 = curve.get("100", {}).get("mid") or {}
    last_n = str(sizes[-1])
    mlast = curve.get(last_n, {}).get("mid") or {}
    hlast = curve.get(last_n, {}).get("high") or {}
    void = m100.get("n", 0) < 10 or mlast.get("n", 0) < 20 or len(base) < 400
    gate = (not void) and (mlast.get("d2", 0) > m100.get("d2", 0) + 1) and (
        hlast.get("d2", 99) < 1.0)
    rec = dict(seed=args.seed, home=str(home_path), foreign=str(fore_path),
               k=k, n400=n400, n_fore=len(base), sizes=list(sizes),
               curve=curve, void=bool(void), gate=bool(gate))
    print(f"VOID {void}  GATE {gate}  (last N {last_n})")
    if void:
        print("\nVOID: foreign tape too thin.")
    elif gate:
        print("\nGO OPENXFER: v1 walk on foreign file; mid grows; high capped.")
    else:
        print("\nSTOP: foreign slice does not replicate v1 length law.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    key = f"{fore_path.stem}_{args.seed}"
    prev[key] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
