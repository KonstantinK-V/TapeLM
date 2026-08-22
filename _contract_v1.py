"""TapeLM contract v1 — the frozen walk. Import this; do not reopen 514–525 as levers.

    from _contract_v1 import walk_v1, bands_v1, k_from_home, LAW

    python _contract_v1.py --seed 1337
    python _contract_v1.py --corpus path/to/other.txt --seed 1337
    python _contract_v1.py --home-corpus data/_wikitext103_train.txt --corpus data/_stage254_news.txt --seed 1337
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from _audit511_ring import graph, mentions, pick_corpus
from _audit518_reldf import pct_band
from _audit519_highcap import eval_cap, walk_cap

LAW = Path("_CONTRACT_V1.txt").read_text(encoding="utf-8") if Path("_CONTRACT_V1.txt").exists() else ""
OUT = Path("results/_contract_v1_smoke.json")


def corpus_min_line(path: Path) -> int:
    return 80 if "wiki" in path.name.lower() else 20


def k_from_home(g400) -> float:
    return 200.0 / max(g400["n"], 1)


def calibrate_k(path, min_line, frame_max, min_fillers, lines, seed, nbytes):
    text = path.open("r", encoding="utf-8", errors="ignore").read(nbytes)
    all_lines = [l.strip() for l in text.split("\n") if len(l.strip()) >= min_line]
    pool = all_lines[: int(0.7 * len(all_lines))][:lines]
    if len(pool) < 400:
        return None, None
    rng = random.Random(seed)
    s_h = rng.randrange(len(pool) - 400 + 1)
    g400 = graph(pool[s_h:s_h + 400], frame_max, min_fillers)
    if g400 is None:
        return None, None
    return k_from_home(g400), g400["n"]


def load_lines(path, min_line, lines, seed, window, nbytes):
    text = path.open("r", encoding="utf-8", errors="ignore").read(nbytes)
    all_lines = [l.strip() for l in text.split("\n") if len(l.strip()) >= min_line]
    pool = all_lines[: int(0.7 * len(all_lines))][:lines]
    rng = random.Random(seed)
    if window < len(pool):
        s0 = rng.randrange(len(pool) - window + 1)
        return pool[s0:s0 + window]
    return pool


def bands_v1(g, by):
    mid, high, p25, p75 = pct_band(g, by)
    return mid, high, p25, p75


def walk_v1(g, by, v, cache, k, high_set):
    """519 walk: relative mid budget, high allow=1. No peaked pin, no confirm, no W."""
    return walk_cap(g, by, v, cache, k, high_set)


def report(g, by, k):
    mid, high, p25, p75 = bands_v1(g, by)
    high_set = set(high)
    mid_rep = eval_cap(g, by, mid, k, high_set)
    high_rep = eval_cap(g, by, high, k, high_set)
    return dict(p25=p25, p75=p75, n_slots=g["n"], k=k, mid=mid_rep, high=high_rep)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=int, default=30_000_000)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=2)
    ap.add_argument("--window-lines", type=int, default=400)
    ap.add_argument("--lines", type=int, default=25000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--corpus", default="")
    ap.add_argument("--home-corpus", default="")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    print(LAW)
    print("--- v1 smoke (519 walk only) ---")
    walk_path, kind, _ = pick_corpus(args.corpus or None)
    walk_min = corpus_min_line(walk_path)
    if args.home_corpus:
        home_path = Path(args.home_corpus)
        if not home_path.exists():
            print(f"no home corpus {home_path}")
            return 1
        home_min = corpus_min_line(home_path)
        k, n400 = calibrate_k(home_path, home_min, args.frame_max, args.min_fillers,
                              args.lines, args.seed, args.bytes)
        if k is None:
            print("no home tape at 400")
            return 1
        home_kind = "wiki" if "wiki" in home_path.name.lower() else "given"
        print(f"HOME {home_kind} k {k:.6f} n400 {n400}")
    else:
        k = n400 = None
    lines = load_lines(walk_path, walk_min, args.lines, args.seed,
                       args.window_lines, args.bytes)
    g = graph(lines, args.frame_max, args.min_fillers)
    if g is None:
        print("no tape")
        return 1
    if k is None:
        k = k_from_home(g)
        n400 = g["n"]
    by = mentions(g)
    rec = report(g, by, k)
    rec.update(seed=args.seed, corpus=kind, n_lines=len(lines), path=str(walk_path),
               k=k, n400=n400, home=str(args.home_corpus) if args.home_corpus else "")
    print(f"corpus {kind}  lines {len(lines)}  k {k:.6f}  p25–75 {rec['p25']}-{rec['p75']}")
    print(f"MID  n {rec['mid']['n']} d1 {rec['mid']['d1']:.2f} d2 {rec['mid']['d2']:.2f} "
          f"allow {rec['mid']['allow']:.1f}")
    print(f"HIGH n {rec['high']['n']} d1 {rec['high']['d1']:.2f} d2 {rec['high']['d2']:.2f} "
          f"allow {rec['high']['allow']:.1f}")
    print("high d2 must stay 0 (contract). peaked/confirm/W not in this walk.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    key = f"{walk_path.stem}_{args.seed}" if args.corpus or args.home_corpus else str(args.seed)
    prev[key] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
