"""590: same v1 walk as 589, foreign tape. No new lever. No Q.

GATE  hop1 PMI-rnd > 0.05  AND  hop2 PMI-rnd > 0.05  AND  n2_pmi >= 40
hop3  printed; VOID3 if n3_pmi < 40 — not the main gate

    python _check590_xfer.py
    python _audit590_xfer.py --seed 1337 --corpus data/_stage254_news.txt
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from _audit511_ring import pick_corpus
from _audit589_hop3 import collect, prefix_windows, score_eps

OUT = Path("results/_stage590_xfer.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--bytes", type=int, default=80_000_000)
    ap.add_argument("--window-lines", type=int, default=400)
    ap.add_argument("--n-win", type=int, default=80)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=2)
    ap.add_argument("--lines", type=int, default=120000)
    ap.add_argument("--cap-probe", type=int, default=4)
    ap.add_argument("--corpus", default="data/_stage254_news.txt")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    path, kind, min_line = pick_corpus(args.corpus or None)
    text = path.open("r", encoding="utf-8", errors="ignore").read(args.bytes)
    all_lines = [ln.strip() for ln in text.split("\n") if len(ln.strip()) >= min_line]
    pool = all_lines[: int(0.7 * len(all_lines))][: args.lines]
    windows = prefix_windows(pool, args.window_lines, args.n_win)
    rng = random.Random(args.seed)
    t0 = time.time()
    print(f"590 xfer  {path}  {kind}  windows={len(windows)}", flush=True)

    eps = []
    for lines in windows:
        eps.extend(collect(lines, args, rng))
    ev = score_eps(eps, random.Random(args.seed + 17))
    if ev is None:
        print("VOID: no episodes")
        return 0

    d1 = ev["fill1_pmi"] - ev["fill1_rnd"]
    d2 = ev["fill2_pmi"] - ev["fill2_rnd"]
    d3 = ev["fill3_pmi"] - ev["fill3_rnd"]
    void = ev["n"] < 40 or ev["n2_pmi"] < 40
    void3 = ev["n3_pmi"] < 40 or ev["cover3"] < 0.15
    gate = (not void) and d1 > 0.05 and d2 > 0.05
    print(
        f"hop1 n {ev['n']}  PMI {ev['fill1_pmi']:.3f}  rnd {ev['fill1_rnd']:.3f}  "
        f"d {d1:+.3f}  r1 {ev['r1']:.3f}"
    )
    print(
        f"hop2 n_pmi {ev['n2_pmi']}  PMI {ev['fill2_pmi']:.3f}  rnd {ev['fill2_rnd']:.3f}  "
        f"d {d2:+.3f}  r2 {ev['r2']:.3f}"
    )
    print(
        f"hop3 n_pmi {ev['n3_pmi']}  PMI {ev['fill3_pmi']:.3f}  rnd {ev['fill3_rnd']:.3f}  "
        f"d {d3:+.3f}  VOID3 {void3}"
    )
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("VOID: foreign tape has no hop1/hop2 mass.")
    elif gate:
        print("GO XFER: v1 walk lives on this tape. hop3 diagnostic only.")
    else:
        print("STOP: stories chain does not transfer.")

    rec = dict(
        seed=args.seed, corpus=kind, path=str(path),
        n_win=len(windows), n_eps=len(eps),
        elapsed_s=round(time.time() - t0, 1),
        void=bool(void), void3=bool(void3), gate=bool(gate),
        d1=d1, d2=d2, d3=d3,
        **ev,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    prev[f"{args.seed}_{path.stem}"] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
