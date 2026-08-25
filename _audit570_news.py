"""570: same PLACE-WALK v1 on a foreign corpus. No new rule.

Smoke: one_query from _place_walk. News first; --corpus overrides.

VOID  n < 40
GATE  pin>0 and refuse>0  — machine runs, not a mind GO.

Print n_cand 1/2/3+ so 10× later is comparable.

    python _check570_news.py
    python _audit570_news.py --seed 1337 --corpus data/_stage254_news.txt
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

from _audit511_ring import graph, mentions, pick_corpus
from _audit518_reldf import pct_band
from _place_walk import one_query, slot_lines

OUT = Path("results/_stage570_news.json")


def windows(pool, n_win, L, rng):
    out = []
    for _ in range(n_win):
        if len(pool) <= L:
            out.append(pool)
        else:
            out.append(pool[rng.randrange(len(pool) - L + 1):][:L])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--bytes", type=int, default=40_000_000)
    ap.add_argument("--window-lines", type=int, default=400)
    ap.add_argument("--n-win", type=int, default=8)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=2)
    ap.add_argument("--lines", type=int, default=25000)
    ap.add_argument("--cap-probe", type=int, default=4)
    ap.add_argument("--corpus", default="data/_stage254_news.txt")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    path, kind, min_line = pick_corpus(args.corpus or None)
    text = path.open("r", encoding="utf-8", errors="ignore").read(args.bytes)
    all_lines = [l.strip() for l in text.split("\n") if len(l.strip()) >= min_line]
    pool = all_lines[: int(0.7 * len(all_lines))][: args.lines]
    rng = random.Random(args.seed)
    t0 = time.time()
    print(f"570 v1-smoke  corpus={path}  {kind}  pool={len(pool)}", flush=True)
    st = Counter()
    W_e = {}
    n_win_ok = 0
    for lines in windows(pool, args.n_win, args.window_lines, rng):
        g = graph(lines, args.frame_max, args.min_fillers)
        if g is None:
            continue
        n_win_ok += 1
        slines = slot_lines(lines, args.frame_max, args.min_fillers)
        by = mentions(g)
        mid, high, _a, _b = pct_band(g, by)
        mid_set, high_set = set(mid), set(high)
        keys = list(mid)
        rng.shuffle(keys)
        for v in keys:
            sl = list(by[v])
            if len(sl) < 4:
                continue
            rng.shuffle(sl)
            for s in sl[: args.cap_probe]:
                row = one_query(g, by, v, s, mid_set, high_set, lines, slines, W_e=W_e)
                if row is None:
                    continue
                st["n"] += 1
                st["hit1"] += int(row["hit1"])
                st["pin"] += int(row["hop"].startswith("PIN"))
                st["refuse"] += int(row["hop"] == "REFUSE")
                st["hit2"] += int(row["hit2"])
                st["reuse"] += int(row.get("reuse") or 0)
                k = row.get("n_cand", -1)
                st[f"c{k}"] += 1
    n = st["n"]
    void = n < 40
    pin_r = st["pin"] / n if n else 0.0
    ref_r = st["refuse"] / n if n else 0.0
    gate = (not void) and pin_r > 0 and ref_r > 0
    print(f"windows {n_win_ok}  n={n}  hop1={st['hit1']/n if n else 0:.3f}  "
          f"pin={pin_r:.3f}  refuse={ref_r:.3f}  hop2={st['hit2']/n if n else 0:.3f}  "
          f"reuse={st['reuse']}")
    print(f"n_cand 1={st['c1']}  2={st['c2']}  3+={st['c3']+st['c4']+st['c5']}  "
          f"other={st['c0']}")
    print(f"VOID {void}  GATE {gate}  (smoke: machine runs, not a mind GO)")
    rec = dict(seed=args.seed, corpus=kind, path=str(path), n=n, n_win=n_win_ok,
               hop1=st["hit1"] / n if n else 0, pin=pin_r, refuse=ref_r,
               hop2=st["hit2"] / n if n else 0, reuse=st["reuse"],
               c1=st["c1"], c2=st["c2"], c3p=st["c3"] + st["c4"] + st["c5"],
               elapsed_s=round(time.time() - t0, 1),
               void=bool(void), gate=bool(gate))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    prev[f"{args.seed}_{kind}"] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
