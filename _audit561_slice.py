"""561: freeze 557 PIN; chooser only on refuse.

560 STOP: Q[n_cand,peaked,width] < always-557. It overrode unique PIN.

    n_cand==1  always P   (law 557, not learned)
    n_cand>=2  Q[(peaked,width)][S|R] on this slice only

Print first:
    u_S1  gold S among unique-PIN trials   (override leftover — not this gate)
    u_S2  gold S among refuse trials       (this arena)

VOID  n2 < 40 or u_S2 <= 0.05
GATE  overall hit - 557 > 0.05  and  slice-S - random > 0.05
      (PIN frozen ⇒ cannot go below 557 on unique)

    python _check561_slice.py
    python _audit561_slice.py --seed 1337 --corpus data/_tinystories_train.txt
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

from _audit560_choose import (
    hit_of, pick, policy_557, rows_of, train_q, windows,
)
from _audit511_ring import pick_corpus

OUT = Path("results/_stage561_slice.json")


def key2(row):
    return row["key"][1:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--bytes", type=int, default=40_000_000)
    ap.add_argument("--window-lines", type=int, default=400)
    ap.add_argument("--n-win", type=int, default=8)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=2)
    ap.add_argument("--lines", type=int, default=25000)
    ap.add_argument("--cap-probe", type=int, default=6)
    ap.add_argument("--corpus", default="")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    path, kind, min_line = pick_corpus(args.corpus or None)
    text = path.open("r", encoding="utf-8", errors="ignore").read(args.bytes)
    all_lines = [l.strip() for l in text.split("\n") if len(l.strip()) >= min_line]
    pool = all_lines[: int(0.7 * len(all_lines))][: args.lines]
    rng = random.Random(args.seed)
    t0 = time.time()
    print(f"561 freeze PIN; choose on refuse  corpus={path}  {kind}", flush=True)
    wins = windows(pool, args.n_win, args.window_lines, rng)
    n_tr = max(1, int(0.7 * len(wins)))
    train_rows, test_rows, sk = [], [], Counter()
    for i, lines in enumerate(wins):
        rs, sv = rows_of(lines, args, rng)
        sk.update(sv)
        (train_rows if i < n_tr else test_rows).extend(rs)
    te = test_rows
    n = len(te)
    u1 = [r for r in te if r["n_cand"] == 1]
    u2 = [r for r in te if r["n_cand"] != 1]
    n1, n2 = len(u1), len(u2)
    u_s1 = (sum(r["gold"] == "S" for r in u1) / n1) if n1 else 0.0
    u_s2 = (sum(r["gold"] == "S" for r in u2) / n2) if n2 else 0.0
    print(f"test {n}  unique {n1}  refuse {n2}  u_S1 {u_s1:.3f}  u_S2 {u_s2:.3f}  "
          f"skip {dict(sk)}", flush=True)
    void = n2 < 40 or u_s2 <= 0.05
    tr2 = [r for r in train_rows if r["n_cand"] != 1]
    qrows = []
    for r in tr2:
        rr = dict(r)
        rr["key"] = key2(r)
        qrows.append(rr)
    q = train_q(qrows)
    rng_te = random.Random(args.seed + 99)

    def chooser(r):
        if r["n_cand"] == 1:
            return "P"
        return pick(q, key2(r), rng_te)

    def mean(rows, fn):
        return sum(hit_of(r, fn(r)) for r in rows) / len(rows) if rows else 0.0

    h_l = mean(te, chooser)
    h_557 = mean(te, policy_557)
    rng_r = random.Random(args.seed + 7)
    h_rnd = mean(u2, lambda r: rng_r.choice(("S", "R")))
    h_s = mean(u2, chooser)
    d557 = h_l - h_557
    ds = h_s - h_rnd
    gate = (not void) and d557 > 0.05 and ds > 0.05
    print(f"learn {h_l:.4f}  557 {h_557:.4f}  d557 {d557:+.4f}")
    print(f"slice learn {h_s:.4f}  rnd {h_rnd:.4f}  dslice {ds:+.4f}")
    print(f"VOID {void}  GATE {gate}")
    if u_s1 > 0.05:
        print(f"OPEN leftover: unique PIN is wrong {u_s1:.3f} — not this gate.")
    if void:
        print("\nVOID: refuse slice has no STAR mass. 557 stands; override is leftover.")
    elif not gate:
        print("\nSTOP: on refuse, counts do not pick STAR above random.")
    else:
        print("\nGO SLICE. Frozen PIN + refuse-chooser beats 557.")
    rec = dict(seed=args.seed, corpus=kind, n_test=n, n1=n1, n2=n2,
               u_s1=u_s1, u_s2=u_s2, skip=dict(sk),
               h_learn=h_l, h_557=h_557, d557=d557,
               h_slice=h_s, h_rnd=h_rnd, dslice=ds,
               elapsed_s=round(time.time() - t0, 1),
               void=bool(void), gate=bool(gate))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
