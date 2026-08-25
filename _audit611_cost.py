"""611: priced SEARCH. Same F-order as 609. No Q.

610 read hit - 0.05·k and overcharged hits that stop early.
Here a trial that hits at t pays t READs, a miss pays min(k, |P|).

F  mention order, stop on first extract==held
MAJ bag majority, 0 READs

c ∈ {0, 0.03, 0.05, 0.08}   0.05 is the gate cost (521)
net(k,c) = hit(k) - c · mean_reads(k)
k* = argmax_{k in KS} net(k, 0.05)

VOID   n < 40
EDGE   k* is 1 or 8
GATE   k* in {2,3,4,6}
DIAG   net(k*,0.05) - MAJ   (not a gate; c is a ruler)

608/609/610 not retrained.

    python _check611_cost.py
    python _audit611_cost.py --seed 1337 --corpus data/_tinystories_train.txt
    python _audit611_cost.py --seed 8642 --corpus data/_tinystories_train.txt
    python _audit611_cost.py --seed 2890 --corpus data/_tinystories_train.txt
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

from _audit511_ring import pick_corpus
from _audit589_hop3 import prefix_windows
from _audit606_bridge import collect
from _audit609_kcurve import KS

OUT = Path("results/_stage611_cost.json")
CS = (0.0, 0.03, 0.05, 0.08)
INTERIOR = {2, 3, 4, 6}


def first_hit(order, held):
    for i, pl in enumerate(order, 1):
        if pl["extract"] == held:
            return i
    return None


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
    ap.add_argument("--corpus", default="")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    path, kind, min_line = pick_corpus(args.corpus or None)
    text = path.open("r", encoding="utf-8", errors="ignore").read(args.bytes)
    all_lines = [ln.strip() for ln in text.split("\n") if len(ln.strip()) >= min_line]
    pool = all_lines[: int(0.7 * len(all_lines))][: args.lines]
    windows = prefix_windows(pool, args.window_lines, args.n_win)
    rng = random.Random(args.seed)
    t0 = time.time()
    print(f"611 cost  {path}  {kind}  windows={len(windows)}", flush=True)

    n = m = 0
    hits = {k: 0 for k in KS}
    reads = {k: 0 for k in KS}
    for lines in windows:
        for row in collect(lines, args, rng):
            n += 1
            held, places, bag0 = row["held"], row["places"], row["bag0"]
            maj = Counter(bag0).most_common(1)[0][0]
            m += int(maj == held)
            t = first_hit(places, held)
            npl = len(places)
            for k in KS:
                if t is not None and t <= k:
                    hits[k] += 1
                    reads[k] += t
                else:
                    reads[k] += min(k, npl)

    def r(x):
        return x / n if n else 0.0

    fm = r(m)
    hit = [r(hits[k]) for k in KS]
    mean_r = [r(reads[k]) for k in KS]
    nets = {
        str(c): [hit[i] - c * mean_r[i] for i in range(len(KS))]
        for c in CS
    }
    net05 = nets["0.05"]
    i_star = max(range(len(KS)), key=lambda i: net05[i])
    k_star = KS[i_star]
    void = n < 40
    edge = (not void) and k_star in (1, 8)
    gate = (not void) and k_star in INTERIOR
    d_maj = net05[i_star] - fm
    print(f"n {n}  bagMAJ {fm:.3f}  k* {k_star}")
    print("k        " + "  ".join(f"{k:>6}" for k in KS))
    print("hit      " + "  ".join(f"{v:6.3f}" for v in hit))
    print("reads    " + "  ".join(f"{v:6.2f}" for v in mean_r))
    for c in CS:
        print(f"net c={c:<4} " + "  ".join(f"{v:6.3f}" for v in nets[str(c)]))
    print(f"net*-MAJ {d_maj:+.3f}")
    print(f"VOID {void}  EDGE {edge}  GATE {gate}")
    if void:
        print("VOID: thin.")
    elif gate:
        print("GO PRICE: interior k* at c=0.05. SEARCH has a priced budget.")
    else:
        print("STOP EDGE: k*=1 or 8. Always-stop or always-go under this ruler.")

    rec = dict(
        seed=args.seed, corpus=kind, path=str(path),
        n_win=len(windows), elapsed_s=round(time.time() - t0, 1),
        void=bool(void), edge=bool(edge), gate=bool(gate), n=n,
        fill_bag_maj=fm, KS=list(KS), hit=hit, mean_reads=mean_r,
        nets=nets, k_star=k_star, d_maj=d_maj,
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
