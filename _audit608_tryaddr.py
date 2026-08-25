"""608: sequential READ of 606 addresses. No Q. Frozen unique+PMI extract.

One-shot first/count lost to REACH (~0.8). This spends k READs and stops
on the first extract==held. Held never orders the list.

Orders: mention (F) | count_key (C) | shuffled (A)
k = 1,2,3. MAJ-P = majority of max-count place; BAG-MAJ report only. REACH = any address.

GATE  C@3 − max(F@1, C@1, MAJ-P) > 0.05
DIAG  C@3 − A@3   (order vs more tickets); BAG-MAJ report only
VOID  n < 40
607 not retrained. Extras still hidden until each READ.

    python _check608_tryaddr.py
    python _audit608_tryaddr.py --seed 1337 --corpus data/_tinystories_train.txt
    python _audit608_tryaddr.py --seed 8642 --corpus data/_tinystories_train.txt
    python _audit608_tryaddr.py --seed 2890 --corpus data/_tinystories_train.txt
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

OUT = Path("results/_stage608_tryaddr.json")


def hit_prefix(order, held, k):
    return int(any(pl["extract"] == held for pl in order[:k]))


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
    rnd = random.Random(args.seed + 61)
    t0 = time.time()
    print(f"608 tryaddr  {path}  {kind}  windows={len(windows)}", flush=True)

    n = o = m = b = 0
    acc = {name: [0, 0, 0] for name in ("F", "C", "A")}
    for lines in windows:
        for row in collect(lines, args, rng):
            n += 1
            held, places, bag0 = row["held"], row["places"], row["bag0"]
            o += int(any(pl["extract"] == held for pl in places))
            pc = max(places, key=lambda pl: pl["count_key"])
            m += int(pc["majority"] == held)
            b += int(Counter(bag0).most_common(1)[0][0] == held)
            mention = list(places)
            count = sorted(places, key=lambda pl: pl["count_key"], reverse=True)
            shuf = list(places)
            rnd.shuffle(shuf)
            for name, order in (("F", mention), ("C", count), ("A", shuf)):
                for k in (1, 2, 3):
                    acc[name][k - 1] += hit_prefix(order, held, k)

    def r(x):
        return x / n if n else 0.0

    def col(name):
        return [r(v) for v in acc[name]]

    f, c, a = col("F"), col("C"), col("A")
    fo, fm, fbm = r(o), r(m), r(b)
    oneshot = max(f[0], c[0], fm)
    d_search = c[2] - oneshot
    d_order = c[2] - a[2]
    void = n < 40
    gate = (not void) and d_search > 0.05
    print(
        f"n {n}  REACH {fo:.3f}  MAJ-P {fm:.3f}  BAG-MAJ {fbm:.3f}  "
        f"F {f[0]:.3f}/{f[1]:.3f}/{f[2]:.3f}  "
        f"C {c[0]:.3f}/{c[1]:.3f}/{c[2]:.3f}  "
        f"A {a[0]:.3f}/{a[1]:.3f}/{a[2]:.3f}"
    )
    print(f"C@3-oneshot {d_search:+.3f}  C@3-A@3 {d_order:+.3f}")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("VOID: thin.")
    elif gate:
        print("GO SEARCH: 3 READs harvest more than one-shot first/count/MAJ-P. Order is DIAG.")
    else:
        print("STOP: extra READs do not beat one-shot. 606 room stays oracle-only.")

    rec = dict(
        seed=args.seed, corpus=kind, path=str(path),
        n_win=len(windows), elapsed_s=round(time.time() - t0, 1),
        void=bool(void), gate=bool(gate), n=n,
        reach=fo, fill_maj_place=fm, fill_bag_maj=fbm,
        F=f, C=c, A=a, oneshot=oneshot,
        d_search=d_search, d_order=d_order,
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
